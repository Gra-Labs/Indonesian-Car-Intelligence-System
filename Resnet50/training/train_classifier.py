# train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

# Definisikan semua konfigurasi, data_transforms, create_resnet50_classifier, train_model
# di luar blok if __name__ == '__main__': tapi pastikan tidak ada eksekusi langsung.

# 1. Konfigurasi
DATA_DIR = 'data'
NUM_CLASSES = 8
CLASS_NAMES = ['MPV', 'Sedan', 'Hatchback', 'SUV', 'City Car', 'LCGC', 'Pickup Truck', 'Commercial Van']
BATCH_SIZE = 16  # Batch size lebih kecil untuk regularisasi
NUM_EPOCHS = 30  # Lebih banyak epoch karena learning rate kecil
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/best_classifier_weight.pt'

# Deteksi device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fungsi untuk mengkonversi palette images dengan transparency ke RGB

def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB untuk menghindari warning PIL"""
    # Tangani semua kasus palette image dengan transparency
    if image.mode == 'P':
        if 'transparency' in image.info:
            # Konversi palette image dengan transparency ke RGBA dulu
            image = image.convert('RGBA')
            # Buat background putih untuk transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Alpha channel sebagai mask
            return background
        else:
            # Konversi palette tanpa transparency langsung ke RGB
            return image.convert('RGB')
    elif image.mode == 'RGBA':
        # Konversi RGBA ke RGB dengan background putih
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    elif image.mode != 'RGB':
        # Konversi mode lain ke RGB
        return image.convert('RGB')
    return image

# Data transforms dengan augmentasi yang lebih agresif untuk mengurangi overfitting

data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(convert_image_mode),  # Tambahkan konversi mode gambar
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Lebih agresif
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Rotasi random
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Random erasing
    ]),
    'val': transforms.Compose([
        transforms.Lambda(convert_image_mode),  # Tambahkan konversi mode gambar
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Definisi fungsi untuk membuat model dengan dropout untuk anti-overfitting
def create_resnet50_classifier(num_classes):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Strategi transfer learning yang lebih konservatif
    # Freeze semua layer kecuali beberapa layer terakhir
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # Unfreeze hanya layer4 untuk fine-tuning yang lebih terbatas
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
        
    # Unfreeze avgpool dan fc
    for param in model_ft.avgpool.parameters():
        param.requires_grad = True
    
    # Ganti classifier dengan yang memiliki dropout
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout layer untuk regularisasi
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),  # Dropout kedua
        nn.Linear(512, num_classes)
    )
    
    # Pastikan classifier layer bisa dilatih
    for param in model_ft.fc.parameters():
        param.requires_grad = True
        
    print(f"Model created with {num_classes} classes")
    print(f"Trainable parameters: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")
        
    return model_ft

# Definisi fungsi pelatihan (ini juga boleh di luar main)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    # History untuk menyimpan loss dan accuracy
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    patience = 7  # Early stopping patience - sedikit lebih toleran
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement threshold
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Setiap epoch memiliki fase training dan validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model ke training mode
            else:
                model.eval()   # Set model ke evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterasi melalui data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history hanya jika di training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize hanya jika di training
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping untuk stabilitas
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                # Statistik
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Simpan ke history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy model jika ini adalah model terbaik
            if phase == 'val':
                # Early stopping berdasarkan improvement yang signifikan
                if epoch_acc > best_acc + min_delta:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0  # Reset patience
                    print(f'Model terbaik baru! Acc: {best_acc:.4f}')
                else:
                    patience_counter += 1
                
                # Cek overfitting: jika train acc >> val acc
                train_acc = history['train_acc'][-1] if history['train_acc'] else 0
                overfitting_gap = train_acc - epoch_acc.item()
                if overfitting_gap > 0.15:  # Gap > 15%
                    print(f'Warning: Possible overfitting detected (gap: {overfitting_gap:.3f})')
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement')
                    break
                
        print()
        
        # Update scheduler berdasarkan validation accuracy
        if 'val' in [x for x in ['train', 'val']]:
            scheduler.step(history['val_acc'][-1] if history['val_acc'] else 0)
        
        # Break dari loop epoch juga jika early stopping
        if patience_counter >= patience:
            break
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Simpan model terbaik
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    return model, history

# --- Ini adalah bagian KRITIS yang harus berada di dalam if __name__ == '__main__': ---
if __name__ == '__main__':
    print(f"Menggunakan device: {device}")
    print("Memuat dataset...")
    
    # Suppress PIL warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    # 3. Muat Dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # Set num_workers ke 0 untuk debugging awal jika masalah tetap ada, lalu tingkatkan
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print("Kelas yang terdeteksi di dataset:", image_datasets['train'].classes)
    print("Ukuran dataset:", dataset_sizes)

    # 4. Definisi Model (ResNet50)
    model_ft = create_resnet50_classifier(NUM_CLASSES)
    model_ft = model_ft.to(device)

    # 5. Fungsi Loss dan Optimizer dengan weight decay untuk regularisasi
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing untuk regularisasi
    # Weight decay sebagai L2 regularization
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=1e-4)
    # Scheduler yang lebih gentle
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.5, patience=3)

    # 7. Jalankan Pelatihan
    print("Memulai pelatihan model klasifikasi mobil...")
    model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                       num_epochs=NUM_EPOCHS)

    # 8. Visualisasi Hasil Pelatihan
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()

    print("\nTraining Log:")
    # Hanya print untuk epoch yang benar-benar dijalankan
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch}: Train Loss={history['train_loss'][epoch]:.4f}, Train Acc={history['train_acc'][epoch]:.4f}, "
              f"Val Loss={history['val_loss'][epoch]:.4f}, Val Acc={history['val_acc'][epoch]:.4f}")

    print(f"\nModel terbaik disimpan di: {MODEL_SAVE_PATH}")
    print("Jangan lupa sertakan `training_performance.png` dan `training log` ini di laporan Anda.")
