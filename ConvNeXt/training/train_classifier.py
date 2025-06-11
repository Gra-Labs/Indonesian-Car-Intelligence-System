# train_classifier_augmented_v2.py - Versi perbaikan dengan hyperparameter optimal dan ConvNeXt Base
import warnings
# Menekan peringatan PIL di awal
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Base_Weights, EfficientNet_V2_M_Weights # <-- Impor bobot ConvNeXt Base dan EfficientNet
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

# Konfigurasi yang dioptimasi
DATA_DIR = 'data'
NUM_CLASSES = 8
BATCH_SIZE = 16 #Lebih kecil untuk dataset yang lebih besar
NUM_EPOCHS = 20 #Lebih sedikit epoch
LEARNING_RATE = 0.00005 #Learning rate lebih kecil
MODEL_SAVE_PATH = 'models/best_classifier_augmented_v2_convnext_base.pt' # <-- Nama file model baru

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB dengan penanganan lebih lengkap"""
    try:
        # Tangani kasus palette dengan transparency secara eksplisit
        if image.mode == 'P':
            # Cek transparansi
            if 'transparency' in image.info:
                # Konversi langsung ke RGBA untuk menangani transparency dengan benar
                return image.convert('RGBA').convert('RGB')
            else:
                return image.convert('RGB')
                
        # Jika gambar sudah RGBA, konversi ke RGB dengan background putih
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Compose dengan alpha channel
            alpha = image.split()[3]
            background.paste(image, mask=alpha)
            return background
            
        # Tangani kasus gambar grayscale
        elif image.mode in ['L', 'LA']:
            return image.convert('RGB')
            
        # Tangani semua mode lain
        elif image.mode != 'RGB':
            return image.convert('RGB')
            
        return image
    except Exception as e:
        print(f"Warning: Error converting image mode {image.mode}: {e}")
        # Fallback ke konversi RGB sederhana
        try:
            return image.convert('RGB')
        except:
            print(f"Critical error: Could not convert {image.mode} to RGB")
            return image

# Data transforms yang lebih konservatif
# EfficientNetV2-M secara default dilatih pada ukuran 384x384. Sesuaikan jika perlu.
# Namun, 224x224 masih dapat diterima dan lebih hemat memori.
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(convert_image_mode),
        transforms.Resize(256),  # Resize awal
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Crop dan resize ke 224x224
        transforms.RandomHorizontalFlip(p=0.2),  # Probabilitas lebih rendah
        # Tambahan augmentasi ringan
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(convert_image_mode),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- GANTI FUNGSI create_resnet50_classifier_v2 dengan ConvNeXt Base ---
def create_convnext_base_classifier(num_classes):
    """Model ConvNeXt Base untuk klasifikasi mobil"""
    # Muat ConvNeXt Base pre-trained pada ImageNet
    model_ft = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    
    # Freeze sebagian besar layer untuk transfer learning yang stabil
    # Pada ConvNeXt, feature extractor-nya adalah 'features'
    for param in model_ft.features.parameters():
        param.requires_grad = False
    
    # Opsional: Unfreeze blok terakhir untuk fine-tuning
    # ConvNeXt Base memiliki 4 stages, kita bisa unfreeze stage terakhir
    for param in model_ft.features[-1].parameters():
        param.requires_grad = True
    
    # Determine the input feature dimension from the original Linear layer
    num_ftrs = model_ft.classifier[2].in_features  # Linear layer's in_features (index 2)
    
    # Replace the classifier while preserving the LayerNorm2d and Flatten layers
    model_ft.classifier = nn.Sequential(
        model_ft.classifier[0],  # Keep LayerNorm2d
        model_ft.classifier[1],  # Keep Flatten
        nn.Dropout(0.3),  # Dropout pertama
        nn.Linear(num_ftrs, 512),  # Layer yang lebih besar karena ConvNeXt memiliki representasi yang lebih kaya
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    # Pastikan parameter dari classifier head baru dapat dilatih
    for param in model_ft.classifier.parameters():
        param.requires_grad = True
        
    print(f"ConvNeXt Base model created with {num_classes} classes")
    print(f"Trainable parameters: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")
        
    return model_ft

# Fungsi train_model_v2 (tetap sama, tidak perlu perubahan besar)
def train_model_v2(model, criterion, optimizer, scheduler, num_epochs=20):
    """Training dengan fokus pada stabilitas"""
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 8 # Patience lebih besar
    patience_counter = 0
    min_delta = 0.001
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Set num_workers=0 jika mengalami masalah di Windows
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping yang lebih agresif
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'val':
                # Perbarui scheduler berdasarkan akurasi validasi
                scheduler.step(epoch_acc) # <-- PENTING: Update scheduler di sini untuk ReduceLROnPlateau

                if epoch_acc > best_acc + min_delta:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f'New best model! Acc: {best_acc:.4f}')
                else:
                    patience_counter += 1
                
                # Overfitting detection yang lebih ketat
                if len(history['train_acc']) > 0:
                    train_acc = history['train_acc'][-1]
                    overfitting_gap = train_acc - epoch_acc.item()
                    if overfitting_gap > 0.08: # Gap > 8%
                        print(f'Warning: Overfitting detected (gap: {overfitting_gap:.3f})')
                
                # Cek early stopping setelah update scheduler
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement')
                    break # Keluar dari loop phase (train/val)
            
        print() # Newline setelah setiap epoch train/val loop
        
        if patience_counter >= patience: # Cek lagi di luar loop phase untuk keluar dari loop epoch
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    return model, history

if __name__ == '__main__':
    print(f"Using device: {device}")
    print("Loading augmented dataset for optimized training...")
    
    def verify_images():
        """Verifikasi semua gambar dalam dataset dan lapor jika ada masalah"""
        print("Memverifikasi gambar dalam dataset...")
        problem_images = []
        
        for phase in ['train', 'val']:
            data_dir = os.path.join(DATA_DIR, phase)
            if not os.path.exists(data_dir):
                print(f"Warning: Directory {data_dir} not found")
                continue
                
            for class_name in os.listdir(data_dir):
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        with Image.open(img_path) as img:
                            if img.mode == 'P' and 'transparency' in img.info:
                                problem_images.append((img_path, f"Palette with transparency ({img.mode})"))
                    except Exception as e:
                        problem_images.append((img_path, str(e)))
        
        if problem_images:
            print(f"Ditemukan {len(problem_images)} gambar bermasalah:")
            for path, issue in problem_images[:10]:  # Tampilkan 10 contoh pertama
                print(f" - {path}: {issue}")
            if len(problem_images) > 10:
                print(f"   ...dan {len(problem_images)-10} lainnya.")
            print("Gambar-gambar ini akan dikonversi secara otomatis oleh fungsi convert_image_mode")
        else:
            print("Tidak ada gambar bermasalah ditemukan.")
    
    # Panggil fungsi verifikasi di sini
    verify_images()
    
    # Asumsikan CLASS_NAMES sudah didefinisikan secara global
    CLASS_NAMES = ['City_car', 'Commercial_Van', 'Hatchback', 'LCGC', 'MPV', 'Pickup_truck', 'SUV', 'Sedan']
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Augmented dataset not found at {DATA_DIR}")
        print("Please ensure your dataset is in the 'data' directory.")
        exit(1)
      # Pastikan semua peringatan ditekan sebelum memuat dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load dataset
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print("Classes detected:", image_datasets['train'].classes)
    print("Dataset sizes:", dataset_sizes)
      # Create optimized model (GANTI PEMANGGILAN FUNGSI DI SINI)
    model_ft = create_convnext_base_classifier(NUM_CLASSES) # <-- Panggil fungsi ConvNeXt Base
    model_ft = model_ft.to(device)
    
    # Loss and optimizer dengan weight decay yang lebih besar
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=5e-4) # <-- Coba AdamW
    
    # Scheduler yang lebih responsif
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', 
                                                      factor=0.7, patience=2)
    
    print("Starting optimized training with augmented dataset...")
    model_trained, history = train_model_v2(model_ft, criterion, optimizer_ft, 
                                            exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Loss per Epoch (Optimized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    plt.title('Accuracy per Epoch (Optimized)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Plot overfitting gap
    epochs_plotted = range(len(history['train_acc']))
    gaps = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    plt.plot(epochs_plotted, gaps, label='Overfitting Gap', color='red', marker='d')
    plt.axhline(y=0.08, color='orange', linestyle='--', label='Warning Threshold')
    plt.title('Overfitting Control')
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc - Val Acc')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_performance_augmented_v2_convnext_base.png', dpi=300, bbox_inches='tight') # <-- Nama file plot baru
    plt.show()
    
    print("\nOptimized Training Log:")
    for epoch in range(len(history['train_loss'])):
        gap = history['train_acc'][epoch] - history['val_acc'][epoch]
        print(f"Epoch {epoch}: Train Loss={history['train_loss'][epoch]:.4f}, "
              f"Train Acc={history['train_acc'][epoch]:.4f}, "
              f"Val Loss={history['val_loss'][epoch]:.4f}, "
              f"Val Acc={history['val_acc'][epoch]:.4f}, Gap={gap:.3f}")
    
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_gap = final_train_acc - final_val_acc
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Overfitting Gap: {final_gap:.3f}")
    
    if final_gap < 0.1:
        print("✅ Good generalization achieved!")
    else:
        print("⚠️  Some overfitting detected but controlled.")
    
    print(f"\nOptimized model saved to: {MODEL_SAVE_PATH}")
    print("Training with optimized augmented dataset completed!")