# train_classifier_augmented_v2.py - Versi perbaikan dengan hyperparameter optimal
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

# Konfigurasi yang dioptimasi
DATA_DIR = 'data_augmented'
NUM_CLASSES = 8
BATCH_SIZE = 16  # Lebih kecil untuk dataset yang lebih besar
NUM_EPOCHS = 20  # Lebih sedikit epoch
LEARNING_RATE = 0.00005  # Learning rate lebih kecil
MODEL_SAVE_PATH = 'models/best_classifier_augmented_v2.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB dengan perbaikan"""
    try:
        if image.mode == 'P':
            if 'transparency' in image.info:
                image = image.convert('RGBA')
            else:
                return image.convert('RGB')
        
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            if len(image.split()) == 4:
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image)
            return background
        elif image.mode in ['L', 'LA']:
            return image.convert('RGB')
        elif image.mode != 'RGB':
            return image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Warning: Error converting image mode {image.mode}: {e}")
        return image.convert('RGB')

# Data transforms yang lebih konservatif
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(convert_image_mode),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Lebih konservatif
        transforms.RandomHorizontalFlip(p=0.2),  # Probabilitas lebih rendah
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

def create_resnet50_classifier_v2(num_classes):
    """Model yang lebih konservatif untuk dataset augmentasi"""
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze lebih banyak layer untuk transfer learning yang stabil
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # Hanya unfreeze layer4 dan avgpool
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    for param in model_ft.avgpool.parameters():
        param.requires_grad = True
    
    # Classifier yang lebih sederhana
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.3),  # Dropout lebih kecil
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    for param in model_ft.fc.parameters():
        param.requires_grad = True
        
    print(f"Model created with {num_classes} classes")
    print(f"Trainable parameters: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")
        
    return model_ft

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
    patience = 8  # Patience lebih besar
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
                    if overfitting_gap > 0.08:  # Gap > 8%
                        print(f'Warning: Overfitting detected (gap: {overfitting_gap:.3f})')
                
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement')
                    break
                
        print()
        
        # Update scheduler
        if len(history['val_acc']) > 0:
            scheduler.step(history['val_acc'][-1])
        
        if patience_counter >= patience:
            break
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    return model, history

if __name__ == '__main__':
    print(f"Using device: {device}")
    print("Loading augmented dataset for optimized training...")
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Augmented dataset not found at {DATA_DIR}")
        print("Please run data_augmentation.py first.")
        exit(1)
    
    # Load dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True, 
                                                 num_workers=2)  # Reduced workers
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print("Classes detected:", image_datasets['train'].classes)
    print("Dataset sizes:", dataset_sizes)
    
    # Create optimized model
    model_ft = create_resnet50_classifier_v2(NUM_CLASSES)
    model_ft = model_ft.to(device)
    
    # Loss and optimizer dengan weight decay yang lebih besar
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Less aggressive label smoothing
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    
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
    epochs = range(len(history['train_acc']))
    gaps = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    plt.plot(epochs, gaps, label='Overfitting Gap', color='red', marker='d')
    plt.axhline(y=0.08, color='orange', linestyle='--', label='Warning Threshold')
    plt.title('Overfitting Control')
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc - Val Acc')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_performance_augmented_v2.png', dpi=300, bbox_inches='tight')
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
        print("⚠️  Some overfitting detected but controlled.")
    
    print(f"\nOptimized model saved to: {MODEL_SAVE_PATH}")
    print("Training with optimized augmented dataset completed!")

