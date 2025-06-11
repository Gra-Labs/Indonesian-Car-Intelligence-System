# train_classifier_augmented.py - Versi dengan dataset augmentasi
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

# Konfigurasi untuk dataset augmentasi
DATA_DIR = 'data_augmented'  # Gunakan dataset yang sudah diaugmentasi
NUM_CLASSES = 8
CLASS_NAMES = ['MPV', 'Sedan', 'Hatchback', 'SUV', 'City Car', 'LCGC', 'Pickup Truck', 'Commercial Van']
BATCH_SIZE = 32  # Bisa dinaikkan karena data lebih banyak
NUM_EPOCHS = 25  # Epoch bisa dikurangi karena data lebih banyak
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/best_classifier_augmented.pt'

# Deteksi device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB untuk menghindari warning PIL"""
    if image.mode == 'P':
        if 'transparency' in image.info:
            image = image.convert('RGBA')
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        else:
            return image.convert('RGB')
    elif image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

# Data transforms - lebih sederhana karena augmentasi sudah dilakukan offline
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(convert_image_mode),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Kurangi karena sudah ada augmentasi
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

def create_resnet50_classifier(num_classes):
    """Buat model ResNet50 dengan classifier yang dioptimasi"""
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # Unfreeze layer3 dan layer4 untuk fine-tuning yang lebih dalam
    for param in model_ft.layer3.parameters():
        param.requires_grad = True
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    for param in model_ft.avgpool.parameters():
        param.requires_grad = True
    
    # Classifier dengan batch normalization
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    for param in model_ft.fc.parameters():
        param.requires_grad = True
        
    print(f"Model created with {num_classes} classes")
    print(f"Trainable parameters: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")
        
    return model_ft

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """Fungsi training yang dioptimasi untuk dataset besar"""
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5  # Patience lebih ketat karena data lebih banyak
    patience_counter = 0
    min_delta = 0.005  # Threshold improvement
    
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
            
            # Progress bar untuk batch
            total_batches = len(dataloaders[phase])
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Print progress setiap 20% batch
                if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                    current_acc = running_corrects.double() / ((batch_idx + 1) * inputs.size(0))
                    print(f'  {phase} [{batch_idx+1}/{total_batches}] - Acc: {current_acc:.4f}')
                
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
                
                # Overfitting detection
                if len(history['train_acc']) > 0:
                    train_acc = history['train_acc'][-1]
                    overfitting_gap = train_acc - epoch_acc.item()
                    if overfitting_gap > 0.1:
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
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    return model, history

if __name__ == '__main__':
    print(f"Using device: {device}")
    print("Loading augmented dataset...")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    # Check if augmented dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Augmented dataset not found at {DATA_DIR}")
        print("Please run data_augmentation.py first to create augmented dataset.")
        exit(1)
    
    # Load dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print("Classes detected:", image_datasets['train'].classes)
    print("Dataset sizes:", dataset_sizes)
    
    # Create model
    model_ft = create_resnet50_classifier(NUM_CLASSES)
    model_ft = model_ft.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Differential learning rates
    backbone_params = []
    classifier_params = []
    
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    optimizer_ft = optim.Adam([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': LEARNING_RATE}       # Higher LR for classifier
    ], weight_decay=1e-4)
    
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', 
                                                     factor=0.5, patience=3)
    
    # Start training
    print("Starting training with augmented dataset...")
    model_trained, history = train_model(model_ft, criterion, optimizer_ft, 
                                       exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Plot overfitting gap
    epochs = range(len(history['train_acc']))
    gaps = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    plt.plot(epochs, gaps, label='Overfitting Gap', color='red')
    plt.axhline(y=0.1, color='orange', linestyle='--', label='Warning Threshold')
    plt.title('Overfitting Gap per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc - Val Acc')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_performance_augmented.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining Log:")
    for epoch in range(len(history['train_loss'])):
        gap = history['train_acc'][epoch] - history['val_acc'][epoch]
        print(f"Epoch {epoch}: Train Loss={history['train_loss'][epoch]:.4f}, "
              f"Train Acc={history['train_acc'][epoch]:.4f}, "
              f"Val Loss={history['val_loss'][epoch]:.4f}, "
              f"Val Acc={history['val_acc'][epoch]:.4f}, Gap={gap:.3f}")
    
    print(f"\nBest model saved to: {MODEL_SAVE_PATH}")
    print("Training with augmented dataset completed!")

