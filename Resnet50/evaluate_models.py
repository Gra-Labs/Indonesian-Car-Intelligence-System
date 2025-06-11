# evaluate_models.py - Script untuk membandingkan performa model
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Konfigurasi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8

def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB"""
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

def create_resnet50_original():
    """Buat model asli (tanpa augmentasi)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.avgpool.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    
    return model

def create_resnet50_augmented():
    """Buat model dengan augmentasi (v1)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.avgpool.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    
    return model

def create_resnet50_augmented_v2():
    """Buat model dengan augmentasi (v2 - optimized)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.avgpool.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    
    return model

def evaluate_model(model, dataloader, class_names):
    """Evaluasi model dan return predictions, labels"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return all_preds, all_labels, accuracy

def confusion_matrix(y_true, y_pred, num_classes):
    """Buat confusion matrix manual"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def classification_report_simple(y_true, y_pred, class_names):
    """Buat classification report sederhana"""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    
    total_correct = 0
    total_samples = len(y_true)
    
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        total_correct += tp
        
        print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
    
    accuracy = total_correct / total_samples
    print("-" * 65)
    print(f"{'Accuracy':<15} {'':<10} {'':<10} {accuracy:<10.3f} {total_samples:<10}")

def main():
    print("=== EVALUASI DAN PERBANDINGAN MODEL ===")
    print(f"Using device: {device}")
    
    # Data transforms untuk evaluasi
    data_transform = transforms.Compose([
        transforms.Lambda(convert_image_mode),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset (menggunakan dataset asli untuk evaluasi fair)
    val_dataset = datasets.ImageFolder('data/val', data_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    class_names = val_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Validation samples: {len(val_dataset)}")
    
    results = {}
    
    # Evaluasi Model 1: Tanpa Augmentasi
    print("\n1. EVALUATING MODEL WITHOUT AUGMENTATION...")
    model1_path = 'models/best_classifier_weight.pt'
    if os.path.exists(model1_path):
        model1 = create_resnet50_original().to(device)
        model1.load_state_dict(torch.load(model1_path, map_location=device))
        
        preds1, labels1, acc1 = evaluate_model(model1, val_dataloader, class_names)
        results['No Augmentation'] = {
            'accuracy': acc1,
            'preds': preds1,
            'labels': labels1
        }
        
        print(f"Accuracy: {acc1:.4f} ({acc1*100:.2f}%)")
        
        # Plot confusion matrix
        plot_confusion_matrix(labels1, preds1, class_names, 
                            'Model Without Augmentation', 
                            'confusion_matrix_no_aug.png')
        
        # Classification report
        print("\nClassification Report:")
        classification_report_simple(labels1, preds1, class_names)
    else:
        print(f"Model not found: {model1_path}")
    
    # Evaluasi Model 2: Dengan Augmentasi (v1)
    print("\n2. EVALUATING MODEL WITH AUGMENTATION (v1)...")
    model2_path = 'models/best_classifier_augmented.pt'
    if os.path.exists(model2_path):
        model2 = create_resnet50_augmented().to(device)
        model2.load_state_dict(torch.load(model2_path, map_location=device))
        
        preds2, labels2, acc2 = evaluate_model(model2, val_dataloader, class_names)
        results['With Augmentation v1'] = {
            'accuracy': acc2,
            'preds': preds2,
            'labels': labels2
        }
        
        print(f"Accuracy: {acc2:.4f} ({acc2*100:.2f}%)")
        
        # Plot confusion matrix
        plot_confusion_matrix(labels2, preds2, class_names, 
                            'Model With Augmentation v1', 
                            'confusion_matrix_aug_v1.png')
        
        # Classification report
        print("\nClassification Report:")
        classification_report_simple(labels2, preds2, class_names)
    else:
        print(f"Model not found: {model2_path}")
    
    # Evaluasi Model 3: Dengan Augmentasi (v2 - optimized)
    print("\n3. EVALUATING MODEL WITH AUGMENTATION (v2 - Optimized)...")
    model3_path = 'models/best_classifier_augmented_v2.pt'
    if os.path.exists(model3_path):
        model3 = create_resnet50_augmented_v2().to(device)
        model3.load_state_dict(torch.load(model3_path, map_location=device))
        
        preds3, labels3, acc3 = evaluate_model(model3, val_dataloader, class_names)
        results['With Augmentation v2'] = {
            'accuracy': acc3,
            'preds': preds3,
            'labels': labels3
        }
        
        print(f"Accuracy: {acc3:.4f} ({acc3*100:.2f}%)")
        
        # Plot confusion matrix
        plot_confusion_matrix(labels3, preds3, class_names, 
                            'Model With Augmentation v2', 
                            'confusion_matrix_aug_v2.png')
        
        # Classification report
        print("\nClassification Report:")
        classification_report_simple(labels3, preds3, class_names)
    else:
        print(f"Model not found: {model3_path}")
    
    # Perbandingan hasil
    print("\n=== COMPARISON RESULTS ===")
    if len(results) >= 2:
        acc_no_aug = results['No Augmentation']['accuracy']
        
        print(f"Model WITHOUT Augmentation: {acc_no_aug:.4f} ({acc_no_aug*100:.2f}%)")
        
        model_names = []
        accuracies = []
        colors = []
        
        model_names.append('No\nAugmentation')
        accuracies.append(acc_no_aug * 100)
        colors.append('lightcoral')
        
        for key in results.keys():
            if 'Augmentation' in key:
                acc = results[key]['accuracy']
                print(f"Model {key}: {acc:.4f} ({acc*100:.2f}%)")
                
                improvement = acc - acc_no_aug
                improvement_pct = (improvement / acc_no_aug) * 100
                print(f"  Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
                
                model_names.append(key.replace('With ', '').replace(' ', '\n'))
                accuracies.append(acc * 100)
                colors.append('lightgreen' if improvement > 0 else 'orange')
        
        # Plot perbandingan
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
        plt.ylabel('Accuracy (%)')
        plt.title('Model Performance Comparison - All Versions')
        plt.ylim(0, 100)
        
        # Tambahkan nilai di atas bar
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_all.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_acc = results[best_model]['accuracy']
        print(f"\n🏆 BEST MODEL: {best_model} with {best_acc:.4f} ({best_acc*100:.2f}%) accuracy")
        
    else:
        print("Could not compare models - missing model files.")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()

