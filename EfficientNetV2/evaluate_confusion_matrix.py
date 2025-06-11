# filepath: c:\Users\anggr\OneDrive\Desktop\car_retrieval_v3\evaluate_confusion_matrix.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import warnings

# Menekan peringatan PIL
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Import model classifier
try:
    from models.car_classifier import CarClassifier
except ImportError:
    print("Error: Tidak dapat mengimpor CarClassifier dari models/car_classifier.py")
    print("Pastikan file models/car_classifier.py ada dan definisi kelasnya benar.")
    exit()

# Konfigurasi
MODEL_PATH = 'models/best_classifier_efficientnetv2m.pt'
DATA_DIR = 'data'  # Sesuaikan dengan direktori dataset validasi
NUM_CLASSES = 8
BATCH_SIZE = 16

# Nama kelas
CLASS_NAMES = ['City_car', 'Commercial_Van', 'Hatchback', 'LCGC', 'MPV', 'Pickup_truck', 'SUV', 'Sedan']

# Deteksi device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fungsi konversi gambar (sama dengan train_classifier.py)
def convert_image_mode(image):
    """Konversi palette images dengan transparency ke RGB dengan penanganan lengkap"""
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

# Data transformasi untuk validasi
data_transforms = transforms.Compose([
    transforms.Lambda(convert_image_mode),
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Standard untuk efficientnetv2-m
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    # Load dataset validasi
    print("Loading validation dataset...")
    val_dir = os.path.join(DATA_DIR, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        print("Please ensure your validation dataset is in the 'data/val' directory.")
        exit(1)
    
    # Load dataset dengan menekan peringatan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val_dataset = datasets.ImageFolder(val_dir, data_transforms)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Validation dataset loaded with {len(val_dataset)} samples")
    
    # Cek class indices untuk memastikan urutan kelas sesuai
    class_to_idx = val_dataset.class_to_idx
    print("Class to index mapping:", class_to_idx)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Try using the direct EfficientNetV2 loading function first
        from models.car_classifier import load_trained_efficientnetv2_classifier
        model = load_trained_efficientnetv2_classifier(MODEL_PATH, num_classes=NUM_CLASSES, device=device)
        print("Model loaded successfully using direct EfficientNetV2 loader.")
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Trying CarClassifier wrapper with key mapping...")
        try:
            # Create CarClassifier and handle key mismatch
            model = CarClassifier(num_classes=NUM_CLASSES)
            saved_state_dict = torch.load(MODEL_PATH, map_location=device)
            
            # Map keys from saved model (with 'model.' prefix) to CarClassifier (with 'backbone.' prefix)
            mapped_state_dict = {}
            for key, value in saved_state_dict.items():
                # Replace 'model.' prefix with 'backbone.' prefix to match CarClassifier structure
                if key.startswith('model.'):
                    new_key = key.replace('model.', 'backbone.', 1)
                    mapped_state_dict[new_key] = value
                else:
                    # If no prefix, add 'backbone.' prefix
                    new_key = f"backbone.{key}"
                    mapped_state_dict[new_key] = value
            
            model.load_state_dict(mapped_state_dict)
            model = model.to(device)
            model.eval()
            print("Model loaded successfully using CarClassifier wrapper with key mapping.")
        except Exception as e2:
            print(f"Error loading model with both methods: {e2}")
            exit(1)
    
    # Evaluasi model dan buat confusion matrix
    print("Evaluating model and generating confusion matrix...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Hitung confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Tampilkan classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Simpan confusion matrix
    output_file = 'confusion_matrix_efficientnetv2m.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {output_file}")
    
    # Hitung metrik tambahan
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {per_class_accuracy[i]:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()