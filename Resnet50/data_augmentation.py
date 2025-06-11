# data_augmentation.py
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import shutil

def create_augmented_dataset(source_dir, target_dir, augment_factor=3):
    """
    Membuat dataset yang diperbanyak dengan augmentasi
    
    Args:
        source_dir: Direktori dataset asli
        target_dir: Direktori untuk dataset yang sudah diaugmentasi
        augment_factor: Berapa kali lipat data akan diperbanyak
    """
    
    # Buat direktori target jika belum ada
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # Copy struktur direktori
    for phase in ['train', 'val']:
        phase_source = os.path.join(source_dir, phase)
        phase_target = os.path.join(target_dir, phase)
        
        if not os.path.exists(phase_source):
            continue
            
        os.makedirs(phase_target, exist_ok=True)
        
        # Get class directories
        classes = [d for d in os.listdir(phase_source) 
                  if os.path.isdir(os.path.join(phase_source, d))]
        
        for class_name in classes:
            class_source = os.path.join(phase_source, class_name)
            class_target = os.path.join(phase_target, class_name)
            os.makedirs(class_target, exist_ok=True)
            
            # Get all images in class
            images = [f for f in os.listdir(class_source) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Processing {phase}/{class_name}: {len(images)} images")
            
            # Copy original images
            for img_name in images:
                src_path = os.path.join(class_source, img_name)
                dst_path = os.path.join(class_target, img_name)
                shutil.copy2(src_path, dst_path)
            
            # Untuk training data, buat augmentasi
            if phase == 'train':
                create_augmented_images(class_source, class_target, images, augment_factor)
            
    print(f"Dataset augmentation completed! Saved to: {target_dir}")

def create_augmented_images(source_dir, target_dir, image_files, augment_factor):
    """
    Membuat gambar-gambar augmentasi untuk satu kelas
    """
    
    for img_file in tqdm(image_files, desc="Augmenting images"):
        img_path = os.path.join(source_dir, img_file)
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Generate augmented versions
            for i in range(augment_factor):
                augmented_img = apply_random_augmentation(img)
                
                # Save augmented image
                name, ext = os.path.splitext(img_file)
                aug_filename = f"{name}_aug_{i+1}{ext}"
                aug_path = os.path.join(target_dir, aug_filename)
                
                augmented_img.save(aug_path)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

def apply_random_augmentation(img):
    """
    Menerapkan augmentasi random pada gambar
    """
    
    # Convert to numpy for some operations
    img_array = np.array(img)
    
    # List of augmentation techniques
    augmentations = [
        lambda x: random_rotation(x),
        lambda x: random_brightness(x),
        lambda x: random_contrast(x),
        lambda x: random_saturation(x),
        lambda x: random_hue_shift(x),
        lambda x: random_blur(x),
        lambda x: random_noise(x),
        lambda x: random_flip(x),
        lambda x: random_crop_and_resize(x),
        lambda x: random_perspective(x)
    ]
    
    # Apply 2-4 random augmentations
    num_augs = random.randint(2, 4)
    selected_augs = random.sample(augmentations, num_augs)
    
    result_img = img
    for aug_func in selected_augs:
        try:
            result_img = aug_func(result_img)
        except:
            continue
    
    return result_img

def random_rotation(img):
    """Rotasi random"""
    angle = random.uniform(-20, 20)
    return img.rotate(angle, fillcolor=(255, 255, 255))

def random_brightness(img):
    """Brightness random"""
    factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def random_contrast(img):
    """Contrast random"""
    factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def random_saturation(img):
    """Saturation random"""
    factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def random_hue_shift(img):
    """Hue shift menggunakan cv2"""
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    hue_shift = random.randint(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb)

def random_blur(img):
    """Blur random"""
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def random_noise(img):
    """Tambah noise random"""
    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def random_flip(img):
    """Flip horizontal random"""
    if random.random() > 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_crop_and_resize(img):
    """Crop random dan resize kembali"""
    width, height = img.size
    
    # Random crop parameters
    crop_factor = random.uniform(0.8, 0.95)
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)
    
    # Random position
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    
    # Crop and resize back
    cropped = img.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((width, height), Image.LANCZOS)

def random_perspective(img):
    """Perspective transform random"""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Random perspective points
    margin = 0.1
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.uniform(0, w*margin), random.uniform(0, h*margin)],
        [w - random.uniform(0, w*margin), random.uniform(0, h*margin)],
        [random.uniform(0, w*margin), h - random.uniform(0, h*margin)],
        [w - random.uniform(0, w*margin), h - random.uniform(0, h*margin)]
    ])
    
    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img_array, matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
    
    return Image.fromarray(result)

def analyze_dataset(data_dir):
    """Analisis dataset untuk melihat distribusi kelas"""
    print("\n=== DATASET ANALYSIS ===")
    
    for phase in ['train', 'val']:
        phase_dir = os.path.join(data_dir, phase)
        if not os.path.exists(phase_dir):
            continue
            
        print(f"\n{phase.upper()} SET:")
        classes = [d for d in os.listdir(phase_dir) 
                  if os.path.isdir(os.path.join(phase_dir, d))]
        
        total_images = 0
        for class_name in sorted(classes):
            class_dir = os.path.join(phase_dir, class_name)
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  {class_name}: {len(images)} images")
            total_images += len(images)
        
        print(f"  TOTAL: {total_images} images")

if __name__ == "__main__":
    # Konfigurasi
    source_dataset = "data"  # Dataset asli
    augmented_dataset = "data_augmented"  # Dataset yang sudah diaugmentasi
    augmentation_factor = 4  # Setiap gambar akan menjadi 4 versi augmentasi
    
    print("Starting data augmentation process...")
    print(f"Source: {source_dataset}")
    print(f"Target: {augmented_dataset}")
    print(f"Augmentation factor: {augmentation_factor}x")
    
    # Analisis dataset asli
    print("\nOriginal dataset:")
    analyze_dataset(source_dataset)
    
    # Buat dataset augmentasi
    create_augmented_dataset(source_dataset, augmented_dataset, augmentation_factor)
    
    # Analisis dataset augmentasi
    print("\nAugmented dataset:")
    analyze_dataset(augmented_dataset)
    
    print("\nData augmentation completed successfully!")
    print(f"You can now use '{augmented_dataset}' for training with improved performance.")

