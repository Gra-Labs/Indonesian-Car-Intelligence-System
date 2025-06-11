import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm # Untuk progress bar
import time # Untuk estimasi waktu

# Impor model klasifikasi mobil Anda dari file yang sesuai
try:
    from models.car_classifier import CarClassifier, load_trained_convnext_classifier
except ImportError:
    print("Error: Tidak dapat mengimpor CarClassifier dari models/car_classifier.py")
    print("Pastikan file models/car_classifier.py ada dan definisi kelasnya benar.")
    exit()

# --- Konfigurasi ---
YOLO_MODEL_PATH = 'best.pt' # Pastikan ini adalah jalur ke model YOLOv11x Anda
# Ganti dengan jalur bobot classifier Anda yang terbaru (EfficientNetV2-M)
CLASSIFIER_MODEL_PATH = 'models/best_classifier_augmented_v2_convnext_base.pt' 
VIDEO_PATH = 'traffic_test.mp4' # Video demo yang akan digunakan
OUTPUT_VIDEO_PATH = 'output_processed_video.mp4' # Jalur untuk video output

# Kelas yang dideteksi oleh YOLO (hanya 'car' dalam kasus ini)
YOLO_CLASS_NAMES = ['car']

# Konfigurasi untuk model klasifikasi mobil Anda
NUM_CLASSES_CLASSIFIER = 8
CLASSIFIER_CLASS_NAMES = ['City_car', 'Commercial_Van', 'Hatchback', 'LCGC', 'MPV', 'Pickup_truck', 'SUV', 'Sedan'] 

# Threshold untuk deteksi YOLO
CONFIDENCE_THRESHOLD = 0.5

# Ukuran input yang diharapkan oleh model klasifikasi Anda
CLASSIFIER_INPUT_SIZE = (224, 224) # Sesuaikan jika Anda melatih dengan ukuran input berbeda (misal 384x384 untuk EfficientNetV2-M)

# Normalisasi yang sama seperti saat pelatihan model klasifikasi
CLASSIFIER_NORM_MEAN = [0.485, 0.456, 0.406]
CLASSIFIER_NORM_STD = [0.229, 0.224, 0.225]

# Deteksi device (GPU jika tersedia, CPU jika tidak)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {device}")

# --- Inisialisasi Model YOLO ---
print(f"Memuat model YOLOv11x dari: {YOLO_MODEL_PATH}")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("Model YOLOv11x berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model YOLO: {e}")
    print("Pastikan file best.pt ada dan Ultralytics YOLO sudah terinstal.")
    exit()

# --- Inisialisasi Model Klasifikasi Mobil (akan dimuat sesuai mode) ---
classifier_model = None # Inisialisasi sebagai None
if os.path.exists(CLASSIFIER_MODEL_PATH):
    print(f"Memuat model klasifikasi mobil dari: {CLASSIFIER_MODEL_PATH}")
    try:
        # Method 1: Try using the direct loading function
        print("Mencoba metode loading langsung...")
        classifier_model = load_trained_convnext_classifier(
            CLASSIFIER_MODEL_PATH, 
            num_classes=NUM_CLASSES_CLASSIFIER, 
            device=device
        )
        print("Model klasifikasi mobil berhasil dimuat dengan metode langsung.")
    except Exception as e1:
        print(f"Metode loading langsung gagal: {e1}")
        print("Mencoba metode loading manual...")
        
        try:
            # Method 2: Manual loading with CarClassifier class
            classifier_model = CarClassifier(num_classes=NUM_CLASSES_CLASSIFIER).to(device)
            
            # Load the saved state dict
            saved_state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)
            
            # Try direct loading first
            try:
                classifier_model.load_state_dict(saved_state_dict, strict=True)
                print("Model berhasil dimuat dengan struktur yang tepat.")
            except RuntimeError as e2:
                print(f"Loading langsung gagal: {e2}")
                print("Mencoba mapping kunci...")
                
                # Create a new state dict with correct key mapping
                new_state_dict = {}
                for key, value in saved_state_dict.items():
                    if key.startswith('model.'):
                        # Remove 'model.' prefix if it exists
                        new_key = key[6:]  # Remove 'model.'
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                # Try loading with mapped keys
                missing_keys, unexpected_keys = classifier_model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Peringatan: Beberapa layers tidak dimuat: {missing_keys[:3]}...")
                if unexpected_keys:
                    print(f"Peringatan: Beberapa keys tidak diharapkan: {unexpected_keys[:3]}...")
            
            classifier_model.eval() # Set model ke mode evaluasi
            print("Model klasifikasi mobil berhasil dimuat dengan metode manual.")
            
        except Exception as e2:
            print(f"Error saat memuat model klasifikasi dengan metode manual: {e2}")
            print("Model klasifikasi tidak akan digunakan.")
            classifier_model = None # Set kembali ke None jika gagal dimuat
else:
    print(f"File model klasifikasi tidak ditemukan di {CLASSIFIER_MODEL_PATH}.")
    print("Model klasifikasi tidak akan digunakan.")


# --- Pilihan Mode ---
print("\nPilih mode operasi:")
print("1. Tampilkan video dengan deteksi YOLO (real-time)")
print("2. Proses video (YOLO + Klasifikasi) dan simpan ke file (tanpa tampilan)")
choice = input("Masukkan pilihan (1/2): ")

if choice not in ['1', '2']:
    print("Pilihan tidak valid. Keluar.")
    exit()

# --- Membaca Video ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if choice == '2':
    # Konfigurasi VideoWriter untuk menyimpan output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec untuk .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Gagal membuat file output video di {OUTPUT_VIDEO_PATH}")
        exit()
    print(f"Memulai pemrosesan video dan menyimpan ke {OUTPUT_VIDEO_PATH}...")
    start_time = time.time()
    pbar = tqdm(total=total_frames, desc="Memproses Frame")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Langkah 1: Deteksi Objek dengan YOLO ---
    results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Iterasi melalui setiap deteksi
    for r in results:
        boxes = r.boxes
        for box in boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if YOLO_CLASS_NAMES[int(cls)] == 'car':
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)

                car_crop = frame[y1:y2, x1:x2]

                if car_crop.shape[0] == 0 or car_crop.shape[1] == 0:
                    continue

                predicted_class = "N/A" # Default label
                if classifier_model is not None: # Hanya klasifikasi jika model dimuat
                    try:
                        # --- Langkah 3: Preprocessing untuk Klasifikasi ---
                        car_crop_resized = cv2.resize(car_crop, CLASSIFIER_INPUT_SIZE)
                        car_crop_rgb = cv2.cvtColor(car_crop_resized, cv2.COLOR_BGR2RGB)
                        input_tensor = torch.from_numpy(car_crop_rgb).permute(2, 0, 1).float() / 255.0
                        normalize = transforms.Normalize(mean=CLASSIFIER_NORM_MEAN, std=CLASSIFIER_NORM_STD)
                        input_tensor = normalize(input_tensor)
                        input_batch = input_tensor.unsqueeze(0).to(device)

                        # --- Langkah 4: Klasifikasi Mobil ---
                        with torch.no_grad():
                            output = classifier_model(input_batch)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            
                            # Only show prediction if confidence is above threshold
                            if confidence.item() > 0.3:  # 30% confidence threshold
                                predicted_class = f"{CLASSIFIER_CLASS_NAMES[predicted_idx.item()]} ({confidence.item():.2f})"
                            else:
                                predicted_class = "Uncertain"
                                
                    except Exception as e:
                        if frame_count == 0:  # Only print error once
                            print(f"Error saat klasifikasi: {e}")
                        predicted_class = "Klasifikasi Gagal"

                # --- Tampilkan Hasil ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{predicted_class} | YOLO ({conf:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if choice == '1':
        cv2.imshow('Car Retrieval System Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif choice == '2':
        out.write(frame)
        pbar.update(1) # Update progress bar
        frame_count += 1
        if frame_count % (fps * 5) == 0: # Perbarui estimasi setiap 5 detik video
            elapsed_time = time.time() - start_time
            if frame_count > 0:
                frames_per_sec_processed = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                estimated_time_remaining = remaining_frames / frames_per_sec_processed
                print(f"Estimasi Waktu Tersisa: {estimated_time_remaining:.0f} detik")

# --- Cleanup ---
cap.release()
if choice == '2':
    out.release()
    pbar.close()
    print(f"\nVideo hasil proses disimpan di: {OUTPUT_VIDEO_PATH}")
cv2.destroyAllWindows()
print("Sistem Pengambilan Mobil selesai. Tekan Enter untuk keluar dari konsol.")