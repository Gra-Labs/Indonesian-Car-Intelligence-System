import os
import shutil

# --- Konfigurasi ---
# Direktori utama tempat data.yaml dan folder train/valid/test berada
# Ganti ini dengan jalur yang benar di sistem Anda
BASE_DATA_DIR = '.' # Jika Anda menjalankan skrip ini dari direktori yang sama dengan data.yaml dan folder data

# Nama file YAML Anda
DATA_YAML_NAME = 'data.yaml'

# Class ID untuk 'car' di dataset asli (berdasarkan names: ['bus', 'car', 'motorbike', 'truck'])
# Jika 'car' adalah indeks ke-1, maka ID-nya adalah 1
ORIGINAL_CAR_CLASS_ID = 1

# Class ID baru untuk 'car' setelah filtering (karena hanya ada 1 kelas, ID-nya harus 0)
NEW_CAR_CLASS_ID = 0

# --- Fungsi untuk Memfilter Label ---
def filter_yolo_labels_for_car(base_dir, split_name, original_car_id, new_car_id):
    """
    Memfilter file label YOLO untuk hanya menyimpan anotasi 'car'
    dan mengubah class_id 'car' menjadi new_car_id.

    Args:
        base_dir (str): Direktori utama dataset (misal: './').
        split_name (str): Nama split (misal: 'train', 'valid', 'test').
        original_car_id (int): Class ID 'car' di dataset asli.
        new_car_id (int): Class ID baru untuk 'car' (setelah difilter).
    """
    input_labels_dir = os.path.join(base_dir, split_name, 'labels')
    output_labels_dir = os.path.join(base_dir, split_name, 'labels_car_only')

    if not os.path.exists(input_labels_dir):
        print(f"Direktori label input tidak ditemukan: {input_labels_dir}. Lewati {split_name}.")
        return False

    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"\nMemproses label di {split_name}...")
    processed_files_count = 0

    for filename in os.listdir(input_labels_dir):
        if filename.endswith('.txt'):
            input_filepath = os.path.join(input_labels_dir, filename)
            output_filepath = os.path.join(output_labels_dir, filename)

            with open(input_filepath, 'r') as infile:
                lines = infile.readlines()

            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                current_class_id = int(parts[0])

                # Jika class ID saat ini adalah class_id untuk 'car'
                if current_class_id == original_car_id:
                    # Ganti class_id ke class_id baru (0)
                    parts[0] = str(new_car_id)
                    filtered_lines.append(' '.join(parts) + '\n')

            # Tulis ulang file label yang sudah difilter
            # Jika tidak ada anotasi 'car', file .txt baru akan kosong (ini valid untuk YOLO)
            with open(output_filepath, 'w') as outfile:
                outfile.writelines(filtered_lines)
            processed_files_count += 1
    print(f"Selesai memfilter {processed_files_count} file label untuk {split_name}. Label disimpan di: {output_labels_dir}")
    return True

# --- Main Program ---
if __name__ == '__main__':
    # Memproses setiap split data
    splits = ['train', 'valid', 'test']
    for split in splits:
        filter_yolo_labels_for_car(BASE_DATA_DIR, split, ORIGINAL_CAR_CLASS_ID, NEW_CAR_CLASS_ID)

    # --- Memperbarui data.yaml ---
    print("\n-----------------------------------------------------------")
    print(f"SELESAI MEMANIPULASI LABEL. Sekarang, Anda harus MEMPERBARUI {DATA_YAML_NAME} Anda.")
    print("Buka file data.yaml dan modifikasi bagian ini:")
    print("-----------------------------------------------------------")
    print(f"nc: {NEW_CAR_CLASS_ID + 1} # Ubah menjadi 1")
    print(f"names: ['car'] # Ubah menjadi hanya 'car'")
    print("\nDan ubah jalur label agar menunjuk ke folder yang baru:")
    print(f"train: {os.path.join(os.path.abspath(BASE_DATA_DIR), 'train', 'images')}")
    print(f"val: {os.path.join(os.path.abspath(BASE_DATA_DIR), 'valid', 'images')}")
    print(f"test: {os.path.join(os.path.abspath(BASE_DATA_DIR), 'test', 'images')} # Jika Anda menggunakan test set")
    print("\nPastikan Anda menunjuk ke direktori 'images', bukan 'labels', karena YOLO akan mencari 'labels' secara otomatis di folder yang sesuai.")
    print("Dan pastikan Anda memperbarui bagian 'labels' di setiap split di file yaml yang Anda gunakan untuk training (jika Anda menggunakan 'path/to/labels' bukannya otomatis dicari).")
    print("Untuk Ultralytics YOLOv8/v11+, Anda biasanya hanya perlu menunjuk ke direktori images di YAML, dan ia akan mencari direktori 'labels' yang sesuai.")
    print("Contoh: train: ../train/images")
    print("Namun, pastikan folder 'labels_car_only' yang baru ini berada di samping folder 'images' di setiap split (train, valid, test).")
    print("\n-----------------------------------------------------------")