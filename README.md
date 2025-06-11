# Indonesian Car Intelligence System

An AI-powered system designed for Indonesian car classification and detection, leveraging computer vision and deep learning techniques. This project aims to analyze the composition of car types on Indonesian roads.

## 🚗 Project Overview

This project focuses on creating an intelligent system capable of identifying and classifying various types of Indonesian cars from video streams. It integrates two main components:

- **Object Detection**: Utilizing YOLO11X for efficient, real-time car detection.
- **Classification**: Employing advanced CNN architectures (ResNet50, EfficientNetV2, and ConvNeXt) for detailed car type classification.

## 📁 Project Structure

```
Indonesian_car_Intelligence_system/
├── Object_detection_Yolo11X/     # YOLO11X object detection implementation (e.g., best.pt)
├── Resnet50/                     # ResNet50 classification model
├── ConvNeXt/                     # ConvNeXt classification model  
├── EfficientNetV2/               # EfficientNetV2 classification model
├── data/                         # Dataset for training and validation of classification models
├── traffic_test.mp4              # Sample video for demonstration and inference
├── Yolov11x_ConvNeXt.mp4         # Demo video showcasing YOLOv11X + ConvNeXt performance
├── traffic_test.py               # Main script for integrated detection and classification demo
├── best.pt                       # Pre-trained YOLO11X model weights
├── requirements_detection.txt    # Dependencies for detection tasks
├── requirements_classification.txt # Dependencies for classification tasks
└── README.md                     # Project documentation
```

## 🎯 Car Categories

The system is trained to classify the following 8 Indonesian car types, commonly found on Indonesian roads:

- **City Car**: Compact urban vehicles (e.g., Honda Brio, Suzuki Ignis)
- **LCGC**: Low Cost Green Car (e.g., Toyota Calya, Daihatsu Sigra)
- **Hatchback**: Rear door that opens upwards (e.g., Honda Jazz, Toyota Yaris)
- **Sedan**: Traditional 4-door passenger cars (e.g., Honda Civic, Toyota Camry)
- **SUV**: Sport Utility Vehicles (e.g., Honda CR-V, Toyota Fortuner)
- **MPV**: Multi-Purpose Vehicles (e.g., Toyota Avanza, Mitsubishi Xpander)
- **Pickup Truck**: Light commercial vehicles (e.g., Toyota Hilux, Mitsubishi Triton)
- **Commercial Van**: Commercial transport vehicles (e.g., Toyota HiAce, Daihatsu Gran Max Blind Van)

## 🛠️ Installation

**Highly recommended to use CUDA-enabled GPU for optimal performance.**

### 1. Check CUDA Installation

Before proceeding, ensure CUDA is correctly installed and recognized by PyTorch.

```python
import torch

def check_cuda_installation():
    if torch.cuda.is_available():
        print("CUDA is correctly installed and recognized by PyTorch!")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT installed or not recognized by PyTorch.")
        print("Models will run on CPU, which will be significantly slower.")

if __name__ == '__main__':
    check_cuda_installation()
```

### 2. Global CUDA and cuDNN Installation (Highly Recommended)

For optimal performance with GPU acceleration, ensure CUDA Toolkit and cuDNN are installed globally on your system. For this project, CUDA 12.6 and cuDNN 8.9.7.29 were used.

**Install CUDA Toolkit:**
- Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Download and install CUDA 12.6 corresponding to your operating system
- Follow NVIDIA's installation instructions
- Verify by running `nvcc --version` in your terminal

**Install cuDNN:**
- Visit [NVIDIA cuDNN Download](https://developer.nvidia.com/cudnn-download) (requires NVIDIA Developer Program registration)
- Download cuDNN version 8.9.7.29 for CUDA 12.x
- Extract the ZIP file
- Copy the contents (bin, include, lib folders) to your CUDA installation directory

### 3. Setup Virtual Environment and Install Dependencies

It is crucial to use a virtual environment to isolate project dependencies.

**Navigate to Project Directory:**
```bash
cd /path/to/your/Indonesian_car_Intelligence_system
```

**Create Virtual Environment:**
```bash
python -m venv venv
```

**Activate Virtual Environment:**
- Windows: `.\venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

**Install PyTorch with CUDA support (Crucial for GPU):**

After global CUDA/cuDNN installation, install PyTorch within your active virtual environment. For CUDA 12.6, use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(Note: Always verify the exact command on the PyTorch website for the most up-to-date compatibility.)*

**Install Remaining Dependencies:**
```bash
pip install -r requirements_detection.txt
pip install -r requirements_classification.txt
```

## 🚀 Usage

### Integrated Car Detection and Classification Demo (traffic_test.py)

The main script `traffic_test.py` provides flexible usage options:

```bash
python traffic_test.py
```

Upon running, you will be prompted to choose a mode:

1. **Real-time Display**: The application will open a video window displaying the YOLO detections and car type classifications live as the video is processed. Ideal for interactive demonstration.

2. **Render to Directory**: The application will process the entire video, performing YOLO detection and classification, and then save the annotated output video to a specified file. A progress bar and estimated time remaining will be shown in the terminal. Suitable for batch processing or generating final output videos.

### Object Detection with YOLO11X (Standalone)

To run YOLO11X detection independently (without classification), you can refer to the `Object_detection_Yolo11X` directory for specific scripts or simply modify `traffic_test.py` to disable the classification part.

### Car Classification (Training)

To train the classification models (ResNet50, EfficientNetV2, ConvNeXt), navigate to their respective directories and run the training scripts:

```bash
cd training
python train_classifier_augmented_v2.py
```

## 📊 Models

### Object Detection
- **YOLO11X**: State-of-the-art real-time object detection framework. Utilized for robust and efficient car detection.

### Classification Models
- **ResNet50**: A deep residual network, serving as a strong baseline for image classification due to its proven effectiveness and transfer learning capabilities.
- **EfficientNetV2**: A family of efficient and scalable models that balance accuracy and computational efficiency. Used to explore modern CNN performance.
- **ConvNeXt**: A modern ConvNet architecture inspired by Vision Transformers, offering state-of-the-art performance while maintaining CNN efficiency. Used to push classification accuracy further.

## 📈 Performance

Detailed performance metrics, including training loss/accuracy graphs, overfitting analysis, confusion matrices, and precision-recall curves for both object detection (YOLO11X) and classification models (ResNet50, EfficientNetV2, ConvNeXt), are documented in the comprehensive Project Report.

A summary of key performance indicators and representative figures may also be found on the project's GitHub Pages (if configured).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **anggr** - Initial work

## 🙏 Acknowledgments

- Thanks to the Indonesian automotive community for inspiration
- YOLO team for the object detection framework
- PyTorch community for the deep learning framework
- Roboflow for dataset management support
- [Add any other acknowledgments here]
