# Indonesian Car Intelligence System

An AI-powered system for Indonesian car classification and detection using computer vision and deep learning techniques.

## 🚗 Project Overview

This project aims to create an intelligent system capable of identifying and classifying various types of Indonesian cars. It includes multiple approaches:

- **Object Detection**: Using YOLO11X for real-time car detection
- **Classification**: Using ResNet50, ConvNeXt, and EfficientNetV2 for car type classification

## 📁 Project Structure

```
Indonesian_car_Intellegence_system/
├── Object_detection_Yolo11X/     # YOLO11X object detection implementation
├── Resnet50/                     # ResNet50 classification model
├── ConvNeXt/                     # ConvNeXt classification model
├── EfficientNetV2/               # EfficientNetV2 classification model
├── data/                         # Dataset for training and validation
├── requirements_detection.txt    # Dependencies for detection tasks
├── requirements_classification.txt # Dependencies for classification tasks
└── README.md                     # Project documentation
```

## 🎯 Car Categories

The system can classify the following Indonesian car types:

- **City Car**: Compact urban vehicles
- **LCGC**: Low Cost Green Car (Mobil Murah Ramah Lingkungan)
- **Hatchback**: Rear door that opens upwards
- **Sedan**: Traditional 4-door passenger cars
- **SUV**: Sport Utility Vehicles
- **MPV**: Multi-Purpose Vehicles
- **Pickup Truck**: Light commercial vehicles
- **Commercial Van**: Commercial transport vehicles

## 🛠️ Installation

### For Object Detection
```bash
pip install -r requirements_detection.txt
```

### For Classification
```bash
pip install -r requirements_classification.txt
```

## 🚀 Usage

### Object Detection with YOLO11X
```python
cd Object_detection_Yolo11X
python detect.py --source your_image.jpg
```

### Car Classification
```python
cd Resnet50
python train_classifier.py
```

## 📊 Models

### Object Detection
- **YOLO11X**: State-of-the-art real-time object detection

### Classification Models
- **ResNet50**: Deep residual network for image classification
- **ConvNeXt**: Modern ConvNet architecture
- **EfficientNetV2**: Efficient and scalable image classification

## 📈 Performance

*Performance metrics will be updated as models are trained and evaluated*

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
- PyTorch and TensorFlow communities for the deep learning frameworks

