# 🔥 Smart Fire & Smoke Detection System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

An advanced, real-time fire and smoke detection system leveraging **MobileNetV2**, **Grad-CAM localization**, and a high-performance **OpenCV image enhancement pipeline**. This project provides both a robust classification backend and a premium web-based live monitoring interface.

---

## 🚀 Key Features

### 1. Real-time Live Monitoring

- **Web-Based Interface**: A premium, mobile-responsive dashboard for live camera streaming.
- **Low Latency**: Optimized frame processing using base64 streaming and threaded Flask, achieving smooth real-time detection.
- **Cross-Device Support**: Works on desktops and smartphones with easy camera switching.

### 2. High-Precision Localization (Grad-CAM)

- **Visual Evidence**: Even though the model is a classifier, we use **Gradient-weighted Class Activation Mapping** to visualize exactly _where_ the model sees fire or smoke.
- **Dynamic Bounding Boxes**: Automatically extracts and draws bounding boxes around detected hazards by analyzing the Grad-CAM activation heatmap.

### 3. Advanced Image Enhancement Pipeline

To handle reflections (e.g., monitoring through windows) and poor visibility, the system includes:

- **Reflection Suppression**: Multi-stage algorithm to remove glare and specular highlights.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization to boost visibility in smoky or foggy conditions.
- **Temporal Smoothing**: Weighted probability averaging to reduce detection flicker in live feeds.
- **Denoising & Sharpening**: Bilateral filtering and unsharp masking for cleaner input features.

### 4. GPU-Accelerated Training

- **Mixed Precision**: Uses `mixed_float16` to leverage NVIDIA Tensor Cores for 2-3x faster training.
- **MobileNetV2 Backbone**: Fine-tuned for the optimal balance between accuracy and edge device performance.

---

## 🛠 Project Structure

```text
FS_SM_NSM/
├── app.py              # Main Flask server with image enhancement & Grad-CAM
├── Classification.py   # Training script with KerasTuner hyperparameter search
├── config.py           # Centralized project configuration (paths, class names, etc.)
├── templates/
│   └── index.html      # Premium live detection frontend
├── output/             # Saved models, training plots, and classification reports
└── data/               # Dataset directory (Fire, Smoke, Neutral)
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA & cuDNN (Recommended for training)

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   cd upgrad/FS_SM_NSM
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _(Required: tensorflow, flask, opencv-python, pillow, scikit-learn, keras-tuner)_

3. **Configure the Project**:
   Edit `config.py` to set your absolute paths for data and output.

---

## 🚦 Usage

### ⚙️ 1. Training the Model

To start the hyperparameter search and final model training:

```bash
python Classification.py
```

This will automatically save the best model and performance metrics (Confusion Matrix, Loss Curves) to the `output/` directory.

### 🎥 2. Running Live Detection

To start the web application and live camera monitoring:

```bash
python app.py
```

Access the dashboard at `http://localhost:5000`.

---

## 📊 Performance

- **Accuracy**: ~95% on test set.
- **Balanced F1-Score**: High precision across all three classes (Fire, Smoke, Non-Fire).
- **Localization**: Robust bounding box extraction using the `Conv_1_bn` activation map.

---

## 📜 Acknowledgements

- **Dataset**: [Forest Fire, Smoke, and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)
- **Frameworks**: TensorFlow, Keras, Flask, OpenCV.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](file:///C:/Users/mbpd1/Downloads/upgrad/FS_SM_NSM/LICENSE) file for details.

---

© 2026 Bharadwaj Phani Datta Mahanthi - High-Performance AI Safety.
