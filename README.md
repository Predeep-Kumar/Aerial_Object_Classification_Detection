🌤️ Aerial Object Classification & Detection

Complete Deep Learning Project for classifying Bird vs Drone and detecting aerial objects using:

🧠 Custom CNN + Transfer Learning Models (ResNet50, MobileNetV2, EfficientNetB0)
🎯 YOLOv8 Object Detection
🌐 Streamlit Web Application

This project contains a full end-to-end pipeline from EDA → Preprocessing → Training → Model Comparison → YOLO Training → Deployment.

📁 Project Structure

Your final directory structure ( EXACTLY as you provided ):

Project-2/
│
└── AERIAL_OBJECT_CLASSIFICATION_&_DETECTION/
    │── app.py
    │── requirements.txt
    │── .gitignore
    │
    ├── config/
    │     ├── class_weights.json
    │     ├── data.yaml
    │     └── preprocessing_config.txt
    │
    ├── data/
    │     ├── classification_dataset/
    │     │     ├── train/
    │     │     ├── valid/
    │     │     └── test/
    │     │
    │     └── object_detection_Dataset/
    │           ├── train/
    │           ├── valid/
    │           ├── test/
    │           ├── labels/
    │           ├── data.yaml
    │           ├── README.dataset.txt
    │           └── README.roboflow.txt
    │
    ├── models/
    │     ├── custom_cnn_best.h5
    │     ├── custom_cnn_best.keras
    │     ├── mobilenetv2_best.h5
    │     ├── mobilenetv2_best.keras
    │     ├── efficientnetb0_best.keras
    │     ├── resnet50_best.keras
    │     ├── yolov8_yolov8n_bird_drone_best.pt
    │     └── yolo_runs/
    │
    ├── notebooks/
    │     ├── 01_EDA_Preprocessing.ipynb
    │     ├── 02_Custom_CNN_Classification.ipynb
    │     ├── 03_Transfer_Learning.ipynb
    │     ├── 04_Model_Comparison.ipynb
    │     └── 05_yolov8_object_detection.ipynb
    │
    └── reports/
          └── model_comparison/
                ├── transfer_learning_metrics.csv
                ├── selected_model.json
                └── misclassified_examples.txt

⚠️ Important Note
Model files are NOT included in this GitHub repository

Because GitHub blocks uploads larger than 100 MB.

Missing files (generated via notebooks):

custom_cnn_best.keras
resnet50_best.keras
mobilenetv2_best.keras
efficientnetb0_best.keras
yolov8_yolov8n_bird_drone_best.pt

👉 They will be created automatically when running notebooks in Google Colab.

OR you can download the complete project including models:

🔗 [ADD YOUR GOOGLE DRIVE LINK HERE]

🚀 Features
✔️ Bird vs Drone Classification

Implemented using 4 approaches:

Custom CNN

ResNet50

MobileNetV2

EfficientNetB0

Includes:

Class imbalance handling

Data augmentation

Confusion matrix

Classification reports

Weighted F1 scoring

Automatic model selection

✔️ YOLOv8 Object Detection

Detects Birds and Drones

Draws bounding boxes

Outputs class & confidence

✔️ Streamlit App

Upload image

Choose classification or detection

Live model outputs

Auto-loads best model via selected_model.json

Simple and clean UI

🧠 Training Pipeline (Google Colab)
▶️ 1. Run all notebooks in /notebooks

In this order:

01_EDA_Preprocessing.ipynb

02_Custom_CNN_Classification.ipynb

03_Transfer_Learning.ipynb

04_Model_Comparison.ipynb

05_yolov8_object_detection.ipynb

▶️ 2. Fix directory paths if required

Before running:

Check dataset paths

Check Drive mount paths

Check BASE_DIR, DATA_DIR, MODELS_DIR

▶️ 3. After run

All trained models appear in:

models/

📥 Download Full Project WITH Models

If you prefer the fully trained version:

👉 [Add your Google Drive link here]

💻 Running the Streamlit App

Inside:

Project-2/AERIAL_OBJECT_CLASSIFICATION_&_DETECTION/

1️⃣ Create virtual environment

Windows

python -m venv venv
venv\Scripts\activate


Mac/Linux

python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


The UI opens in your browser.

🔍 How It Works
1️⃣ Classification Mode

The selected model:

Loads automatically

Outputs Bird / Drone

Prints confidence score

2️⃣ Detection Mode

YOLOv8:

Detects objects

Draws bounding boxes

Shows confidence

📊 Included Analytics
✔️ Model performance comparison
✔️ Correct vs misclassified samples
✔️ Auto-select best model
✔️ YOLO performance
✔️ All documented inside notebooks
⭐ Future Enhancements

Multi-class (Bird / Drone / Plane / Helicopter)

TensorFlow Lite mobile deployment

Live webcam inference

Docker deployment

Real-time drone feed analysis

🤝 Author

Predeep Kumar
Aerial Object Classification & Detection — Complete DL + CV Project