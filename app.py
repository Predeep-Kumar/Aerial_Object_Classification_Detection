"""
app.py
Streamlit app for:
 - Binary classification (Bird vs Drone)
 - Optional YOLOv8 object detection mode

Notes:
 - Place this file in the project root where MODELS_DIR, REPORTS_DIR, CONFIG_DIR exist.
 - Run locally: `streamlit run app.py`
"""

import os
# IMPORTANT: set this BEFORE importing TensorFlow/Keras if you need legacy .h5 loading
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path
import logging

import streamlit as st
import numpy as np
from PIL import Image

# ML libraries (import after env var)
import tensorflow as tf

# Optional YOLO
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Model-specific preprocessors
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# -------------------- USER-CONFIG (edit if needed) --------------------
BASE_DIR = Path(os.environ.get("PROJECT_BASE_DIR", Path(__file__).resolve().parent))
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "model_comparison"
CONFIG_DIR = BASE_DIR / "config"


# Default classification image size (must match your classifier input)
CLASS_IMG_SIZE = (224, 224)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# -------------------- HELPERS --------------------
@st.cache_resource
def load_selected_model_info():
    """Load selected_model.json saved from model selection."""
    sel_path = Path(REPORTS_DIR) / "selected_model.json"
    if not sel_path.exists():
        return None
    try:
        with open(sel_path, "r") as f:
            return json.load(f)
    except Exception:
        logger.exception("Error loading selected_model.json")
        return None

@st.cache_resource
def load_classifier(model_path):
    """
    Load a Keras/TensorFlow model. Returns model or None if path missing.
    Handles legacy .h5 files robustly (tries tf.keras, then keras package if available).
    """
    if model_path is None:
        return None

    model_path = str(model_path)
    if not os.path.exists(model_path):
        logger.warning("Requested classifier path does not exist: %s", model_path)
        return None

    # Try tf.keras first (should work for SavedModel, .keras, and .h5 with legacy flag)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Loaded classifier via tf.keras from %s", model_path)
        return model
    except Exception as e_tf:
        logger.warning("tf.keras failed to load model (%s). Trying fallback keras package if available. Error: %s", model_path, e_tf)
        try:
            import keras
            model = keras.models.load_model(model_path, compile=False)
            logger.info("Loaded classifier via keras package fallback from %s", model_path)
            return model
        except Exception as e_keras:
            logger.exception("Failed to load model with both tf.keras and keras: %s", e_keras)
            return None

def pick_preprocess_fn_for_model_name(name_or_path: str):
    """Return the appropriate preprocessing function for a model name/path."""
    if not name_or_path:
        return None
    n = name_or_path.lower()
    if "efficient" in n:
        return eff_preprocess
    if "mobilenet" in n:
        return mobilenet_preprocess
    if "resnet" in n:
        return resnet_preprocess
    return None

def preprocess_for_classifier(pil_img, target_size=CLASS_IMG_SIZE, preprocess_fn=None):
    """
    Resize + apply model-specific preprocessing.
    - If preprocess_fn is None, scales to [0,1].
    - If preprocess_fn provided, pass raw uint8 array (0-255) into it (Keras apps expect that).
    """
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    if preprocess_fn is None:
        arr = arr / 255.0
    else:
        # many keras preprocess_input functions expect array in 0-255 range
        arr = preprocess_fn(arr)
    return np.expand_dims(arr, axis=0)

def get_prediction_text(prob):
    """Return class label and formatted confidence text given probability of 'drone' class."""
    prob = float(prob)
    pred_label = "drone" if prob >= 0.5 else "bird"
    conf = prob if prob >= 0.5 else 1.0 - prob
    return pred_label, conf

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="🦅 Aerial Objects Classification & Detection", layout="centered")
st.title("🦅 Aerial Objects Classification & Detection")
st.markdown(
    "This system performs both **classification** and **detection** on aerial images to reliably distinguish between **Birds** and **Drones**."
)

st.markdown("Upload an image to run classification or detection.")

# Sidebar: settings
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["Classification", "Detection (YOLO)"], index=0)

# Load selected model info (if exists)
selected_info = load_selected_model_info()
classifier = None
preprocess_fn = None
if selected_info:
    chosen_model = selected_info.get("chosen_model", "<unknown>")
    chosen_path = selected_info.get("chosen_model_path") or selected_info.get("model_path") or selected_info.get("model_file") or selected_info.get("model")
    if chosen_path:
        # resolve relative paths relative to project root
        chosen_path = str((BASE_DIR / chosen_path) if not Path(chosen_path).is_absolute() else Path(chosen_path))
    st.sidebar.markdown(f"**Selected model:** {chosen_model}")
    st.sidebar.markdown(f"**Model path:** `{chosen_path}`")
    classifier = load_classifier(chosen_path)
    preprocess_fn = pick_preprocess_fn_for_model_name(chosen_model) or pick_preprocess_fn_for_model_name(chosen_path)
else:
    st.sidebar.warning("No selected model info found at reports/model_comparison/selected_model.json")

# If detection mode chosen, try to load YOLO weights
yolo_model = None
if mode == "Detection (YOLO)":
    if not YOLO_AVAILABLE:
        st.sidebar.error("ultralytics (YOLO) not installed in this environment.")
    else:
        try:
            # attempt to find a candidate YOLO weights file under MODELS_DIR
            yolo_candidates = list(Path(MODELS_DIR).glob("yolov8*best.pt"))
            if yolo_candidates:
                yolo_path = str(yolo_candidates[0])
                st.sidebar.markdown(f"YOLO weights found: `{yolo_path}`")
                try:
                    yolo_model = YOLO(yolo_path)
                except Exception as e:
                    st.sidebar.error(f"Failed to load YOLO model: {e}")
            else:
                st.sidebar.warning("No YOLO weights found in models/. Place yolov8_*_best.pt there.")
        except Exception as e:
            st.sidebar.error(f"Error searching for YOLO weights: {e}")

# Image uploader + example button
st.sidebar.markdown("---")
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

img_src = None
if uploaded is not None:
    try:
        img_src = Image.open(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not open uploaded image: {e}")

if img_src is None:
    st.info("Please upload an image to continue.")
    st.stop()

# Show input image
st.image(img_src, caption="Input image", use_column_width=True)

# -------------------- Classification --------------------
if mode == "Classification":
    if classifier is None:
        st.error("Classifier model not found. Ensure selected_model.json points to a valid Keras model (SavedModel / .keras / .h5).")
    else:
        try:
            # Use model-specific preprocess function if determined above
            x = preprocess_for_classifier(img_src, CLASS_IMG_SIZE, preprocess_fn=preprocess_fn)

            # Predict
            preds = classifier.predict(x, verbose=0)
            preds = np.array(preds).ravel()

            # Handle different output shapes (sigmoid vs softmax)
            if preds.size == 1:
                # single sigmoid output -> probability of positive class (assumed 'drone')
                prob_drone = float(preds[0])
            elif preds.size == 2:
                # two outputs (softmax) -> we assume ordering [bird, drone]
                prob_drone = float(preds[1])
            else:
                # fallback: use first value
                prob_drone = float(preds[0])

            pred_label, conf = get_prediction_text(prob_drone)
            st.metric("Prediction", f"{pred_label} ({conf*100:.2f}%)")
        except Exception as e:
            st.error(f"Classification error: {e}")
            st.stop()

# -------------------- Detection (YOLO) --------------------
elif mode == "Detection (YOLO)":
    if yolo_model is None:
        st.error("YOLO model not loaded. Place weights in models/ or install ultralytics.")
    else:
        try:
            # ultralytics accepts numpy arrays as source
            results = yolo_model.predict(source=np.array(img_src), imgsz=640, conf=0.25, save=False)
            annotated = results[0].plot()  # numpy array
            st.image(annotated, caption="YOLOv8 detections", use_column_width=True)

            # show per-detection details
            dets = None
            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                try:
                    dets = results[0].boxes.data.cpu().numpy()
                except Exception:
                    try:
                        dets = np.array(results[0].boxes.data)
                    except Exception:
                        dets = None

            if dets is not None and dets.size > 0:
                st.write("Detections (x1,y1,x2,y2,score,class_id):")
                for row in dets:
                    x1, y1, x2, y2, score, class_id = row[:6]
                    st.write(f"Class {int(class_id)} score {float(score):.2f} bbox [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            else:
                st.write("No detections above confidence threshold.")
        except Exception as e:
            st.error(f"YOLO inference error: {e}")

st.markdown("---")
st.caption("Streamlit interface for aerial object classification and detection (Bird vs Drone).")

