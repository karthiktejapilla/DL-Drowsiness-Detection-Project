"""
Configuration settings for the Drowsiness Detection project.
All hyperparameters and paths are centralized here.

Dataset: Drowsiness Prediction Dataset (DPD)
Task: Binary face-level classification — Active vs Fatigue
"""
import os

# ─── Project Root ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset Paths (DPD — Drowsiness Prediction Dataset) ───
DPD_DATASET_DIR = os.path.abspath(os.environ.get(
    "DPD_DATASET_DIR",
    os.path.join(PROJECT_ROOT, "DPD 9000 image dataset", "0 FaceImages"),
))
DPD_ACTIVE_DIR = os.path.join(DPD_DATASET_DIR, "Active Subjects")
DPD_FATIGUE_DIR = os.path.join(DPD_DATASET_DIR, "Fatigue Subjects")

# ─── Model Paths ───
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "drowsiness_model.keras")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FACE_CROP_CACHE_DIR = os.path.join(OUTPUTS_DIR, "dpd_face_crops")
FACE_CROP_CACHE_VERSION = "haar-640px-pad15-v2-loose"

# ─── Image Settings ───
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─── Training Hyperparameters ───
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
EPOCHS = 15
FINE_TUNE_EPOCHS = 10
DROPOUT_RATE = 0.3
RANDOM_SEED = 42

# ─── Label Mapping ───
# Active subjects = 0, Fatigue subjects = 1
CLASS_NAMES = ["Active", "Fatigue"]
NUM_CLASSES = 2

# ─── Data Split Ratios ───
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Drowsiness Engine Settings ───
# These thresholds apply to temporal tracking in video/webcam modes
FATIGUE_THRESHOLD_SEC = 2.0           # Seconds of continuous fatigue for warning
BLINK_MAX_DURATION_SEC = 0.4          # Maximum duration for a transient state change
PREDICTION_SMOOTHING_WINDOW = 5       # Number of frames for smoothing
CONFIDENCE_THRESHOLD = 0.475          # Validation-selected threshold for the fine-tuned checkpoint
MIN_INFERENCE_CONFIDENCE = 0.55       # Below this, live UI reports uncertain/no decision
WEBCAM_CAPTURE_WIDTH = 640
WEBCAM_CAPTURE_HEIGHT = 480
WARNING_COOLDOWN_SEC = 5.0            # Cooldown between audible warnings

# ─── Webcam Settings ───
CAMERA_INDEX = 0
WEBCAM_FPS_TARGET = 15

# ─── Video Processing ───
VIDEO_FRAME_SAMPLE_RATE = 3           # Process every Nth frame
MAX_VIDEO_FRAMES = 5000               # Maximum frames to process per video

# ─── Face Detection Settings (for webcam/uploaded images) ───
FACE_DETECTION_MIN_SIZE = (60, 60)
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 4
