"""
Configuration settings for the Drowsiness Detection project.
All hyperparameters and paths are centralized here.
"""
import os

# ─── Project Root ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset Paths ───
MRL_DATA_DIR = os.path.join(PROJECT_ROOT, "archive", "data")
MRL_TRAIN_DIR = os.path.join(MRL_DATA_DIR, "train")
MRL_VAL_DIR = os.path.join(MRL_DATA_DIR, "val")
MRL_TEST_DIR = os.path.join(MRL_DATA_DIR, "test")

UTA_RLDD_DIR = os.path.join(PROJECT_ROOT, "UTA-RLDD Dataset")
UTA_TRAIN_DIR = os.path.join(UTA_RLDD_DIR, "train")
UTA_VAL_DIR = os.path.join(UTA_RLDD_DIR, "val")
UTA_TEST_DIR = os.path.join(UTA_RLDD_DIR, "test")

# ─── Model Paths ───
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EYE_STATE_MODEL_PATH = os.path.join(MODELS_DIR, "eye_state_model.keras")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

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
# Folder name -> numeric label
# awake (eyes open) = 0, sleepy (eyes closed) = 1
CLASS_NAMES = ["awake", "sleepy"]
NUM_CLASSES = 2

# ─── Drowsiness Engine Settings ───
EYE_CLOSURE_THRESHOLD_SEC = 2.0     # Seconds of continuous closure for warning
BLINK_MAX_DURATION_SEC = 0.4        # Maximum duration for a blink (not drowsy)
PREDICTION_SMOOTHING_WINDOW = 5     # Number of frames for smoothing
CONFIDENCE_THRESHOLD = 0.507        # Validation-set F1 threshold; see audit for scope
WARNING_COOLDOWN_SEC = 5.0          # Cooldown between audible warnings

# ─── Webcam Settings ───
CAMERA_INDEX = 0
WEBCAM_FPS_TARGET = 15

# ─── Video Processing ───
VIDEO_FRAME_SAMPLE_RATE = 3         # Process every Nth frame
MAX_VIDEO_FRAMES = 5000             # Maximum frames to process per video

# ─── MediaPipe Face Mesh Settings ───
FACE_MESH_MAX_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# Eye landmark indices for MediaPipe Face Mesh
# Left eye
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
