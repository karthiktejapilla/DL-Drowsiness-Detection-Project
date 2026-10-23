# Human Drowsiness Detection

A local Streamlit application for face-level Active/Fatigue classification using the Kaggle Drowsiness Prediction Dataset (DPD), TensorFlow, and MobileNetV3Small. This is a research/demo prototype, not a certified driver safety or medical system.

## Dataset and labels

The active training and evaluation pipeline uses only [Kaggle DPD](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset). The verified local folder layout is:

```text
DPD 9000 image dataset/
└── 0 FaceImages/
    ├── Active Subjects/   # label index 0: Active
    └── Fatigue Subjects/  # label index 1: Fatigue
```

The current copy contains 9,120 JPEG images: 4,560 in each directory. All decoded successfully; no exact file duplicates were found. The source files are full-size portraits at varied resolutions. Training and full-frame inference use the same Haar face detector and 15% padded face crop before resize. The detector missed 63 images; they are explicitly excluded and listed in `outputs/face_crop_exclusions.csv`. The remaining 9,057 images use a seeded, stratified image-level 70/15/15 split recorded in `outputs/split_manifest.csv`.

These are face-level Active/Fatigue labels. They are not Open/Closed eye labels. MRL Eye and UTA-RLDD folders may remain on disk, but the configured training, evaluation, and inference checkpoint use DPD only. The old eye-state checkpoint is not used as a fallback.

## Setup (Windows PowerShell)

From the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default dataset path is `DPD 9000 image dataset\0 FaceImages` inside the project. To use a different copy, set the environment variable before training or evaluation:

```powershell
$env:DPD_DATASET_DIR = "E:\datasets\0 FaceImages"
```

The folder must contain both `Active Subjects` and `Fatigue Subjects` directories. If that environment variable is set only in one PowerShell window, it applies only to commands run from that window.

## Train and evaluate

The current workspace includes the face-crop checkpoint and its 12-epoch history. To train a new checkpoint:

```powershell
python train.py --epochs 8 --fine-tune-epochs 4 --batch-size 32
```

This reuses cached crops in `outputs/dpd_face_crops/`, writes the best checkpoint and metadata under `models/`, the training history and split manifest under `models/` and `outputs/`, and training curves to `outputs/training_curves.png`. If source images change or the crop detector version changes, crops are prepared again.

Evaluate the checkpoint on the held-out DPD test split:

```powershell
python evaluate.py
```

Evaluation writes `outputs/evaluation.json`, `outputs/evaluation_report.md`, `outputs/confusion_matrix.png`, `outputs/roc_curve.png`, and `outputs/misclassified_examples.csv`. Binary precision, recall, and F1 treat Fatigue as the positive class. Test data must not be used to tune training choices.

## Run the application

```powershell
python -m streamlit run app.py
```

The dashboard provides image analysis, sampled video analysis, browser camera still capture, dataset exploration, training/evaluation plots, project overview, and architecture information. Continuous webcam monitoring is provided by the separate local OpenCV program (the camera attached to the computer running it):

```powershell
python realtime_webcam.py --threshold 2.0
```

The webcam monitor requests a practical 640×480 capture size, reports measured display FPS and per-frame inference latency, resets its temporal state when the stream ends, and shows no-face/low-confidence frames as `NO FACE DETECTED` instead of forcing a class. Press `q` in the OpenCV window to stop it. Browser camera capture in Streamlit is a still-image capture, not a continuous browser video stream. Physical webcam accuracy was not measured in this environment because no representative ground-truth webcam recording was available.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Artifacts and structure

- `config.py`: dataset path, class mapping, model dimensions, and training settings.
- `src/data_loader.py`: DPD discovery, splitting, preprocessing input pipeline, and class distribution.
- `src/model.py`: classifier definition and active checkpoint compatibility checks.
- `src/inference.py`: shared face-level model inference for application input modes.
- `train.py` / `evaluate.py`: training and held-out test evaluation.
- `models/drowsiness_model.keras`: active model checkpoint (excluded from Git).
- `models/model_metadata.json`: architecture, data, preprocessing, split and evaluation metadata.
- `models/training_history.json`: epoch-by-epoch training and validation values.
- `outputs/`: split manifests, plots, evaluation reports, and misclassified file paths.

The application checks that the active checkpoint metadata, class mapping, input dimensions, and binary output shape agree. If the new checkpoint or its metadata is missing or incompatible, it reports an error instead of loading the old eye-state model.

## Limitations

- Evaluation is image-level and may include images from people represented in training because no subject IDs are available.
- Face detection/localization, image-classifier accuracy, and end-to-end webcam/video performance are different measurements. The classifier’s test metrics do not establish safe real-world driving performance.
- Lighting, pose, eyewear, camera quality, and domain changes can affect face detection and predictions.
- The temporal alert logic is a heuristic over frame classifications; it does not measure eye-closure duration or diagnose fatigue.
- Webcam and codec behavior depend on the local device and were not verified by dataset/model unit tests.
