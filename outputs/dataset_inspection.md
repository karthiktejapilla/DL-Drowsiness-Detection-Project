# Dataset Inspection Report

## Continuation Verification Addendum (2026-10-16)

Image files were counted directly from the local folders: MRL totals 84,898 (train 50,937; validation 16,980; test 16,981), and UTA-RLDD totals 11,787 (train 9,054; validation 1,824; test 909). MRL filenames identify the same 37 subjects in each split, with all 37 subjects overlapping between every pair of splits; this is not subject-independent evaluation. File counts do not establish image decodability.

UTA-RLDD annotation files contain 6,384 train lines, 1,715 validation lines, and 909 test lines, all referencing existing files. The train/validation lists do not cover every image present in their corresponding folders, and `src/video_dataset.py` prioritizes these annotation lists. Counts below describing UTA folders should not be mistaken for annotated loader counts.

The local report paths below are historical notes. Use relative repository paths; no private absolute path is needed to access the data.

**Date:** 2026-10-15
**Project:** Human Drowsiness Detection Using Deep Learning

---

## 1. Project Root

`D:\D\projects\DL-Project Drowsiness Detection`

---

## 2. MRL Eye Dataset (Forked Version)

### Location
`D:\D\projects\DL-Project Drowsiness Detection\archive\data\`

### Structure
```
archive/
├── archive.zip (344 MB)
└── data/
    ├── train/
    │   ├── awake/   (25,770 images)
    │   └── sleepy/  (25,167 images)
    ├── val/
    │   ├── awake/   (8,591 images)
    │   └── sleepy/  (8,389 images)
    ├── test/
    │   ├── awake/   (8,591 images)
    │   └── sleepy/  (8,390 images)
    ├── get_info.py
    ├── split_data.py
    ├── labels.txt
    └── readme.md
```

### Image Format
- **Format:** PNG (grayscale infrared eye images)
- **Typical file size:** 2–4 KB per image
- **Total images:** 84,898

### Class Distribution

| Split    | Awake   | Sleepy  | Total   |
|----------|---------|---------|---------|
| Train    | 25,770  | 25,167  | 50,937  |
| Val      | 8,591   | 8,389   | 16,980  |
| Test     | 8,591   | 8,390   | 16,981  |
| **Total**| **42,952** | **41,946** | **84,898** |

**Class balance:** Approximately 50.6% awake / 49.4% sleepy — well balanced.

### Filename Convention
```
s{subject}_{imageID}_{eyeState}_{gender}_{glasses}_{reflections}_{lighting}_{sensorID}.png
```

Example: `s0001_01842_0_0_1_0_0_01.png`

### Label Mapping

> **IMPORTANT:** This is a **forked version** of the MRL dataset. The **folder name** (awake/sleepy) is the ground truth label, NOT the eye_state field in the filename.

The "awake" folder contains a mix of eye_state=0 (19,629) and eye_state=1 (6,141) in filenames, confirming that the folder name is the canonical label and the filename encoding is from the original MRL dataset which used a different labeling scheme.

**Project label convention:**
- `0 = OPEN (awake)` → Eyes are open, person alert
- `1 = CLOSED (sleepy)` → Eyes are closed, person drowsy

### Subjects
- **37 subjects** (s0001 through s0037)
- Subject ID is embedded in filenames

### Metadata Available
- Subject ID, Image ID, Eye state (original), Gender, Glasses, Reflections, Lighting, Sensor ID

### Sensors
- `01`: RealSense (640×480)
- `02`: IDS (1280×1024)
- `03`: Aptina (752×480)

### Existing Split
The dataset comes **pre-split** into train/val/test (60/20/20).

> **Note:** The existing split was done randomly by images (via `split_data.py`), NOT by subject. For the project, we will use the existing split since it's already organized, but document this limitation. A subject-wise re-split would be ideal but risks disrupting the folder structure.

---

## 3. UTA-RLDD Dataset

### Location
`D:\D\projects\DL-Project Drowsiness Detection\UTA-RLDD Dataset\`

### Structure
```
UTA-RLDD Dataset/
├── UTA-RLDD Dataset.zip (2.77 GB)
├── train/
│   ├── active/   (5,862 images)
│   └── fatigue/  (3,192 images)
├── val/
│   ├── active/   (912 images)
│   └── fatigue/  (912 images)
├── test/
│   ├── active/   (456 images)
│   └── fatigue/  (453 images)
├── train.txt     (6,385 lines)
├── val.txt       (1,716 lines)
└── test.txt      (910 lines)
```

### Image Format
- **Format:** JPG (face images extracted from videos)
- **Typical file size:** 79–255 KB per image (much larger than MRL)
- **Total images:** 11,787

### Class Distribution

| Split    | Active  | Fatigue | Total   |
|----------|---------|---------|---------|
| Train    | 5,862   | 3,192   | 9,054   |
| Val      | 912     | 912     | 1,824   |
| Test     | 456     | 453     | 909     |
| **Total**| **7,230** | **4,557** | **11,787** |

**Class balance:** 61.3% active / 38.7% fatigue — moderately imbalanced.

### Annotation Format
Text files with format: `relative_path label`
- Label `0` = active
- Label `1` = fatigue

Example: `train/active/img_f_3422.jpg 0`

### Key Observations
1. This is a **frame-extracted** version of UTA-RLDD — the original dataset contains videos, but this version has pre-extracted frames as JPG images.
2. Filenames suggest subject identifiers via letter prefixes (img_a, img_b, ..., img_k, image_).
3. These are **face images** (not eye crops), so they represent a different task from MRL.
4. The labels are "active" vs "fatigue" (video-level drowsiness state), NOT eye-open vs eye-closed.

### Feasibility for Video-Level Model
Since UTA-RLDD is already frame-extracted with face-level images, it CAN be used to train a separate face-level fatigue classifier. However:
- It should NOT be merged with MRL's eye-state task
- It can supplement the drowsiness detection by providing a separate face-level fatigue signal

---

## 4. Existing Code Files

| File | Purpose |
|------|---------|
| `archive/data/get_info.py` | Dataset statistics and schema |
| `archive/data/split_data.py` | Train/val/test splitting script |

**No existing:**
- Model files or checkpoints
- Training scripts
- Notebooks
- Pretrained weights

---

## 5. System Environment

| Property | Value |
|----------|-------|
| OS | Windows |
| RAM | ~8 GB |
| GPU | None (CPU-only) |
| Free Disk | ~98 GB |
| Python | Not installed (installing 3.11) |

---

## 6. Recommended Strategy

### Model A: Eye-State Classification (MRL)
- **Architecture:** MobileNetV3Small with ImageNet weights
- **Input:** Grayscale images converted to RGB, resized to 128×128×3
- **Task:** Binary classification (awake=0, sleepy=1)
- **Split:** Use existing train/val/test split
- **Training:** Transfer learning with frozen backbone → optional fine-tuning
- **Batch size:** 16 (CPU-friendly)
- **Expected training time:** ~30-60 min for 10 epochs on CPU

### Model B: Temporal Drowsiness Engine
- **Approach:** Use Model A predictions over time with a rule-based temporal engine
- **No separate neural network needed** — Model A + temporal logic is sufficient
- **UTA-RLDD:** Use for face-level fatigue validation if time permits

### UTA-RLDD Usage
- Can be used to validate the face-level drowsiness detection pipeline
- Can train an optional face-level fatigue classifier as a bonus model
- Should NOT be merged with MRL eye-state labels

### Corrupted/Unreadable Files
- Not yet verified programmatically (requires Python/PIL)
- Will be checked during data loading with error handling
