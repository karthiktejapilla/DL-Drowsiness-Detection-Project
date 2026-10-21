# Human Drowsiness Detection — Project Audit & Upgrade Plan

**Audit Date:** October 2026  
**Auditor:** Senior Deep Learning & Computer Vision Engineer  
**Workspace:** `D:\D\projects\DL-Project Drowsiness Detection`  
**Target Environment:** Windows 10/11 64-bit, Python 3.11, CPU-Only Execution  

---

## 1. Executive Summary

This audit evaluates the codebase, neural network architecture, data pipelines, and user interfaces of the **Human Drowsiness Detection** project. The project aims to identify driver drowsiness using deep learning image classification coupled with temporal sequence tracking.

The project currently contains:
1. A trained Keras model (`models/eye_state_model.keras`) based on **MobileNetV3Small** for binary eye-state classification (Awake vs Sleepy).
2. Two local datasets: **MRL Eye Dataset** (forked, 84,898 eye crops) and **UTA-RLDD Dataset** (11,787 face video frames).
3. A basic Streamlit video processor (`app.py`) and a standalone OpenCV webcam runner (`realtime_webcam.py`).
4. An automated test suite (`tests/test_drowsiness.py`) passing 12 initial unit tests.

While the core deep learning model exhibits strong accuracy (96.30% on MRL test split), several major architectural, functional, and UI/UX gaps prevent it from being a cohesive, production-grade application.

---

## 2. Directory Structure & Asset Inventory

```
D:\D\projects\DL-Project Drowsiness Detection\
├── config.py                     # Central configuration & hyperparameters
├── app.py                        # Current Streamlit app (video upload only)
├── realtime_webcam.py            # Standalone OpenCV webcam script
├── train.py                      # Training script for eye state classifier
├── evaluate.py                   # Evaluation script on held-out test sets
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
│
├── archive/                      # MRL Eye Dataset (forked)
│   ├── archive.zip               # Compressed original archive (344 MB)
│   └── data/
│       ├── train/                # 50,937 images (25,770 awake, 25,167 sleepy)
│       ├── val/                  # 16,980 images (8,591 awake, 8,389 sleepy)
│       └── test/                 # 16,981 images (8,591 awake, 8,390 sleepy)
│
├── UTA-RLDD Dataset/             # Video-derived face dataset
│   ├── UTA-RLDD Dataset.zip      # Compressed original archive (2.77 GB)
│   ├── train/                    # 9,054 images (5,862 active, 3,192 fatigue)
│   ├── val/                      # 1,824 images (912 active, 912 fatigue)
│   ├── test/                     # 909 images (456 active, 453 fatigue)
│   ├── train.txt                 # Annotations
│   ├── val.txt
│   └── test.txt
│
├── models/
│   └── eye_state_model.keras     # Trained MobileNetV3Small model (9.5 MB)
│
├── outputs/
│   ├── confusion_matrix.png      # MRL test split confusion matrix
│   ├── training_curves.png       # Loss & accuracy curves
│   ├── training_history.csv      # Epoch-by-epoch loss & accuracy log
│   ├── evaluation.json           # Evaluation metrics on MRL & UTA-RLDD
│   ├── splits.csv                # Dataset split counts
│   └── dataset_inspection.md     # Initial dataset overview
│
├── src/
│   ├── __init__.py
│   ├── config.py (symlink/ref)
│   ├── data_loader.py            # MRL dataset discovery & tf.data generation
│   ├── video_dataset.py          # UTA-RLDD annotation parser
│   ├── preprocessing.py          # Scaling, augmentation, landmark crop
│   ├── model.py                  # MobileNetV3Small builder & transfer learning
│   └── drowsiness_engine.py      # Temporal rolling-window decision logic
│
└── tests/
    └── test_drowsiness.py        # Unit tests
```

---

## 3. Discovered Limitations & Critical Deficiencies

### 3.1 Functionality Deficiencies
1. **Missing Image Upload & Analysis Mode:** The web app only processes video uploads. Users cannot upload a static portrait, photo, or eye image to evaluate eye state.
2. **Missing In-Browser Webcam Mode:** Live webcam detection is confined to a separate terminal command launching an OpenCV native window, which is inaccessible from web clients.
3. **No Dual-Eye Detection:** `app.py` picks only one eye crop (left or right) and ignores the other, rather than analyzing both eyes and computing a comprehensive bilateral metric.
4. **No Unified Inference Engine:** Image preprocessing, landmark extraction, and model inference are re-implemented across `realtime_webcam.py` and `app.py` rather than relying on a shared `src/inference.py` module.
5. **No Sample Media:** Users must provide their own driving video or face picture; no sample test images or video clips are bundled for immediate evaluation.

### 3.2 UI/UX & Design Deficiencies
1. **Single-Page Cluttered Layout:** All options are packed into a single linear Streamlit page.
2. **Missing Navigation Hierarchy:** No multi-page structure or sidebar navigation for Project Overview, Architecture, Dataset Explorer, Model Evaluation, and About/Limitations.
3. **Basic Visual Styling:** Relies heavily on default Streamlit white styling with raw inputs and rudimentary text banners instead of a sleek, dark-themed computer vision dashboard.
4. **Lack of Educational Value:** Does not explain the deep learning pipeline, activation functions, transfer learning backbone, or the critical difference between instantaneous eye-closure and temporal drowsiness.

### 3.3 Deep Learning & Architectural Limitations
1. **Activation Functions Undocumented:** MobileNetV3Small relies on Hard-Swish and Hard-Sigmoid alongside ReLU and Sigmoid, but these are neither documented nor explained in the UI.
2. **No PERCLOS Metric:** The temporal engine only measures continuous duration, omitting the standardized PERCLOS (Percentage of Eye Closure over Time) metric standard in automotive safety research.
3. **Dataset Role Confusion:** Users might assume UTA-RLDD video frames can be directly classified by the eye model without understanding domain shift (eye crop vs full face).

---

## 4. Comprehensive Upgrade Plan

### Phase 1: Shared Inference Engine & Backend Enhancement
- Create `src/inference.py`:
  - `InferencePipeline`: Unified interface for image, video frame, and live video stream.
  - Dual-eye localization with MediaPipe Face Mesh + Haar Cascade fallback.
  - Returns bounding boxes, landmarks, left/right eye crops, and individual eye probabilities.
  - Image-level state resolution: Both Eyes Open, Left Eye Closed, Right Eye Closed, Both Closed, or Undetected.
- Enhance `src/drowsiness_engine.py`:
  - Implement PERCLOS calculation over configurable sliding windows (e.g., 60-frame window).
  - Track blink frequency (blinks/minute) and micro-sleep events ($\ge 1.5$–$2.0$s).
- Create `src/dataset_inspector.py`:
  - High-performance cached metadata queries for MRL and UTA-RLDD datasets.

### Phase 2: Multi-Page Web Dashboard Redesign
Implement a modern, professional, 9-section computer vision web application in `app.py`:
1. **🏠 Home / Dashboard:** Hero banner, model readiness indicator, quick-start feature cards, architecture summary, and non-diagnostic disclaimer.
2. **🖼️ Image Analysis:** Interactive photo upload (JPG/PNG), dual-eye cropping, landmark overlays, per-eye predictions, confidence meters, and inference latency.
3. **🎬 Video Analysis:** Multi-format video upload (MP4/AVI/MOV), frame rate sampling, live progress preview, time-series telemetry charts, PERCLOS calculation, closure event timeline, and CSV export.
4. **📹 Live Webcam:** Dedicated webcam interface with live video feed, real-time eye-state prediction, temporal tracking, closure alerts, and session reset.
5. **📖 Project Overview:** Academic deep-dive covering problem statement, objectives, full pipeline workflow, tech stack, and responsible use.
6. **🏛️ System Architecture:** Comprehensive architectural diagrams, 5-tier system decomposition, MobileNetV3Small layer-by-layer breakdown, and **Activation Function Matrix** (Hard-Swish, ReLU, Sigmoid, Hard-Sigmoid).
7. **📊 Model Training & Evaluation:** Test accuracy (96.30%), F1-score (96.22%), confusion matrix heatmap, ROC/PR curves, CPU latency (5.11 ms), and subject-aware splitting analysis.
8. **🔍 Dataset Explorer:** Interactive dataset browser with class distribution bar charts, subject breakdowns, sensor distribution, and sampling statistics for MRL and UTA-RLDD.
9. **ℹ️ About / Limitations:** Environmental caveats (glasses, shadows, low light), physiological variances, and safety disclaimers.

### Phase 3: Sample Assets & Comprehensive Documentation
- Generate curated sample assets in `sample_data/` (open-eye photo, closed-eye photo, driving sequence) so users can immediately test all 3 modes without uploading personal media.
- Create detailed markdown reports in `outputs/`:
  - `outputs/project_audit.md` (this file)
  - `outputs/dataset_inspection.md` (updated with exact file counts)
  - `outputs/model_architecture.md` (exact layer parameters and activation equations)
  - `outputs/evaluation_report.md` (comprehensive performance breakdown)
  - `outputs/training_report.md` (training history and convergence dynamics)
- Update `README.md` with complete Windows PowerShell instructions, architecture diagrams, and reproduction guides.

### Phase 4: Verification & Automated Testing
- Expand `tests/test_drowsiness.py` to cover:
  - Single image inference and dual-eye localization.
  - Bounding box coordinates calculation.
  - PERCLOS computation in `DrowsinessEngine`.
  - Dataset inspector caching and statistics validation.
  - Video analysis frame generator and CSV export format.
- Run all tests to achieve 100% pass rate.
