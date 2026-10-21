# Codex Master Audit & Remediation Plan: Human Drowsiness Detection

**Audit Date:** 2026-10-21  
**Operating Environment:** Windows, Python 3.11.9, TensorFlow 2.17.1, OpenCV 4.11.0, Streamlit 1.60.0  
**Trained Model:** `models/eye_state_model.keras` (MobileNetV3Small transfer learning, 128x128x3 input, ~9.5 MB)

---

## 1. What Already Works

- **Trained Model Artifact:** `models/eye_state_model.keras` is a functional MobileNetV3Small checkpoint trained on MRL Eye Dataset images. Direct inference on isolated eye crops (`sample_eye_awake.png`, `sample_eye_sleepy.png`) yields 100% confidence with correct label alignment (Class 0: Awake/Open, Class 1: Sleepy/Closed).
- **Core Preprocessing Contract:** Grayscale/BGR to 3-channel replicated format with `[-1, 1]` normalization matches the MobileNetV3 backbone expectations across training and inference.
- **Temporal Engine Logic:** `DrowsinessEngine` correctly differentiates transient blinks (<0.4s) from sustained ocular closure (>=2.0s) when valid eye-state predictions are continuously supplied.
- **Local Datasets Intact:**
  - MRL Eye Dataset: 84,898 grayscale images (42,952 awake, 41,946 sleepy) across 37 subjects.
  - UTA-RLDD Dataset: 11,787 annotated video frames across train/val/test splits.
- **Automated Test Suite:** 17 unit and integration tests in `tests/test_drowsiness.py` currently pass.

---

## 2. What Is Incomplete, Fragile, or Broken

1. **MediaPipe FaceMesh Initialization Failure (Silent Fallback):**
   - MediaPipe version 1.1.1 is installed in `.venv`. In MediaPipe 1.1.1, `mediapipe.solutions` is completely removed in favor of the new Tasks API.
   - `InferenceEngine._init_mediapipe()` attempts `mp.solutions.face_mesh`, which raises `AttributeError: module 'mediapipe' has no attribute 'solutions'` on every single run.
   - The application was silently falling back to a fragile, single Haar cascade configuration.

2. **Single-Cascade Face Detection Failure on Fatigue/Rotated Faces:**
   - The previous fallback relied solely on `haarcascade_frontalface_default.xml`.
   - On fatigue driver images (e.g. `sample_data/sample_face_fatigue.jpg`), `haarcascade_frontalface_default.xml` detected **0 faces**, reporting "NO FACE/EYES DETECTED".
   - In contrast, testing showed that `haarcascade_frontalface_alt2.xml` and CLAHE-equalized preprocessing successfully detect the face `[273, 97, 181, 181]`.

3. **Closed Eyes Cause Complete Detection Failure (The "Closed-Eye Paradox"):**
   - Haar eye cascades (`haarcascade_eye.xml`) are trained to detect open irises and sclera. When a driver's eyes are closed or squinting in fatigue, eye cascades frequently find **0 eyes**.
   - In the previous code, when 0 eyes were detected inside a detected face, `eyes_count` became 0, which set `overall_class = None`.
   - In `DrowsinessEngine.update()`, `eye_state_pred is None` immediately reset `closure_start_time = None` and `current_closure_duration = 0.0`.
   - **Crucial flaw:** When the driver shut their eyes to sleep, detection failed, resetting the drowsiness timer and completely preventing drowsiness alerts from triggering!

4. **Lack of Anatomical Ocular Geometry Fallback:**
   - If a face is detected, human facial anatomy places the ocular regions predictably in the upper 20%-52% of the face height.
   - Without an anatomical ocular fallback, missing an eye cascade bounding box resulted in complete drop of the face.

5. **Inconsistent Eyeglass Handling:**
   - Drivers wearing glasses or experiencing glare failed standard eye detection. The available `haarcascade_eye_tree_eyeglasses.xml` was not being leveraged as an ensemble member.

6. **UI/UX Limitations:**
   - The existing Streamlit interface looks utilitarian, lacks fluid visual hierarchy, uses basic metric boxes, and does not provide an intuitive experience for live camera, video timeline, or manual ROI crop.
   - Manual eye selection relied on complex percentage sliders rather than a clean visual crop helper.
   - The video and camera dashboards did not offer real-time visual badges, responsive cards, or interactive diagnostics.

---

## 3. Root Cause Analysis

### Face & Eye Detection Failures
- **Root Cause A:** MediaPipe legacy API mismatch (`mp.solutions` missing in modern MediaPipe).
- **Root Cause B:** Fragile Haar face cascade (`frontalface_default` fails on angled or shadowed faces).
- **Root Cause C:** Eye cascade dependency: Closed eyes do not have open-eye Haar patterns. Without anatomical geometric ocular estimation within the face bounding box, closed eyes cause detection failure rather than closed-eye classification.

### Prediction Discrepancies
- The model itself is well-calibrated (predicting 0 for awake, 1 for sleepy).
- When eyes were closed, no crop was extracted, returning `None`.
- In live webcam, contrast and resolution variations in laptop cameras degrade raw Haar cascades unless adaptive histogram equalization (CLAHE) is applied.

---

## 4. Planned Engineering & Design Fixes

### Phase 2 & 3: Robust Multi-Stage Computer Vision Pipeline
- Implement **Cascaded Multi-Detector Face Ensemble**:
  1. `haarcascade_frontalface_alt2.xml` (primary robust frontal/semi-frontal)
  2. `haarcascade_frontalface_default.xml` (secondary standard)
  3. `haarcascade_frontalface_alt.xml` (tertiary)
  4. `haarcascade_profileface.xml` (for profile/head-turned drivers)
  5. CLAHE (Contrast-Limited Adaptive Histogram Equalization) pre-pass for challenging low-light or backlit frames.
- Implement **Multi-Tier Eye Localization**:
  1. Primary: Eyeglass-aware and standard eye cascades (`haarcascade_eye_tree_eyeglasses.xml`, `haarcascade_eye.xml`, `haarcascade_lefteye_2splits.xml`, `haarcascade_righteye_2splits.xml`).
  2. Ocular Symmetry Fallback: If 1 eye is detected with high confidence, estimate the symmetrical counterpart eye.
  3. Anatomical Ocular Extraction: If a face is detected but eyes cannot be isolated via cascades (e.g., closed eyes during sleep!), extract bilateral ocular ROIs using facial anthropometric proportions (upper 20%-52% of face height, 16%-46% and 54%-84% face width).
  4. Direct Eye Crop Mode: For standalone eye images (MRL format) or user manual crop.
- Confidence Thresholding & Uncertainty:
  - If prediction probability is between 0.45 and 0.55 or crop quality is ambiguous, flag as "Uncertain / Indeterminate" rather than forcing false positives.

### Phase 4: Shared Unified Inference Pipeline
- Ensure `src/inference.py` is the single source of truth for:
  - Single image upload
  - Sample images
  - Video frame processing
  - Browser camera capture
  - Standalone `realtime_webcam.py` script

### Phase 6: Complete UI/UX Redesign
- Modern, high-aesthetic dark theme with deep navy/charcoal background (`#080e1a`, `#0f172a`), teal/cyan primary accents (`#06b6d4`, `#14b8a6`), subtle emerald (`#10b981`) for alert/awake states, amber (`#f59e0b`) for warning/closing, and crimson (`#ef4444`) for critical drowsiness.
- Rebuild all pages:
  - **Dashboard:** Hero section, KPI cards, quick-start workflow cards, live system health status.
  - **Image Analysis:** Clean drag-and-drop, visual dual-panel comparison, per-eye ocular telemetry cards, interactive ROI selector with intuitive visual presets, confidence gauges.
  - **Video Analysis:** Upload preview, processing progress, interactive telemetry charts (PERCLOS, closure duration timeline), event table, CSV download.
  - **Live Webcam:** Real-time snapshot analysis, interactive diagnostic guide, native OpenCV monitor launcher with audible alarm instructions.
  - **Project Overview, Architecture, Dataset Explorer, Model Evaluation:** Polished cards, real filesystem counts, genuine confusion matrix and training curves, architectural diagrams.

### Phase 10: Rigorous Automated Testing
- Expand unit tests to verify:
  - Multi-detector face detection on challenging samples.
  - Anatomical ocular extraction when eyes are closed.
  - Closed-eye detection on fatigue faces (`sample_face_fatigue.jpg`, `image_0010.jpg`).
  - Preprocessing consistency across all input sources.
