# Codex Continuation Audit

**Audit date:** 2026-10-20  
**Workspace:** Existing Windows project directory  
**Repository status:** No `.git` directory was present at the workspace root; Git status/history are unavailable.

## Existing structure

- `app.py`: Streamlit dashboard with Home, Image Analysis, Video Analysis, Live Webcam, Project Overview, System Architecture, Model Training & Evaluation, Dataset Explorer, and About pages.
- `src/inference.py`, `src/preprocessing.py`, `src/drowsiness_engine.py`: shared eye detection, preprocessing, model inference, and temporal closure logic.
- `src/model.py`, `train.py`, `evaluate.py`, `src/data_loader.py`: MobileNetV3Small transfer-learning model, training, evaluation, and MRL loading.
- `src/dataset_inspector.py`, `src/video_dataset.py`: local dataset inventory and UTA-RLDD annotation parsing.
- `realtime_webcam.py`: separate local OpenCV webcam process.
- `models/eye_state_model.keras`: existing 9,504,859-byte model checkpoint.
- `outputs/`: inherited dataset, training, model, and evaluation reports, evaluation JSON, plots, and split metadata.
- `archive/data/{train,val,test}/{awake,sleepy}`: local MRL image dataset.
- `UTA-RLDD Dataset/{train,val,test}/{active,fatigue}` and split text files: local UTA-RLDD face-frame dataset.
- `tests/test_drowsiness.py`: existing unit/integration tests.

## Features already present

- Nine-section Streamlit navigation and a coordinated dark dashboard style.
- Image, sampled-video, and temporal analysis paths built on a shared inference engine.
- FaceMesh eye localization with Haar fallback; explicit direct-crop model path and manual ROI crop controls.
- Video timestamp telemetry, closure tracking, PERCLOS values, event display, and CSV export.
- Browser camera still capture and a distinct native OpenCV webcam script.
- MRL training and evaluation scripts, checkpoint, reports, and saved plots.

## Verified local dataset counts

Counts below were read from the actual image folders during this audit. These are image/frame counts, not counts of original driving videos.

| Dataset | Split | Class A | Class B | Total |
|---|---:|---:|---:|---:|
| MRL Eye | Train | awake 25,770 | sleepy 25,167 | 50,937 |
| MRL Eye | Validation | awake 8,591 | sleepy 8,389 | 16,980 |
| MRL Eye | Test | awake 8,591 | sleepy 8,390 | 16,981 |
| MRL Eye | Total | awake 42,952 | sleepy 41,946 | 84,898 |
| UTA-RLDD | Train | active 5,862 | fatigue 3,192 | 9,054 |
| UTA-RLDD | Validation | active 912 | fatigue 912 | 1,824 |
| UTA-RLDD | Test | active 456 | fatigue 453 | 909 |
| UTA-RLDD | Total | active 7,230 | fatigue 4,557 | 11,787 |

The MRL filename subject identifiers show all 37 subjects in each split and an overlap of 37 subjects for every split pair. This is an image-random split, not a subject-independent evaluation; scores may not generalize to unseen people.

Training/evaluation now build an in-memory, deterministic subject-aware partition (`seed=42` by default) across the unchanged image folders. A local inventory using the implemented 70/15/15 subject assignment and class-balance search yields: train 25 subjects (29,028 awake, 28,164 sleepy; 57,192 total), validation 6 (6,897 awake, 6,951 sleepy; 13,848 total), and test 6 (7,027 awake, 6,831 sleepy; 13,858 total). The three subject sets are disjoint. These are partition counts, not training/evaluation results; the test suite and model evaluation could not run here.

UTA-RLDD annotation files contain 6,384 train lines, 1,715 validation lines, and 909 test lines; every listed file reference exists. The train and validation annotation files cover fewer frames than their corresponding folders (9,054 and 1,824). The current `src/video_dataset.py` prioritizes annotation files, so effective train/validation dataset counts through that loader differ from directory inventory counts.

## Incomplete, risky, or not verified

- Continuous webcam capture is not embedded in the browser. `st.camera_input` is snapshot capture. `realtime_webcam.py` opens the camera attached to the machine running the process; both now route frames through `InferenceEngine`.
- The previous evaluation JSON had unverified inherited results; it was regenerated on 2026-10-21 from the current checkpoint by `evaluate.py`, using the generated subject-aware MRL test partition and all 909 annotated UTA test frames. Optional sample limits are available via `--sample-limit` and `--uta-sample-limit`.
- The saved `.keras` archive config was inspected without loading TensorFlow: it identifies a Functional `eye_state_classifier` with a MobileNetV3Small backbone, 128×128×3 input, GAP, dropout 0.3, Dense(64, ReLU), dropout 0.15, and Dense(1, sigmoid). The backbone has 30 layers marked trainable and 125 marked non-trainable; checkpoint compile metadata records Adam at ~1e-5 and binary crossentropy. Serialized config shows hard-swish (`hard_silu`) activations in the backbone, ReLU layers (including SE), and sigmoid output; no separate hard-sigmoid is recorded. Runtime loading/forward inference and parameter counts were not verified. Training history (epochs, data and validation curves) remains unverified.
- No original UTA-RLDD driving-video files are present in the listed extracted tree; the local resource is annotated face frames. Video-upload analysis accepts user-provided video files.
- The project `.venv` runs Python 3.11.9 with OpenCV 4.11.0.86, MediaPipe 1.1.1, TensorFlow 2.17.1, and Streamlit 1.60.0. MediaPipe 1.1.1 no longer exposes the legacy `mp.solutions` API, so the engine uses OpenCV Haar cascades. A quick sample of 40 MRL face frames detected eyes in 39/40; this is a smoke sample, not a detector benchmark. Browser startup and webcam hardware were not exercised.
- The configured decision threshold 0.507 came from the prior validation-F1 analysis (16,980 grayscale eye images): precision 98.25%, recall 98.22%, F1 98.24%, and accuracy 98.26%. Validation subject IDs overlap the inherited training data, so these are not held-out-person or webcam results.
- On 2026-10-21, `python evaluate.py` completed on the current checkpoint. MRL subject-aware test partition: 13,858 eye crops, accuracy 97.85%, sleepy precision 97.64%, recall 98.01%, F1 97.82%; matrix [[6865,162],[136,6695]]. UTA-RLDD face-frame transfer: 909 frames, accuracy 48.29%, F1 53.56%; matrix [[168,288],[182,271]]. The checkpoint's historical training provenance is not tied to the new subject-aware partitions, so the MRL result is not asserted as an unseen-driver score; UTA is a cross-domain check, not a directly comparable eye-state benchmark.

## Changes made

- Replaced hardcoded dataset statistics in `src/dataset_inspector.py` with filesystem counts, discovered image formats, and parsed MRL subject IDs. Empty splits now return zero percentages instead of dividing by zero.
- Added deterministic subject-aware, class-balance-seeking MRL partitioning for training/evaluation without moving or rewriting original dataset files; added a no-overlap test.
- Made the Dataset Explorer charts use those local counts.
- Added 10 MB image and 200 MB video upload caps plus clear image/video decode errors.
- Image inference now requires explicit user selection before a no-detection image may be treated as a direct eye crop, preventing arbitrary portraits from receiving fabricated eye-state outputs.
- Added percentage-coordinate manual eye ROI selection to the Image Analysis page.
- Video upload temporary files are created only for decoding/analysis and removed after use; accepted video extensions are MP4, AVI, MOV.
- Removed unverified benchmark claims from the Home page; the evaluation page now reports the refreshed metrics with split provenance caveats.
- Corrected the webcam page to distinguish browser snapshots from the local OpenCV process.
- Refactored `realtime_webcam.py` to share the app's inference engine; Haar fallback can fill missing FaceMesh eye crops and checks multiple detected faces.
- Updated dashboard palette, cards, focus states, and responsive typography.
- Ran `\.venv\Scripts\python.exe -m unittest discover -s tests -v`: **17 tests passed** (13.0 seconds).
- `py_compile` succeeded for `app.py`, `src/inference.py`, `src/preprocessing.py`, and `realtime_webcam.py`; a brief headless Streamlit run returned `ok` from `/_stcore/health`.
- The Model Training & Evaluation page now charts train/validation accuracy and loss from `outputs/training_history.csv`, shows both current confusion matrices from `outputs/evaluation.json`, and displays their sample counts, label meanings, threshold, timestamp, and provenance caveat. The Architecture page and Home workflow now describe the implemented detector and shared runtime path.

## Remaining work / blockers

- Start Streamlit in `.venv` and manually verify page layout, uploads, and camera permissions on the target Windows machine.
- Training history CSV lacks a run ID or checkpoint hash, so the site labels it as recorded history and does not claim it necessarily belongs to the current checkpoint.
- Retrain on the subject-aware MRL training partition before reporting held-out-subject results; the existing checkpoint learned from a split with subject overlap.
- Continuous browser webcam, annotated video output, frame resolution sampling, model and training history integrity checks, and actual CPU inference measurements remain unverified or unimplemented.
