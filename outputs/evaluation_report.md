# Comprehensive Model Evaluation & Benchmarks Report

> **Verification status (2026-10-16):** Inherited evaluation output; not independently reproduced in the current environment. Reported sample sizes also conflict with the current `evaluate.py` defaults (full MRL test split and at most 300 UTA frames). Do not treat the following metrics as validated until re-running evaluation and saving fresh outputs. The existing MRL split has subject overlap across train, validation, and test.

**Date:** October 2026  
**Project:** Human Drowsiness Detection Using Deep Learning  
**Trained Model:** `models/eye_state_model.keras`  
**Evaluation Target:** Held-out Test Splits (MRL Eye Dataset & UTA-RLDD)  
**Inference Hardware:** Laptop CPU (Intel/AMD x86_64, Windows 11)  

---

## 1. MRL Eye Dataset (Held-out Test Split Benchmark)

Evaluated on $N = 2,000$ balanced test samples from the held-out test split of the MRL Eye Dataset ($1,028$ Awake, $972$ Sleepy).

### Key Performance Indicators (KPIs)

| Metric | Measured Value | Standard Target | Assessment |
|:---|:---:|:---:|:---|
| **Overall Accuracy** | **96.30%** | $\ge 90.0\%$ | **Exceptional** |
| **Precision (Sleepy)** | **95.54%** | $\ge 92.0\%$ | High confidence, minimal false closures |
| **Recall (Sleepy)** | **96.91%** | $\ge 95.0\%$ | Catches 942 of 972 genuine eye closures |
| **F1-Score (Sleepy)** | **96.22%** | $\ge 93.0\%$ | Harmonic balance of precision & recall |
| **CPU Latency per Sample** | **5.11 ms** | $< 20\text{ ms}$ | Sub-millisecond order of magnitude |
| **CPU Throughput** | **195.6 FPS** | $> 30\text{ FPS}$ | Over $6\times$ real-time camera requirements |

---

## 2. Per-Class Performance Breakdown

| Class Name | Definition | Support (Samples) | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Awake (0)** | Eyes Open | 1,028 | 97.04% | 95.72% | 96.38% |
| **Sleepy (1)** | Eyes Closed | 972 | 95.54% | 96.91% | 96.22% |
| **Macro Average** | Balanced Mean | 2,000 | 96.29% | 96.32% | 96.30% |
| **Weighted Average** | Weighted by Size | 2,000 | 96.31% | 96.30% | 96.30% |

---

## 3. Confusion Matrix

```
                     Predicted Awake (0)    Predicted Sleepy (1)
Actual Awake (0)            984 (95.7%)            44 (4.3%)
Actual Sleepy (1)            30 (3.1%)            942 (96.9%)
```

- **True Negatives (Awake correctly predicted):** 984
- **False Positives (Awake misclassified as Sleepy):** 44 (4.28%)
- **False Negatives (Sleepy misclassified as Awake):** 30 (3.09%)
- **True Positives (Sleepy correctly predicted):** 942

*Visual confusion matrix saved to:* [`outputs/confusion_matrix.png`](confusion_matrix.png)

---

## 4. Cross-Domain Evaluation: UTA-RLDD Dataset

To rigorously assess whether an eye-crop specialist classifier transfers directly to whole-face video frames, Model A was tested on 909 face frames from UTA-RLDD:

- **Accuracy:** 51.16%
- **Confusion Matrix:**
  - True Active: 358 | False Fatigue: 98
  - False Active: 346 | True Fatigue: 107
- **F1-Score:** 32.52%

### Critical Engineering Insight: Domain Shift
This experimental outcome confirms that **Model A is specifically an eye-crop classifier**, not a full-face classifier. Full-face images contain cheeks, noses, foreheads, and backgrounds that dilute the eye signal. Therefore, the production architecture implements a **Dual-Stage Pipeline**:
1. Stage 1: Face Mesh / Haar detector extracts tight eye regions.
2. Stage 2: MobileNetV3 classifies the localized eye regions.
3. Stage 3: Temporal Decision Engine assesses prolonged closure across video frames.
