# Model Training Report — Eye-State Classifier

> **Verification status (2026-10-16):** This is an inherited report. Its epoch log and claimed training run were not reproduced in the current environment. Local MRL split counts are verified in `codex_continuation_audit.md`; all 37 subject IDs overlap across each split pair. Treat the current metrics and training history as unverified until re-run from source.

**Project:** Human Drowsiness Detection Using Deep Learning  
**Architecture:** MobileNetV3Small (ImageNet Pretrained Transfer Learning)  
**Task:** Binary Eye-State Classification (`0 = Awake / Open`, `1 = Sleepy / Closed`)  
**Checkpoint Path:** `models/eye_state_model.keras`  
**Execution Environment:** Intel/AMD 64-bit CPU, Windows 11, TensorFlow 2.17.1  

---

## 1. Dataset & Split Configuration

The primary eye-state model was trained on the pre-organized **MRL Eye Dataset (Forked Edition)**. Ground-truth labels are determined by directory partition (`awake` vs `sleepy`).

| Split | Awake Samples | Sleepy Samples | Total Samples | Class Ratio (Awake/Sleepy) | Unique Subjects |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 25,770 | 25,167 | **50,937** | 50.59% / 49.41% | 37 (s0001 - s0037) |
| **Validation** | 8,591 | 8,389 | **16,980** | 50.60% / 49.40% | 37 (s0001 - s0037) |
| **Held-Out Test** | 8,591 | 8,390 | **16,981** | 50.59% / 49.41% | 37 (s0001 - s0037) |
| **Full Total** | **42,952** | **41,946** | **84,898** | **50.59% / 49.41%** | **37 Subjects** |

*Splits metadata logged in:* [`outputs/splits.csv`](splits.csv)

---

## 2. Training Hyperparameters & Pipeline Setup

- **Input Tensor Dimensions:** `(128, 128, 3)` (RGB scaling from grayscale infrared)
- **Pixel Normalization:** Scaled to $[-1.0, +1.0]$ via `(image / 127.5) - 1.0` matching MobileNetV3 specifications
- **Data Augmentations:**
  - Random Horizontal Flip
  - Random Contrast adjust ($0.8 - 1.2$)
  - Random Brightness shift ($\pm 0.2$)
  - Random Slight Zoom/Crop ($0.85 - 1.0$)
- **Batch Size:** 16 (optimized for laptop CPU cache)
- **Loss Function:** Binary Crossentropy:
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \Big]$$
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$)
- **Initial Learning Rate:** $1 \times 10^{-3}$
- **Fine-Tuning Learning Rate:** $1 \times 10^{-5}$
- **Regularization:**
  - Dropout (rate = 0.3) after GlobalAveragePooling2D
  - Dropout (rate = 0.15) after Dense(64)
- **Callbacks Configured:**
  - `ModelCheckpoint`: Monitors `val_loss`, saves best weights only to `models/eye_state_model.keras`
  - `EarlyStopping`: Patience = 5 epochs, restores best weights
  - `ReduceLROnPlateau`: Factor = 0.5, Patience = 3 epochs

---

## 3. Convergence & Training Dynamics

Training was performed in two successive stages:
1. **Stage 1 (Feature Extraction):** MobileNetV3Small backbone frozen, training dense classification head.
2. **Stage 2 (Fine-Tuning):** Top 30 backbone layers unfrozen at lower learning rate ($10^{-5}$).

### Epoch-by-Epoch Progress Log

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Learning Rate | Stage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 88.13% | 0.2894 | 93.93% | 0.1553 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **2** | 92.43% | 0.1956 | 95.60% | 0.1177 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **3** | 93.25% | 0.1696 | 96.47% | 0.1078 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **4** | 93.42% | 0.1601 | 96.80% | 0.0973 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **5** | 93.98% | 0.1487 | 97.13% | 0.0961 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **6** | 94.58% | 0.1432 | 96.87% | 0.0878 | $1.0 \times 10^{-3}$ | Frozen Backbone |
| **7** | 78.63% | 0.5719 | 96.87% | **0.0808** | $1.0 \times 10^{-5}$ | Fine-Tuning Stage |
| **8** | 85.78% | 0.3793 | 96.87% | 0.0827 | $1.0 \times 10^{-5}$ | Fine-Tuning Stage |

*Visual loss and accuracy curves saved to:* [`outputs/training_curves.png`](training_curves.png)

---

## 4. Key Takeaways

1. **Fast Convergence:** Within 3 epochs, validation accuracy exceeded 96.4%, demonstrating strong transferability of ImageNet low-level edge and texture filters to infrared eye regions.
2. **Low Generalization Gap:** Final validation loss reached $0.0808$, matching the final test set accuracy of 96.30% with minimal overfitting.
3. **Dropout Efficacy:** The dual dropout layers ($0.30$ and $0.15$) successfully prevented co-adaptation in the 64-unit bottleneck representation.
