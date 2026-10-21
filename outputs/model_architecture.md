# Model Architecture: Source Configuration and Verification Status

**Checkpoint:** `models/eye_state_model.keras` (9,504,859 bytes)  
**Checkpoint inspection:** The `.keras` archive's serialized `config.json` was read with Python's standard ZIP/JSON libraries. Runtime loading and a forward pass were not possible because TensorFlow is unavailable.

## Architecture configured in source

The serialized checkpoint config identifies a Functional model named `eye_state_classifier`, input `(None, 128, 128, 3)`, and a `MobileNetV3Small` backbone. The head is GlobalAveragePooling2D, Dropout(0.30), Dense(64, ReLU), Dropout(0.15), and Dense(1, sigmoid). The binary output represents the sleepy/closed class probability. The source builder has a custom-CNN fallback, but this checkpoint's serialized config does identify the MobileNetV3Small graph.

The checkpoint compile config records binary crossentropy with Adam at approximately `1e-5`, consistent with fine-tuning. The source defaults are batch size 16, up to 15 head epochs plus 10 fine-tuning epochs, and fine-tuning the last 30 backbone layers at `1e-5`. The serialized backbone has `trainable=True`; 30 nested layers are marked trainable and 125 marked non-trainable. The config does not establish the complete training history or executed epoch count. `train.py` configures best-validation-loss checkpointing, early stopping (patience 5), and learning-rate reduction.

## Activations

- **ReLU:** explicitly configured in the custom classification head, `Dense(64, activation="relu")`.
- **Sigmoid:** explicitly configured in the binary output, `Dense(1, activation="sigmoid")`.
- **Hard-swish (`hard_silu` in config):** appears in 18 serialized MobileNetV3Small Activation layers.
- **ReLU:** appears in MobileNetV3Small layers, including SE blocks, and the explicit Dense(64) head.
- **Squeeze-and-excitation gate:** the config contains two linear Conv2D layers, ReLU, and Multiply. It does not record a separate hard-sigmoid activation; no hard-sigmoid claim is made.
- **Sigmoid:** explicit final Dense(1) head activation.
- **Fallback model:** source uses ReLU in convolutional and Dense layers, and sigmoid in its output layer.

## Preprocessing

`src/preprocessing.py` converts BGR to RGB, resizes to 128×128, casts to float32, scales pixels to `[-1, 1]`, and adds a batch dimension. The model builder disables MobileNetV3 built-in preprocessing, making this external normalization intentional in the source pipeline.

## Parameter counts and model summary

No verified parameter count is reported because the available runtime has neither TensorFlow nor h5py. Run the following in the project's installed environment to inspect and execute the actual checkpoint:

```powershell
python -c "import tensorflow as tf; m=tf.keras.models.load_model('models/eye_state_model.keras'); m.summary()"
```
