"""
Model definition for drowsiness/fatigue face classification.

Uses MobileNetV3Small with ImageNet pretrained weights
for transfer learning on the DPD Drowsiness Prediction Dataset.

Task: Binary classification — Active (0) vs Fatigue (1)
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_drowsiness_model(
    input_shape: tuple = config.IMG_SHAPE,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
    freeze_backbone: bool = True,
    use_fallback: bool = False,
) -> keras.Model:
    """
    Build the drowsiness face classification model.

    Architecture:
    - MobileNetV3Small backbone (ImageNet pretrained)
    - Global Average Pooling
    - Dropout
    - Dense(64, relu)
    - Dropout
    - Dense(1, sigmoid) for binary classification

    Args:
        input_shape: Input image shape (H, W, C).
        dropout_rate: Dropout rate.
        learning_rate: Learning rate for Adam optimizer.
        freeze_backbone: Whether to freeze backbone weights.
        use_fallback: If True, use a lightweight custom CNN instead.

    Returns:
        Compiled Keras model.
    """
    if use_fallback:
        return _build_fallback_model(input_shape, dropout_rate, learning_rate)

    try:
        # Load MobileNetV3Small without classification head
        backbone = keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            include_preprocessing=False,  # We handle preprocessing ourselves
        )

        if freeze_backbone:
            backbone.trainable = False
        else:
            backbone.trainable = True

        # Build the model
        inputs = keras.Input(shape=input_shape)
        x = backbone(inputs, training=not freeze_backbone)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs, name='drowsiness_classifier')

    except Exception as e:
        print(f"Error loading MobileNetV3Small: {e}")
        print("Falling back to lightweight custom CNN...")
        return _build_fallback_model(input_shape, dropout_rate, learning_rate)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def _build_fallback_model(
    input_shape: tuple,
    dropout_rate: float,
    learning_rate: float,
) -> keras.Model:
    """
    Lightweight custom CNN fallback if MobileNetV3 weights unavailable.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        # Block 4
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate / 2),
        layers.Dense(1, activation='sigmoid'),
    ], name='drowsiness_cnn_fallback')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def unfreeze_and_finetune(
    model: keras.Model,
    fine_tune_lr: float = config.FINE_TUNE_LR,
    unfreeze_from: int = -30,
) -> keras.Model:
    """
    Unfreeze the last N layers of the backbone for fine-tuning.

    Args:
        model: The trained model with frozen backbone.
        fine_tune_lr: Learning rate for fine-tuning (should be lower).
        unfreeze_from: Unfreeze layers from this index onwards (negative = from end).

    Returns:
        Re-compiled model ready for fine-tuning.
    """
    # Find the backbone layer
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        print("Warning: Could not find backbone layer for fine-tuning.")
        return model

    # Unfreeze backbone
    backbone.trainable = True

    # Freeze all layers except the last N
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    trainable_count = sum(1 for layer in backbone.layers if layer.trainable)
    print(f"Fine-tuning: {trainable_count}/{len(backbone.layers)} backbone layers unfrozen")

    # Re-compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def get_callbacks(checkpoint_path: str = config.MODEL_PATH) -> list:
    """
    Create training callbacks.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks


def load_trained_model(model_path: str = config.MODEL_PATH) -> keras.Model:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded Keras model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Please run train.py first to train the model."
        )

    is_active_checkpoint = os.path.abspath(model_path) == os.path.abspath(config.MODEL_PATH)
    metadata = None
    if is_active_checkpoint:
        if not os.path.isfile(config.MODEL_METADATA_PATH):
            raise FileNotFoundError(
                f"Model metadata is missing: {config.MODEL_METADATA_PATH}. "
                "The active checkpoint cannot be verified; retrain with train.py."
            )
        import json
        with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        expected_mapping = {name: index for index, name in enumerate(config.CLASS_NAMES)}
        if "DPD" not in str(metadata.get("dataset", "")):
            raise ValueError("Active model metadata does not identify the DPD dataset.")
        if metadata.get("class_names") != config.CLASS_NAMES:
            raise ValueError("Active model class names do not match config.CLASS_NAMES.")
        if metadata.get("class_mapping", expected_mapping) != expected_mapping:
            raise ValueError("Active model class mapping does not match the configured output order.")
        if metadata.get("input_shape") != list(config.IMG_SHAPE):
            raise ValueError("Active model input shape does not match the configured preprocessing shape.")
        if metadata.get("face_crop_cache_version") != config.FACE_CROP_CACHE_VERSION:
            raise ValueError(
                "Active checkpoint was not trained with the current face-crop preprocessing. "
                "Retrain with train.py before using it for face-crop inference."
            )

    model = keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != tuple(config.IMG_SHAPE):
        raise ValueError(
            f"Checkpoint input shape {model.input_shape[1:]} does not match {config.IMG_SHAPE}."
        )
    if model.output_shape[-1] != 1:
        raise ValueError(
            f"Expected a single sigmoid output for Active/Fatigue binary classification; "
            f"got output shape {model.output_shape}."
        )
    print(f"Loaded model from: {model_path}")
    return model
