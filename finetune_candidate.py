"""Fine-tune a copy of the active checkpoint without changing it."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.data_loader import (discover_dpd_images, stratified_split, prepare_face_crop_cache,
                             filter_splits_for_face_crops, create_dataset, compute_class_weights)
from src.model import load_trained_model, get_callbacks


def main():
    paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
    crops = prepare_face_crop_cache(paths)
    splits = filter_splits_for_face_crops(
        stratified_split(paths, labels, seed=config.RANDOM_SEED), crops
    )
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    train_ds = create_dataset([crops[path] for path in train_paths], train_labels,
                              batch_size=16, augment=True, shuffle=True, cache=False)
    val_ds = create_dataset([crops[path] for path in val_paths], val_labels,
                            batch_size=32, augment=False, shuffle=False, cache=False)

    model = load_trained_model(config.MODEL_PATH)
    backbone = next(layer for layer in model.layers
                    if hasattr(layer, "layers") and len(layer.layers) > 10)
    backbone.trainable = True
    train_from = max(0, len(backbone.layers) - 20)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= train_from and not isinstance(layer, tf.keras.layers.BatchNormalization)
    trainable_backbone = sum(1 for layer in backbone.layers if layer.trainable)
    print(f"Fine-tuning {trainable_backbone} non-BN layers from the last 20 backbone layers")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])
    output_path = os.path.join(config.MODELS_DIR, "drowsiness_model_finetune_candidate.keras")
    callbacks = get_callbacks(output_path)
    history = model.fit(train_ds, validation_data=val_ds, epochs=6,
                        class_weight=compute_class_weights(train_labels),
                        callbacks=callbacks, verbose=2)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    history_path = os.path.join(config.OUTPUTS_DIR, "finetune_candidate_history.csv")
    pd.DataFrame({key: [float(v) for v in values]
                  for key, values in history.history.items()}).to_csv(history_path, index=False)
    print(f"Candidate checkpoint: {output_path}")
    print(f"Validation history: {history_path}")


if __name__ == "__main__":
    main()
