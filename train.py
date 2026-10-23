"""
Training pipeline for Face-Level Drowsiness/Fatigue Classification using MobileNetV3Small.

Dataset: DPD (Drowsiness Prediction Dataset from Kaggle)
Classes: Active (0) vs Fatigue (1)

Supports:
- Stratified train/val/test splitting
- Pretrained MobileNetV3Small transfer learning + fine-tuning
- Checkpointing, early stopping, LR scheduling
- Training curves and history export
- Model metadata recording for inference verification
- Fast smoke test option (--smoke-test)
"""
import os
import sys
import argparse
import random
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import (
    discover_dpd_images,
    stratified_split,
    create_dataset,
    compute_class_weights,
    get_class_distribution,
    split_manifest_rows,
    prepare_face_crop_cache,
    filter_splits_for_face_crops,
)
from src.model import build_drowsiness_model, unfreeze_and_finetune, get_callbacks


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_and_save_curves(history_df: pd.DataFrame, output_path: str):
    """Plot training and validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history_df['accuracy'], label='Train Accuracy', color='#2563eb', linewidth=2)
    if 'val_accuracy' in history_df:
        axes[0].plot(history_df['val_accuracy'], label='Val Accuracy', color='#16a34a', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy over Epochs', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')

    # Loss
    axes[1].plot(history_df['loss'], label='Train Loss', color='#dc2626', linewidth=2)
    if 'val_loss' in history_df:
        axes[1].plot(history_df['val_loss'], label='Val Loss', color='#ea580c', linewidth=2, linestyle='--')
    axes[1].set_title('Loss over Epochs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Binary Crossentropy')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def train(args):
    start_time = time.time()
    set_seed(args.seed)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    model_output = os.path.abspath(args.model_output)
    artifact_prefix = args.artifact_prefix or ""
    history_csv = os.path.join(config.OUTPUTS_DIR, f"{artifact_prefix}training_history.csv")
    history_json = os.path.join(config.OUTPUTS_DIR, f"{artifact_prefix}training_history.json")
    curves_path = os.path.join(config.OUTPUTS_DIR, f"{artifact_prefix}training_curves.png")
    metadata_output = (config.MODEL_METADATA_PATH if model_output == os.path.abspath(config.MODEL_PATH)
                       else os.path.splitext(model_output)[0] + "_metadata.json")

    print("=" * 65)
    print(" DROWSINESS DETECTION — DPD MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Discover data
    print("\n[Step 1/5] Discovering DPD dataset images...")
    paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
    if not paths:
        raise FileNotFoundError(
            f"No images found in DPD dataset directory: {config.DPD_DATASET_DIR}\n"
            f"Please verify that 'Active Subjects' and 'Fatigue Subjects' folders exist."
        )

    print(f"  Total discovered images: {len(paths)}")
    print(f"  Class distribution: {get_class_distribution(labels)}")

    # 2. Stratified train/val/test split
    splits = stratified_split(
        paths,
        labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        seed=args.seed,
    )

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    print(f"\n  Splits generated:")
    print(f"    Train: {len(train_paths)} images, {get_class_distribution(train_labels)}")
    print(f"    Val:   {len(val_paths)} images, {get_class_distribution(val_labels)}")
    print(f"    Test:  {len(test_paths)} images, {get_class_distribution(test_labels)}")

    # Save split summary and prepare face crops using the runtime detector.
    print("  Localizing faces and preparing reusable face crops...")
    source_to_crop = prepare_face_crop_cache(paths)
    unavailable = [(path, label) for path, label in zip(paths, labels) if path not in source_to_crop]
    pd.DataFrame([
        {"path": path, "label": config.CLASS_NAMES[int(label)], "reason": "face detector did not find a face"}
        for path, label in unavailable
    ]).to_csv(os.path.join(config.OUTPUTS_DIR, "face_crop_exclusions.csv"), index=False)
    splits = filter_splits_for_face_crops(splits, source_to_crop)
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]
    print(f"  Images excluded due to missing face crop: {len(unavailable)}")
    splits_summary = [
        {"split": name, "total": len(split_paths), **get_class_distribution(split_labels)}
        for name, (split_paths, split_labels) in splits.items()
    ]
    splits_csv = os.path.join(config.OUTPUTS_DIR, "splits.csv")
    pd.DataFrame(splits_summary).to_csv(splits_csv, index=False)
    print(f"  Saved usable split summary to: {splits_csv}")
    manifest_path = os.path.join(config.OUTPUTS_DIR, "split_manifest.csv")
    pd.DataFrame(split_manifest_rows(splits, source_to_crop)).to_csv(manifest_path, index=False)
    print(f"  Saved per-image split manifest to: {manifest_path}")

    # Subsample if smoke test or sample limit specified
    if args.smoke_test:
        print("\n [SMOKE TEST MODE]: Using small balanced subset (200 train, 100 val) for rapid verification...")
        train_paths, train_labels = train_paths[:200], train_labels[:200]
        val_paths, val_labels = val_paths[:100], val_labels[:100]
        test_paths, test_labels = test_paths[:100], test_labels[:100]
        args.epochs = min(args.epochs, 2)
        args.fine_tune_epochs = 0
    elif args.max_samples:
        train_paths = train_paths[:args.max_samples]
        train_labels = train_labels[:args.max_samples]
        val_paths = val_paths[:args.max_samples // 4]
        val_labels = val_labels[:args.max_samples // 4]

    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"  Computed class weights: {class_weights}")

    # 3. Build tf.data datasets
    print("\n[Step 2/5] Creating optimized tf.data pipelines...")
    train_model_paths = [source_to_crop[path] for path in train_paths]
    val_model_paths = [source_to_crop[path] for path in val_paths]
    train_ds = create_dataset(train_model_paths, train_labels, batch_size=args.batch_size, augment=True, shuffle=True, cache=False)
    val_ds = create_dataset(val_model_paths, val_labels, batch_size=args.batch_size, augment=False, shuffle=False, cache=False)

    # 4. Build Model
    print(f"\n[Step 3/5] Building MobileNetV3Small (input_shape={config.IMG_SHAPE})...")
    model = build_drowsiness_model(
        input_shape=config.IMG_SHAPE,
        dropout_rate=config.DROPOUT_RATE,
        learning_rate=args.lr,
        freeze_backbone=True,
        use_fallback=args.use_fallback,
    )
    model.summary()

    # 5. Phase 1: Train classification head
    callbacks = get_callbacks(checkpoint_path=model_output)
    print(f"\n[Step 4/5] Training classification head for up to {args.epochs} epochs...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history_head.history.items()}

    # 6. Phase 2: Optional Fine-tuning
    if args.fine_tune_epochs > 0 and not args.use_fallback:
        print(f"\n[Step 5/5] Fine-tuning backbone for {args.fine_tune_epochs} epochs with lr={config.FINE_TUNE_LR}...")
        model = unfreeze_and_finetune(model, fine_tune_lr=config.FINE_TUNE_LR, unfreeze_from=-30)
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        for k, vals in history_finetune.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    # Save final model if not already saved by checkpoint
    if not os.path.exists(model_output):
        model.save(model_output)
    print(f"\n Best model saved to: {model_output}")

    # Save training history as CSV and JSON
    history_df = pd.DataFrame(combined_history)
    history_df.to_csv(history_csv, index=False)
    print(f" Training history CSV saved to: {history_csv}")

    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f" Training history JSON saved to: {history_json}")

    # Plot curves
    plot_and_save_curves(history_df, curves_path)

    # Save model metadata
    elapsed_sec = time.time() - start_time
    best_val_acc = float(max(combined_history.get('val_accuracy', [0.0])))
    best_val_loss = float(min(combined_history.get('val_loss', [999.0])))

    metadata = {
        "model_name": "Custom CNN Drowsiness Classifier" if model.name == "drowsiness_cnn_fallback" else "MobileNetV3Small Drowsiness Classifier",
        "model_architecture": "MobileNetV3Small backbone + GAP + Dense(64, ReLU) + sigmoid binary head" if model.name != "drowsiness_cnn_fallback" else model.name,
        "dataset": "DPD (Drowsiness Prediction Dataset)",
        "dataset_path": os.path.relpath(config.DPD_DATASET_DIR, config.PROJECT_ROOT),
        "task": "Binary Face-Level Fatigue Classification",
        "class_names": config.CLASS_NAMES,
        "class_mapping": {"Active": 0, "Fatigue": 1},
        "input_shape": list(config.IMG_SHAPE),
        "preprocessing": "OpenCV Haar face localization with 15% padding (same detector as inference); crop to RGB; resize to 128x128; scale to [-1, 1]",
        "face_crop_cache_version": config.FACE_CROP_CACHE_VERSION,
        "face_crop_excluded_images": len(unavailable),
        "split_manifest": "outputs/split_manifest.csv",
        "split_strategy": "Deterministic stratified image-level 70/15/15 split (seeded); no subject IDs available",
        "random_seed": args.seed,
        "checkpoint_version": "2026-10-23-dpd-mobilenetv3small-facecrop-v2",
        "split_class_distribution": {
            split_name: get_class_distribution(split_labels)
            for split_name, (_split_paths, split_labels) in splits.items()
        },
        "training_config": {
            "head_epochs_requested": args.epochs,
            "fine_tune_epochs_requested": args.fine_tune_epochs,
            "batch_size": args.batch_size,
            "head_learning_rate": args.lr,
            "fine_tune_learning_rate": config.FINE_TUNE_LR,
            "augmentation_on_training_only": True,
            "class_weights": {config.CLASS_NAMES[int(key)]: float(value) for key, value in class_weights.items()},
        },
        "total_train_samples": len(train_paths),
        "total_val_samples": len(val_paths),
        "total_test_samples": len(test_paths),
        "batch_size": args.batch_size,
        "epochs_trained": len(combined_history.get('loss', [])),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "training_duration_seconds": round(elapsed_sec, 2),
        "timestamp": datetime.now().isoformat(),
        "model_path": os.path.relpath(model_output, config.PROJECT_ROOT),
    }

    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f" Model metadata saved to: {metadata_output}")

    print("\n Training successfully completed!")
    print(f" Total time: {round(elapsed_sec / 60.0, 2)} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face-level drowsiness model on DPD dataset.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of head training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on subset")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum training samples to use")
    parser.add_argument("--use-fallback", action="store_true", help="Force lightweight custom CNN instead of MobileNetV3")
    parser.add_argument("--model-output", type=str, default=config.MODEL_PATH, help="Checkpoint output path (use a separate path for experiments)")
    parser.add_argument("--artifact-prefix", type=str, default="", help="Prefix for history and curve artifacts")

    parsed_args = parser.parse_args()
    train(parsed_args)
