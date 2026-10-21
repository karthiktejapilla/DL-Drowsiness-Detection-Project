"""
Training pipeline for Eye-State Classification using MobileNetV3Small.

Supports:
- Data discovery and split verification
- ImageNet transfer learning + fine-tuning
- Checkpointing, early stopping, LR scheduling
- Training curves and history export
- Fast smoke test option (--smoke-test)
"""
import os
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import discover_subject_aware_splits, create_dataset, compute_class_weights, get_class_distribution, get_subject_split_info
from src.model import build_eye_state_model, unfreeze_and_finetune, get_callbacks


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
    set_seed(args.seed)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print(" DROWSINESS DETECTION — EYE-STATE MODEL TRAINING")
    print("=" * 60)

    # 1. Discover data
    print("\n[Step 1/5] Discovering dataset files...")
    subject_splits = discover_subject_aware_splits(seed=args.seed)
    train_paths, train_labels, train_subjs = subject_splits["train"]
    val_paths, val_labels, val_subjs = subject_splits["val"]
    test_paths, test_labels, test_subjs = subject_splits["test"]
    if not train_paths or not val_paths or not test_paths:
        raise FileNotFoundError("MRL image splits are empty. Check archive/data/{train,val,test}/{awake,sleepy} before training.")

    print(f"  Train: {len(train_paths)} samples, distribution: {get_class_distribution(train_labels)}")
    print(f"  Val:   {len(val_paths)} samples, distribution: {get_class_distribution(val_labels)}")
    print(f"  Test:  {len(test_paths)} samples, distribution: {get_class_distribution(test_labels)}")
    subject_info = get_subject_split_info(train_subjs, val_subjs, test_subjs)
    print(f"  Subject overlaps: train/val={len(subject_info['train_val_overlap'])}, train/test={len(subject_info['train_test_overlap'])}, val/test={len(subject_info['val_test_overlap'])}")

    # Record splits metadata
    splits_summary = [
        {"split": "train", "total": len(train_paths), **get_class_distribution(train_labels), "unique_subjects": len(set(train_subjs))},
        {"split": "val", "total": len(val_paths), **get_class_distribution(val_labels), "unique_subjects": len(set(val_subjs))},
        {"split": "test", "total": len(test_paths), **get_class_distribution(test_labels), "unique_subjects": len(set(test_subjs))},
    ]
    splits_csv = os.path.join(config.OUTPUTS_DIR, "splits.csv")
    pd.DataFrame(splits_summary).to_csv(splits_csv, index=False)
    print(f"  Split summary saved to {splits_csv}")

    # Pair and shuffle so both classes (awake and sleepy) are evenly distributed
    combined_train = list(zip(train_paths, train_labels, train_subjs))
    random.shuffle(combined_train)
    train_paths, train_labels, train_subjs = [list(t) for t in zip(*combined_train)]

    combined_val = list(zip(val_paths, val_labels, val_subjs))
    random.shuffle(combined_val)
    val_paths, val_labels, val_subjs = [list(t) for t in zip(*combined_val)]

    # Subsample if smoke test or sample limit specified
    if args.smoke_test:
        print("\n [SMOKE TEST MODE]: Using balanced subset (400 train, 200 val) for rapid verification...")
        train_paths, train_labels = train_paths[:400], train_labels[:400]
        val_paths, val_labels = val_paths[:200], val_labels[:200]
        args.epochs = 2
        args.fine_tune_epochs = 0
    elif args.max_samples:
        train_paths, train_labels = train_paths[:args.max_samples], train_labels[:args.max_samples]
        val_paths, val_labels = val_paths[:args.max_samples // 4], val_labels[:args.max_samples // 4]

    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"  Computed class weights: {class_weights}")

    # 2. Build tf.data datasets
    print("\n[Step 2/5] Creating tf.data data pipelines...")
    train_ds = create_dataset(train_paths, train_labels, batch_size=args.batch_size, augment=True, shuffle=True, cache=False)
    val_ds = create_dataset(val_paths, val_labels, batch_size=args.batch_size, augment=False, shuffle=False, cache=False)

    # 3. Build Model
    print(f"\n[Step 3/5] Building MobileNetV3Small (input_shape={config.IMG_SHAPE})...")
    model = build_eye_state_model(
        input_shape=config.IMG_SHAPE,
        dropout_rate=config.DROPOUT_RATE,
        learning_rate=args.lr,
        freeze_backbone=True,
        use_fallback=args.use_fallback,
    )
    model.summary()

    # 4. Phase 1: Train classification head
    callbacks = get_callbacks(checkpoint_path=config.EYE_STATE_MODEL_PATH)
    print(f"\n[Step 4/5] Training classification head for up to {args.epochs} epochs...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = history_head.history

    # 5. Phase 2: Optional Fine-tuning
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
        for k in combined_history.keys():
            combined_history[k].extend(history_finetune.history.get(k, []))

    # Save final model if not already saved by checkpoint
    if not os.path.exists(config.EYE_STATE_MODEL_PATH):
        model.save(config.EYE_STATE_MODEL_PATH)
    print(f"\n Best model saved to: {config.EYE_STATE_MODEL_PATH}")

    # Save training history
    history_df = pd.DataFrame(combined_history)
    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f" Training history saved to: {history_path}")

    # Plot and save curves
    curves_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    plot_and_save_curves(history_df, curves_path)

    print("\n Training process completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train eye-state classification model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of head training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on subset")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum training samples to use")
    parser.add_argument("--use-fallback", action="store_true", help="Force lightweight custom CNN instead of MobileNetV3")

    parsed_args = parser.parse_args()
    train(parsed_args)
