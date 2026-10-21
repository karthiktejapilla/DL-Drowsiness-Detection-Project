"""
Evaluation script for Eye-State Classification and Video/Face-level Drowsiness Analysis.

Generates:
- outputs/evaluation.json (detailed metrics, per-class stats, latency)
- outputs/confusion_matrix.png (visual confusion matrix)
- UTA-RLDD evaluation on held-out test split
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import discover_subject_aware_splits, create_dataset
from src.video_dataset import load_uta_rldd_split
from src.model import load_trained_model


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix as a high-resolution heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def evaluate_eye_state_model(model, test_paths, test_labels, batch_size=config.BATCH_SIZE):
    """Evaluate eye state model on MRL test split."""
    print("\n--- Evaluating Eye-State Classification (MRL Test Set) ---")
    print(f"Evaluating {len(test_paths)} test images...")

    test_ds = create_dataset(test_paths, test_labels, batch_size=batch_size, augment=False, shuffle=False, cache=False)

    # Measure CPU inference latency
    start_time = time.time()
    raw_preds = model.predict(test_ds, verbose=1)
    inference_duration = time.time() - start_time

    latency_per_sample_ms = (inference_duration / len(test_paths)) * 1000.0
    fps = len(test_paths) / inference_duration

    # Binary threshold at 0.5
    binary_preds = (raw_preds.ravel() >= config.CONFIDENCE_THRESHOLD).astype(int)
    y_true = np.array(test_labels)

    # Compute metrics
    acc = accuracy_score(y_true, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average='binary', zero_division=0)
    prec_per_cls, rec_per_cls, f1_per_cls, support = precision_recall_fscore_support(y_true, binary_preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, binary_preds)

    print(f"\nResults on MRL Test Split:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  Latency:   {latency_per_sample_ms:.2f} ms/sample ({fps:.1f} FPS on CPU)")
    print(f"  Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path, title="Eye State Confusion Matrix (0=Awake, 1=Sleepy)")

    metrics = {
        "dataset": "MRL Eye Dataset (Forked)",
        "test_samples": len(test_paths),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "cpu_latency_ms": round(float(latency_per_sample_ms), 2),
        "cpu_fps": round(float(fps), 2),
        "per_class": {
            config.CLASS_NAMES[0]: {
                "precision": round(float(prec_per_cls[0]), 4),
                "recall": round(float(rec_per_cls[0]), 4),
                "f1": round(float(f1_per_cls[0]), 4),
                "support": int(support[0]),
            },
            config.CLASS_NAMES[1]: {
                "precision": round(float(prec_per_cls[1]), 4),
                "recall": round(float(rec_per_cls[1]), 4),
                "f1": round(float(f1_per_cls[1]), 4),
                "support": int(support[1]),
            }
        },
        "confusion_matrix": cm.tolist()
    }

    return metrics


def evaluate_uta_rldd(model, max_samples=None):
    """
    Evaluate on UTA-RLDD test set.
    Note: Model A classifies eye states (open vs closed).
    UTA-RLDD contains full face crops labeled as active (0) vs fatigue (1).
    Evaluating Model A on full faces or extracted face-frames validates domain shift
    and documents performance limitations honestly.
    """
    print("\n--- Evaluating on UTA-RLDD (Face-level Fatigue Domain) ---")
    paths, labels, subjects = load_uta_rldd_split(split_name="test", sample_limit=max_samples)

    if not paths:
        print("  Warning: No UTA-RLDD test samples available.")
        return {"status": "unavailable", "note": "UTA-RLDD test split not found or empty."}

    print(f"  Loaded {len(paths)} UTA-RLDD test samples across subjects: {set(subjects)}")
    uta_ds = create_dataset(paths, labels, batch_size=config.BATCH_SIZE, augment=False, shuffle=False, cache=False)

    raw_preds = model.predict(uta_ds, verbose=0)
    binary_preds = (raw_preds.ravel() >= config.CONFIDENCE_THRESHOLD).astype(int)
    y_true = np.array(labels)

    acc = accuracy_score(y_true, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, binary_preds)

    print(f"  UTA-RLDD Domain Test Accuracy: {acc * 100:.2f}% (F1: {f1 * 100:.2f}%)")
    print(f"  Note: UTA-RLDD labels reflect whole-face fatigue vs active, whereas Model A was trained on eye crops.")

    return {
        "dataset": "UTA-RLDD (Fatigue vs Active)",
        "evaluated_samples": len(paths),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "task_distinction_note": "Model A classifies eye closure. UTA-RLDD tests whole-face drowsiness transferability."
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate eye state model.")
    parser.add_argument("--model-path", type=str, default=config.EYE_STATE_MODEL_PATH)
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit test samples for fast evaluation")
    parser.add_argument("--uta-sample-limit", type=int, default=None, help="Optional limit for UTA-RLDD evaluation; default evaluates the full annotated test split")
    args = parser.parse_args()

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    model = load_trained_model(args.model_path)

    # 1. MRL Eye Dataset evaluation
    subject_splits = discover_subject_aware_splits(seed=config.RANDOM_SEED)
    test_paths, test_labels, test_subjs = subject_splits["test"]
    if not test_paths:
        raise FileNotFoundError("MRL test partition is empty. Check the local MRL image folders.")

    if args.sample_limit:
        test_paths = test_paths[:args.sample_limit]
        test_labels = test_labels[:args.sample_limit]

    eye_metrics = evaluate_eye_state_model(model, test_paths, test_labels)

    # 2. UTA-RLDD evaluation
    uta_metrics = evaluate_uta_rldd(model, max_samples=args.uta_sample_limit)

    # Combine into evaluation.json
    final_evaluation = {
        "model_architecture": "MobileNetV3Small (ImageNet pretrained)",
        "input_shape": list(config.IMG_SHAPE),
        "eye_state_classification": eye_metrics,
        "video_face_drowsiness_analysis": uta_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    eval_json_path = os.path.join(config.OUTPUTS_DIR, "evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(final_evaluation, f, indent=4)
    print(f"\nComplete evaluation report saved to: {eval_json_path}")


if __name__ == "__main__":
    main()
