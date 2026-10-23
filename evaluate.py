"""
Evaluation pipeline for Face-Level Drowsiness/Fatigue Classification on DPD Dataset.

Generates:
- outputs/evaluation.json (detailed metrics, per-class stats, latency, ROC-AUC)
- outputs/confusion_matrix.png (visual confusion matrix)
- outputs/roc_curve.png (ROC curve with AUC score)
- outputs/evaluation_report.md (readable summary report)
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import (
    discover_dpd_images,
    stratified_split,
    create_dataset,
    get_class_distribution,
    split_manifest_rows,
    prepare_face_crop_cache,
    filter_splits_for_face_crops,
)
from src.model import load_trained_model


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, title: str = "DPD Fatigue Confusion Matrix"):
    """Plot and save confusion matrix heatmap."""
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


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str):
    """Plot and save Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {output_path}")
    return roc_auc


def evaluate_dpd_model(model, test_paths, test_labels, model_input_paths=None, batch_size=config.BATCH_SIZE,
                       threshold=config.CONFIDENCE_THRESHOLD, output_dir=config.OUTPUTS_DIR):
    """Evaluate drowsiness model on DPD test split."""
    print("\n" + "=" * 60)
    print(" EVALUATING FACE-LEVEL DROWSINESS MODEL (DPD TEST SET)")
    print("=" * 60)
    print(f"Evaluating {len(test_paths)} test images...")

    test_ds = create_dataset(model_input_paths or test_paths, test_labels, batch_size=batch_size, augment=False, shuffle=False, cache=False)

    # Measure CPU inference latency
    start_time = time.time()
    raw_preds = model.predict(test_ds, verbose=1)
    inference_duration = time.time() - start_time

    latency_per_sample_ms = (inference_duration / len(test_paths)) * 1000.0
    fps = len(test_paths) / inference_duration

    # Binary threshold at 0.5
    raw_scores = raw_preds.ravel()
    binary_preds = (raw_scores >= threshold).astype(int)
    y_true = np.array(test_labels)

    # Compute metrics
    acc = accuracy_score(y_true, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average='binary', zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_true, binary_preds, average='macro', zero_division=0)[2]
    weighted_f1 = precision_recall_fscore_support(y_true, binary_preds, average='weighted', zero_division=0)[2]
    prec_per_cls, rec_per_cls, f1_per_cls, support = precision_recall_fscore_support(y_true, binary_preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, binary_preds)
    roc_fpr, roc_tpr, _ = roc_curve(y_true, raw_scores)

    print(f"\nResults on DPD Test Split:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  Latency:   {latency_per_sample_ms:.2f} ms/sample ({fps:.1f} FPS on CPU)")
    print(f"  Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path, title="DPD Confusion Matrix (Active vs Fatigue)")

    # Plot ROC curve
    roc_path = os.path.join(output_dir, "roc_curve.png")
    roc_auc = plot_roc_curve(y_true, raw_scores, roc_path)

    metrics = {
        "dataset": "DPD (Drowsiness Prediction Dataset)",
        "task": "Binary Face-Level Fatigue Classification",
        "test_samples": len(test_paths),
        "face_crop_excluded_images": int(getattr(config, "FACE_CROP_EXCLUDED_COUNT", 0)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "positive_class": "Fatigue",
        "decision_threshold": float(threshold),
        "roc_auc": round(float(roc_auc), 4),
        "cpu_latency_ms": round(float(latency_per_sample_ms), 2),
        "cpu_fps": round(float(fps), 2),
        "evaluation_batch_size": int(batch_size),
        "performance_measurement_scope": "Batched test-set model.predict including decode/preprocessing pipeline; excludes face detection and live-camera capture.",
        "roc_curve_points": [
            {"false_positive_rate": float(fpr), "true_positive_rate": float(tpr)}
            for fpr, tpr in zip(roc_fpr, roc_tpr)
        ],
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

    misclassified = [
        {"path": path, "actual": config.CLASS_NAMES[int(actual)],
         "predicted": config.CLASS_NAMES[int(predicted)], "fatigue_probability": float(score)}
        for path, actual, predicted, score in zip(test_paths, y_true, binary_preds, raw_scores)
        if actual != predicted
    ]
    pd.DataFrame(misclassified).to_csv(
        os.path.join(output_dir, "misclassified_examples.csv"), index=False
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate drowsiness model.")
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH)
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit test samples for fast evaluation")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD, help="Fatigue probability decision threshold")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUTS_DIR, help="Directory for evaluation outputs")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model = load_trained_model(args.model_path)

    # 1. Discover DPD data and generate reproducible test split
    paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
    if not paths:
        raise FileNotFoundError(f"DPD dataset directory is empty or not found: {config.DPD_DATASET_DIR}")

    source_to_crop = prepare_face_crop_cache(paths)
    splits = stratified_split(paths, labels, seed=config.RANDOM_SEED)
    unavailable = [(path, label) for path, label in zip(paths, labels) if path not in source_to_crop]
    config.FACE_CROP_EXCLUDED_COUNT = len(unavailable)
    pd.DataFrame([
        {"path": path, "label": config.CLASS_NAMES[int(label)], "reason": "face detector did not find a face"}
        for path, label in unavailable
    ]).to_csv(os.path.join(output_dir, "face_crop_exclusions.csv"), index=False)
    splits = filter_splits_for_face_crops(splits, source_to_crop)
    pd.DataFrame(split_manifest_rows(splits, source_to_crop)).to_csv(
        os.path.join(output_dir, "split_manifest.csv"), index=False
    )
    test_paths, test_labels = splits["test"]

    if args.sample_limit:
        test_paths = test_paths[:args.sample_limit]
        test_labels = test_labels[:args.sample_limit]

    test_model_inputs = [source_to_crop[path] for path in test_paths]
    metrics = evaluate_dpd_model(model, test_paths, test_labels, model_input_paths=test_model_inputs,
                                 batch_size=args.batch_size, threshold=args.threshold, output_dir=output_dir)

    # Save outputs/evaluation.json
    final_evaluation = {
        "model_architecture": "MobileNetV3Small (ImageNet pretrained)",
        "input_shape": list(config.IMG_SHAPE),
        "evaluation_metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    eval_json_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(final_evaluation, f, indent=4)
    print(f"\nComplete evaluation JSON saved to: {eval_json_path}")

    metadata_path = config.MODEL_METADATA_PATH
    if os.path.isfile(metadata_path) and os.path.abspath(args.model_path) == os.path.abspath(config.MODEL_PATH):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["evaluation_metrics"] = metrics
        metadata["evaluation_split"] = "held-out stratified test split"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated checkpoint metadata with test metrics: {metadata_path}")

    # Save outputs/evaluation_report.md
    report_md_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# DPD Model Evaluation Report\n\n")
        f.write(f"- **Evaluated At**: {final_evaluation['timestamp']}\n")
        f.write(f"- **Dataset**: {metrics['dataset']}\n")
        f.write(f"- **Test Samples**: {metrics['test_samples']}\n")
        f.write(f"- **Images excluded (no face crop found)**: {metrics['face_crop_excluded_images']}\n")
        f.write(f"- **Accuracy**: {metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"- **Precision**: {metrics['precision'] * 100:.2f}%\n")
        f.write(f"- **Recall**: {metrics['recall'] * 100:.2f}%\n")
        f.write(f"- **F1 Score**: {metrics['f1_score'] * 100:.2f}%\n")
        f.write(f"- **Macro F1**: {metrics['macro_f1']:.4f}\n")
        f.write(f"- **Weighted F1**: {metrics['weighted_f1']:.4f}\n")
        f.write("- **Positive class for binary metrics**: Fatigue\n")
        f.write(f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n")
        f.write(f"- **Decision Threshold**: {metrics['decision_threshold']:.3f} (selected on validation split)\n")
        f.write(f"- **CPU Inference Latency**: {metrics['cpu_latency_ms']:.2f} ms/frame ({metrics['cpu_fps']:.1f} FPS)\n\n")
        f.write(f"- **Evaluation Batch Size**: {metrics['evaluation_batch_size']}\n\n")
        f.write("## Per-Class Performance\n\n")
        f.write("| Class | Precision | Recall | F1 Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for cls_name, cls_vals in metrics['per_class'].items():
            f.write(f"| {cls_name} | {cls_vals['precision']*100:.2f}% | {cls_vals['recall']*100:.2f}% | {cls_vals['f1']*100:.2f}% | {cls_vals['support']} |\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"                Predicted Active    Predicted Fatigue\n")
        f.write(f"Actual Active:       {metrics['confusion_matrix'][0][0]:<15} {metrics['confusion_matrix'][0][1]}\n")
        f.write(f"Actual Fatigue:      {metrics['confusion_matrix'][1][0]:<15} {metrics['confusion_matrix'][1][1]}\n")
        f.write("```\n")
    print(f"Readable report saved to: {report_md_path}")


if __name__ == "__main__":
    main()
