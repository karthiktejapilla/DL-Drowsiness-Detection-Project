"""Select a fatigue decision threshold using only the fixed validation split."""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from src.data_loader import (discover_dpd_images, stratified_split, prepare_face_crop_cache,
                             filter_splits_for_face_crops, create_dataset)
from src.model import load_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/threshold_tuning.json")
    args = parser.parse_args()

    paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
    crops = prepare_face_crop_cache(paths)
    val_paths, val_labels = filter_splits_for_face_crops(
        stratified_split(paths, labels, seed=config.RANDOM_SEED), crops
    )["val"]
    inputs = [crops[path] for path in val_paths]
    dataset = create_dataset(inputs, val_labels, batch_size=args.batch_size,
                             augment=False, shuffle=False, cache=False)
    model = load_trained_model(args.model_path)
    scores = model.predict(dataset, verbose=0).ravel()
    y_true = np.asarray(val_labels)

    rows = []
    for threshold in np.arange(0.25, 0.751, 0.005):
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        rows.append({
            "threshold": round(float(threshold), 3),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "fatigue_precision": float(precision),
            "fatigue_recall": float(recall),
            "fatigue_f1": float(f1),
        })

    best_f1 = max(rows, key=lambda row: (row["fatigue_f1"], row["accuracy"]))
    best_accuracy = max(rows, key=lambda row: (row["accuracy"], row["fatigue_f1"]))
    result = {
        "split": "validation",
        "sample_count": len(y_true),
        "model_path": os.path.abspath(args.model_path),
        "selection_rule": "maximize fatigue F1, then accuracy",
        "selected_threshold": best_f1["threshold"],
        "best_f1_metrics": best_f1,
        "best_accuracy_metrics": best_accuracy,
        "threshold_sweep": rows,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in (
        "sample_count", "selection_rule", "selected_threshold", "best_f1_metrics", "best_accuracy_metrics"
    )}, indent=2))
    print(f"Full threshold sweep saved to {output}")


if __name__ == "__main__":
    main()
