"""
Dataset Inspector and Analytics Module for Human Drowsiness Detection.

Provides:
- Efficient, cached discovery of local DPD dataset files and split distributions
- Statistical breakdown of classes, file types, and balance
- Sample image retrieval for dataset gallery rendering
"""
import os
import glob
import csv
import json
from collections import Counter
from typing import Dict, List, Optional, Any
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DatasetInspector:
    """High-performance metadata querying and statistics for the DPD dataset."""

    _cached_dpd_stats: Optional[Dict[str, Any]] = None

    @classmethod
    def get_dpd_statistics(cls) -> Dict[str, Any]:
        """Return verified DPD Dataset statistics."""
        if cls._cached_dpd_stats is not None:
            return cls._cached_dpd_stats

        dpd_root = config.DPD_DATASET_DIR
        active_dir = config.DPD_ACTIVE_DIR
        fatigue_dir = config.DPD_FATIGUE_DIR

        exists = os.path.isdir(dpd_root)
        active_files = [p for p in glob.glob(os.path.join(active_dir, "*.*")) if os.path.isfile(p)] if os.path.isdir(active_dir) else []
        fatigue_files = [p for p in glob.glob(os.path.join(fatigue_dir, "*.*")) if os.path.isfile(p)] if os.path.isdir(fatigue_dir) else []

        total_active = len(active_files)
        total_fatigue = len(fatigue_files)
        total_images = total_active + total_fatigue

        extensions = Counter()
        for p in active_files + fatigue_files:
            extensions[os.path.splitext(p)[1].lower()] += 1

        # Sample resolution from first available image
        sample_res = "Unknown"
        sample_channels = 3
        if active_files:
            img = cv2.imread(active_files[0])
            if img is not None:
                sample_res = f"{img.shape[1]}x{img.shape[0]}"
                sample_channels = img.shape[2] if len(img.shape) > 2 else 1

        # Prefer the actual latest split manifest, which reflects detector misses.
        split_counts = {
            name: {"total": 0, "active": 0, "fatigue": 0}
            for name in ("train", "val", "test")
        }
        manifest_path = os.path.join(config.OUTPUTS_DIR, "split_manifest.csv")
        if os.path.isfile(manifest_path):
            with open(manifest_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = row.get("split")
                    label = row.get("label", "").lower()
                    if name in split_counts and label in ("active", "fatigue"):
                        split_counts[name]["total"] += 1
                        split_counts[name][label] += 1
        else:
            for name, ratio in (("train", config.TRAIN_RATIO), ("val", config.VAL_RATIO), ("test", config.TEST_RATIO)):
                total = int(total_images * ratio)
                split_counts[name] = {"total": total, "active": total // 2, "fatigue": total - total // 2}

        excluded_count = 0
        cache_manifest_path = os.path.join(config.FACE_CROP_CACHE_DIR, "cache_manifest.json")
        try:
            with open(cache_manifest_path, encoding="utf-8") as f:
                cache_data = json.load(f)
            excluded_count = len(cache_data.get("failed_items", {}))
        except (OSError, json.JSONDecodeError):
            pass

        stats = {
            "name": "DPD (Drowsiness Prediction Dataset)",
            "role": "Face-Level Binary Drowsiness Classification (Active vs Fatigue)",
            "source": "Kaggle (rakibuleceruet/drowsiness-prediction-dataset)",
            "path": os.path.relpath(dpd_root, config.PROJECT_ROOT),
            "exists": exists,
            "total_images": total_images,
            "total_active": total_active,
            "total_fatigue": total_fatigue,
            "active_pct": round((total_active / total_images) * 100, 2) if total_images else 0.0,
            "fatigue_pct": round((total_fatigue / total_images) * 100, 2) if total_images else 0.0,
            "balance": "Balanced directory counts" if total_active == total_fatigue else "Imbalanced directory counts",
            "image_format": ", ".join(f"{ext or '[no ext]'} ({count})" for ext, count in sorted(extensions.items())) or "No images found",
            "native_resolution": "Varied (source images are resized during preprocessing)",
            "target_resolution": f"{config.IMG_WIDTH}x{config.IMG_HEIGHT}x{config.IMG_CHANNELS} RGB",
            "splits": split_counts,
            "face_crop_excluded": excluded_count,
            "classes": ["0 = Active (directory label)", "1 = Fatigue (directory label)"],
            "notes": (
                "Source images are full-size portraits at varied resolutions. The pipeline uses padded detector-generated face crops. "
                "Detailed annotation protocol was not independently verified."
            )
        }
        cls._cached_dpd_stats = stats
        return stats

    @classmethod
    def get_sample_images(cls, count_per_class: int = 4) -> Dict[str, List[str]]:
        """Retrieve sample image paths for UI gallery preview."""
        samples = {}
        for c_name, c_dir in [("Active", config.DPD_ACTIVE_DIR), ("Fatigue", config.DPD_FATIGUE_DIR)]:
            if os.path.isdir(c_dir):
                files = [p for p in glob.glob(os.path.join(c_dir, "*.jpg")) if os.path.isfile(p)][:count_per_class]
                if not files:
                    files = [p for p in glob.glob(os.path.join(c_dir, "*.*")) if os.path.isfile(p)][:count_per_class]
                samples[c_name] = files
            else:
                samples[c_name] = []
        return samples
