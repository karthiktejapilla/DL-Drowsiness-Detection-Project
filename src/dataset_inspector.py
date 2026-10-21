"""
Dataset Inspector and Analytics Module for Human Drowsiness Detection.

Provides:
- Efficient, cached discovery of local dataset files and split distributions
- Statistical breakdown of classes, subjects, sensor types, and formats
- Verified dataset counts without memory overload
- Sample image retrieval for dataset gallery rendering
"""
import os
import glob
from collections import Counter
from typing import Dict, List, Optional, Any
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import parse_mrl_filename


class DatasetInspector:
    """High-performance metadata querying and statistics for local datasets."""

    _cached_mrl_stats: Optional[Dict[str, Any]] = None
    _cached_uta_stats: Optional[Dict[str, Any]] = None

    @classmethod
    def get_mrl_statistics(cls) -> Dict[str, Any]:
        """Return verified MRL Eye Dataset statistics."""
        if cls._cached_mrl_stats is not None:
            return cls._cached_mrl_stats

        mrl_root = config.MRL_DATA_DIR
        exists = os.path.isdir(mrl_root)
        splits_data = {}
        subjects = set()
        extensions = Counter()
        for split in ("train", "val", "test"):
            counts = {}
            for label in ("awake", "sleepy"):
                folder = os.path.join(mrl_root, split, label)
                files = [p for p in glob.iglob(os.path.join(folder, "**", "*"), recursive=True) if os.path.isfile(p)]
                counts[label] = len(files)
                for path in files:
                    extensions[os.path.splitext(path)[1].lower()] += 1
                    metadata = parse_mrl_filename(os.path.basename(path))
                    if metadata:
                        subjects.add(metadata["subject"])
            counts["total"] = counts["awake"] + counts["sleepy"]
            splits_data[split] = counts

        total_images = sum(s["total"] for s in splits_data.values())
        total_awake = sum(s["awake"] for s in splits_data.values())
        total_sleepy = sum(s["sleepy"] for s in splits_data.values())

        stats = {
            "name": "MRL Eye Dataset (Forked Edition)",
            "role": "Eye-State Binary Classification (Primary CNN Model)",
            "path": os.path.relpath(mrl_root, config.PROJECT_ROOT),
            "exists": exists,
            "total_images": total_images,
            "total_awake": total_awake,
            "total_sleepy": total_sleepy,
            "awake_pct": round((total_awake / total_images) * 100, 2) if total_images else 0.0,
            "sleepy_pct": round((total_sleepy / total_images) * 100, 2) if total_images else 0.0,
            "unique_subjects": len(subjects),
            "subject_range": f"{min(subjects)} - {max(subjects)}" if subjects else "Unavailable",
            "image_format": ", ".join(f"{ext or '[no extension]'} ({count})" for ext, count in sorted(extensions.items())) or "No images found",
            "native_resolution": "See source images; not sampled by this inventory",
            "target_resolution": f"{config.IMG_WIDTH}x{config.IMG_HEIGHT}x3 RGB",
            "sensors": [
                {"id": "01", "name": "Intel RealSense", "res": "640x480"},
                {"id": "02", "name": "IDS Imaging", "res": "1280x1024"},
                {"id": "03", "name": "Aptina Imaging", "res": "752x480"},
            ],
            "splits": splits_data,
            "classes": ["0 = Awake (Eyes Open)", "1 = Sleepy (Eyes Closed)"],
            "notes": (
                "Folder hierarchy dictates canonical ground truth ('awake' vs 'sleepy'). "
                "The dataset features varied lighting, gender, glasses wearers, and lens reflections."
            )
        }
        cls._cached_mrl_stats = stats
        return stats

    @classmethod
    def get_uta_rldd_statistics(cls) -> Dict[str, Any]:
        """Return verified UTA-RLDD Dataset statistics."""
        if cls._cached_uta_stats is not None:
            return cls._cached_uta_stats

        uta_root = config.UTA_RLDD_DIR
        exists = os.path.isdir(uta_root)
        splits_data = {}
        extensions = Counter()
        annotation_counts = {}
        for split in ("train", "val", "test"):
            counts = {}
            for label in ("active", "fatigue"):
                folder = os.path.join(uta_root, split, label)
                files = [p for p in glob.iglob(os.path.join(folder, "**", "*"), recursive=True) if os.path.isfile(p)]
                counts[label] = len(files)
                for path in files:
                    extensions[os.path.splitext(path)[1].lower()] += 1
            counts["total"] = counts["active"] + counts["fatigue"]
            splits_data[split] = counts
            annotation_path = os.path.join(uta_root, f"{split}.txt")
            if os.path.isfile(annotation_path):
                with open(annotation_path, encoding="utf-8") as annotation_file:
                    annotation_counts[split] = sum(1 for line in annotation_file if line.strip())
            else:
                annotation_counts[split] = None

        total_frames = sum(s["total"] for s in splits_data.values())
        total_active = sum(s["active"] for s in splits_data.values())
        total_fatigue = sum(s["fatigue"] for s in splits_data.values())

        stats = {
            "name": "UTA-RLDD (Real-Life Drowsiness Dataset)",
            "role": "Video-Derived Whole-Face Drowsiness & Domain Shift Evaluation",
            "path": os.path.relpath(uta_root, config.PROJECT_ROOT),
            "exists": exists,
            "total_frames": total_frames,
            "total_active": total_active,
            "total_fatigue": total_fatigue,
            "active_pct": round((total_active / total_frames) * 100, 2) if total_frames else 0.0,
            "fatigue_pct": round((total_fatigue / total_frames) * 100, 2) if total_frames else 0.0,
            "image_format": ", ".join(f"{ext or '[no extension]'} ({count})" for ext, count in sorted(extensions.items())) or "No images found",
            "native_resolution": "See source images; not sampled by this inventory",
            "annotation_format": "train.txt, val.txt, test.txt (relative_path label)",
            "splits": splits_data,
            "annotation_lines": annotation_counts,
            "classes": ["0 = Active (Alert/Awake)", "1 = Fatigue (Drowsy Video State)"],
            "notes": (
                "These are face-level video frames rather than isolated infrared eye crops. "
                "Their active/fatigue labels are a distinct task from MRL open/closed eye labels."
            )
        }
        cls._cached_uta_stats = stats
        return stats

    @classmethod
    def get_sample_images(cls, dataset: str = "mrl", split: str = "test", count_per_class: int = 3) -> Dict[str, List[str]]:
        """Retrieve sample image paths for UI gallery preview."""
        samples = {}
        if dataset.lower() == "mrl":
            base = os.path.join(config.MRL_DATA_DIR, split)
            for c in ["awake", "sleepy"]:
                c_dir = os.path.join(base, c)
                if os.path.isdir(c_dir):
                    files = glob.glob(os.path.join(c_dir, "*.png"))[:count_per_class]
                    samples[c] = files
        elif dataset.lower() == "uta":
            base = os.path.join(config.UTA_RLDD_DIR, split)
            for c in ["active", "fatigue"]:
                c_dir = os.path.join(base, c)
                if os.path.isdir(c_dir):
                    files = glob.glob(os.path.join(c_dir, "*.jpg"))[:count_per_class]
                    samples[c] = files
        return samples
