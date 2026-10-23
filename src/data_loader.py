"""
Data loading utilities for the DPD Drowsiness Prediction Dataset.

Handles:
- Recursive image discovery from Active Subjects / Fatigue Subjects directories
- Stratified train/val/test splitting
- Duplicate detection and corrupt image checking
- tf.data.Dataset creation with efficient batching and augmentation
- Class distribution reporting
"""
import os
import glob
import hashlib
import json
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import decode_and_preprocess, crop_face_from_frame


def discover_dpd_images(dataset_dir: str = config.DPD_DATASET_DIR) -> Tuple[List[str], List[int]]:
    """
    Recursively discover images in the DPD dataset.

    Expected structure:
        dataset_dir/Active Subjects/*.jpg
        dataset_dir/Fatigue Subjects/*.jpg

    Returns:
        Tuple of (image_paths, labels) where Active=0, Fatigue=1
    """
    image_paths = []
    labels = []

    class_dirs = [
        (os.path.join(dataset_dir, "Active Subjects"), 0),    # Active = 0
        (os.path.join(dataset_dir, "Fatigue Subjects"), 1),    # Fatigue = 1
    ]

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    missing_dirs = [class_dir for class_dir, _ in class_dirs if not os.path.isdir(class_dir)]
    if missing_dirs:
        raise FileNotFoundError(
            "DPD dataset must contain both class directories. Missing: "
            + ", ".join(missing_dirs)
        )

    for class_dir, label in class_dirs:

        for root, _dirs, files in os.walk(class_dir):
            _dirs.sort()
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, fname)
                    image_paths.append(full_path)
                    labels.append(label)

    return image_paths, labels


def check_image_readable(path: str) -> bool:
    """Verify an image file exists and can be decoded."""
    try:
        raw = tf.io.read_file(path)
        tf.image.decode_image(raw, channels=3)
        return True
    except Exception:
        return False


def find_corrupt_images(paths: List[str], sample_limit: int = 500) -> List[str]:
    """
    Check a sample of images for corruption.

    Args:
        paths: List of image file paths.
        sample_limit: Maximum images to check (0 = all).

    Returns:
        List of paths that failed decoding.
    """
    check_paths = paths if sample_limit == 0 else paths[:sample_limit]
    corrupt = []
    for p in check_paths:
        if not check_image_readable(p):
            corrupt.append(p)
    return corrupt


def find_exact_duplicates(paths: List[str], sample_limit: int = 2000) -> List[Tuple[str, str]]:
    """
    Find exact duplicate images by file hash.

    Args:
        paths: List of image file paths.
        sample_limit: Maximum number of files to hash (0 = all).

    Returns:
        List of (path_a, path_b) duplicate pairs.
    """
    check_paths = paths if sample_limit == 0 else paths[:sample_limit]
    hash_map: Dict[str, str] = {}
    duplicates = []

    for p in check_paths:
        try:
            h = hashlib.md5(open(p, 'rb').read()).hexdigest()
            if h in hash_map:
                duplicates.append((hash_map[h], p))
            else:
                hash_map[h] = p
        except Exception:
            continue

    return duplicates


def stratified_split(
    paths: List[str],
    labels: List[int],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Create stratified train/val/test splits.

    Ensures proportional class representation in each split.
    Shuffles data deterministically using the provided seed.

    Args:
        paths: List of image paths.
        labels: List of corresponding labels.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys,
        each mapping to (paths, labels) tuples.
    """
    rng = random.Random(seed)

    # Group by class
    class_groups: Dict[int, List[str]] = {}
    for path, label in zip(paths, labels):
        class_groups.setdefault(label, []).append(path)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for label, group_paths in sorted(class_groups.items()):
        shuffled = list(group_paths)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_paths.extend(shuffled[:train_end])
        train_labels.extend([label] * train_end)

        val_paths.extend(shuffled[train_end:val_end])
        val_labels.extend([label] * (val_end - train_end))

        test_paths.extend(shuffled[val_end:])
        test_labels.extend([label] * (n - val_end))

    # Shuffle within each split
    def _shuffle_together(p, l):
        combined = list(zip(p, l))
        rng.shuffle(combined)
        if combined:
            return [list(x) for x in zip(*combined)]
        return [], []

    train_paths, train_labels = _shuffle_together(train_paths, train_labels)
    val_paths, val_labels = _shuffle_together(val_paths, val_labels)
    test_paths, test_labels = _shuffle_together(test_paths, test_labels)

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def prepare_face_crop_cache(
    image_paths: List[str],
    cache_dir: str = config.FACE_CROP_CACHE_DIR,
    dataset_dir: str = config.DPD_DATASET_DIR,
) -> Dict[str, str]:
    """Cache the same padded face crops used by runtime inference.

    Large source images are face-localized once, then reused during every epoch.
    A failed image or face detection is recorded for explicit exclusion; the loader
    never falls back to resizing a whole portrait as though it were a face crop.
    """
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            cache_manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        cache_manifest = {}
    if cache_manifest.get("version") != config.FACE_CROP_CACHE_VERSION:
        cache_manifest = {"version": config.FACE_CROP_CACHE_VERSION, "items": {}}

    previous_items = cache_manifest.get("items", {})
    previous_failures = cache_manifest.get("failed_items", {})
    new_items = {}
    source_to_crop = {}
    failures = []

    ordered_paths = sorted(set(os.path.abspath(p) for p in image_paths))
    for index, source_path in enumerate(ordered_paths, start=1):
        try:
            stat = os.stat(source_path)
        except OSError as exc:
            failures.append((source_path, f"cannot stat source: {exc}"))
            continue

        old = previous_items.get(source_path, {})
        crop_path = old.get("crop_path")
        reusable = (
            old.get("source_size") == stat.st_size
            and old.get("source_mtime_ns") == stat.st_mtime_ns
            and os.path.isfile(crop_path or "")
        )

        if reusable:
            new_items[source_path] = old
            source_to_crop[source_path] = crop_path
            continue

        old_failure = previous_failures.get(source_path, {})
        if (
            isinstance(old_failure, dict)
            and old_failure.get("source_size") == stat.st_size
            and old_failure.get("source_mtime_ns") == stat.st_mtime_ns
        ):
            failures.append((source_path, old_failure.get("reason", "face detector did not find a face")))
            continue

        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if image is None:
            failures.append((source_path, "OpenCV could not decode the source image"))
            continue
        face_crop, bbox, method = crop_face_from_frame(image)
        if face_crop is None or bbox is None:
            failures.append((source_path, "face detector did not find a face"))
            continue

        relative = os.path.relpath(source_path, dataset_dir)
        crop_path = os.path.abspath(os.path.join(cache_dir, relative))
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        if not cv2.imwrite(crop_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            failures.append((source_path, f"could not write face crop to {crop_path}"))
            continue

        item = {
            "crop_path": crop_path,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "face_bbox_xyxy": [int(v) for v in bbox],
            "detection_method": method,
        }
        new_items[source_path] = item
        source_to_crop[source_path] = crop_path
        if index % 1000 == 0 or index == len(ordered_paths):
            print(f"  Face crops prepared: {index}/{len(ordered_paths)}", flush=True)

    cache_manifest = {
        "version": config.FACE_CROP_CACHE_VERSION,
        "items": new_items,
        "failed_items": {
            path: {
                "reason": reason,
                "source_size": os.path.getsize(path) if os.path.isfile(path) else None,
                "source_mtime_ns": os.stat(path).st_mtime_ns if os.path.isfile(path) else None,
            }
            for path, reason in failures
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cache_manifest, f, indent=2)

    if failures:
        examples = "\n".join(f"  - {path}: {reason}" for path, reason in failures[:5])
        print(
            f"  WARNING: face crops unavailable for {len(failures)} of {len(ordered_paths)} images; "
            "these files must be excluded and are recorded in outputs/face_crop_exclusions.csv.\n" + examples,
            flush=True,
        )
    return source_to_crop


def filter_splits_for_face_crops(splits, source_to_crop):
    """Keep only images with a valid shared train/runtime face crop."""
    filtered = {}
    for split_name, (paths, labels) in splits.items():
        available = [(path, label) for path, label in zip(paths, labels) if path in source_to_crop]
        filtered[split_name] = (
            [path for path, _ in available],
            [label for _, label in available],
        )
    return filtered


def get_class_distribution(labels: List[int]) -> Dict[str, int]:
    """
    Report class distribution using config class names.

    Args:
        labels: List of integer labels.

    Returns:
        Dictionary mapping class name to count.
    """
    counts = Counter(labels)
    return {config.CLASS_NAMES[k]: v for k, v in sorted(counts.items())}


def split_manifest_rows(
    splits: Dict[str, Tuple[List[str], List[int]]],
    source_to_crop: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Create a reproducible per-image manifest for the generated split."""
    rows = []
    for split_name in ("train", "val", "test"):
        paths, labels = splits[split_name]
        rows.extend(
            {
                "path": path,
                "model_input_path": (source_to_crop or {}).get(path, path),
                "label": config.CLASS_NAMES[int(label)],
                "label_index": int(label),
                "split": split_name,
            }
            for path, label in zip(paths, labels)
        )
    return rows


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Compute class weights to handle imbalance.

    Args:
        labels: List of integer labels.

    Returns:
        Dictionary mapping class index to weight.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)

    weights = {}
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_classes * count)

    return weights


def create_dataset(image_paths: List[str], labels: List[int],
                   batch_size: int = config.BATCH_SIZE,
                   augment: bool = False,
                   shuffle: bool = False,
                   cache: bool = True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from image paths and labels.

    Args:
        image_paths: List of image file paths.
        labels: List of integer labels.
        batch_size: Batch size.
        augment: Whether to apply data augmentation.
        shuffle: Whether to shuffle the dataset.
        cache: Whether to cache the dataset in memory.

    Returns:
        Batched tf.data.Dataset yielding (image, label) pairs.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000),
                        seed=config.RANDOM_SEED)

    def _load_and_preprocess(path, label):
        image = decode_and_preprocess(path, augment=augment)
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def generate_dataset_report(
    dataset_dir: str = config.DPD_DATASET_DIR,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Generate a comprehensive dataset inspection report.

    Returns:
        Dictionary with image counts, class distribution,
        split sizes, duplicate info, and corrupt image checks.
    """
    paths, labels = discover_dpd_images(dataset_dir)
    dist = get_class_distribution(labels)

    # Check for corrupt images (sample)
    corrupt = find_corrupt_images(paths, sample_limit=500)

    # Check for duplicates (sample)
    duplicates = find_exact_duplicates(paths, sample_limit=2000)

    # Generate splits
    splits = stratified_split(paths, labels, seed=seed)

    report = {
        "dataset_name": "DPD — Drowsiness Prediction Dataset",
        "dataset_source": "https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset",
        "dataset_dir": dataset_dir,
        "total_images": len(paths),
        "class_distribution": dist,
        "class_balance": "Balanced" if len(set(dist.values())) == 1 else "Imbalanced",
        "image_extensions": dict(Counter(os.path.splitext(p)[1].lower() for p in paths)),
        "corrupt_images_checked": min(500, len(paths)),
        "corrupt_images_found": len(corrupt),
        "corrupt_image_paths": corrupt,
        "duplicate_pairs_checked": min(2000, len(paths)),
        "duplicate_pairs_found": len(duplicates),
        "splits": {
            split_name: {
                "count": len(split_paths),
                "distribution": get_class_distribution(split_labels),
            }
            for split_name, (split_paths, split_labels) in splits.items()
        },
    }

    return report
