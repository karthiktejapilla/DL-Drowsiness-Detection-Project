"""
Data loading utilities for the MRL Eye Dataset.

Handles:
- Recursive image discovery
- Filename metadata parsing
- Subject-wise split tracking
- tf.data.Dataset creation with efficient batching and caching
- Class distribution reporting
"""
import os
import glob
import re
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import decode_and_preprocess


def parse_mrl_filename(filename: str) -> Optional[Dict]:
    """
    Parse MRL Eye Dataset filename to extract metadata.

    Filename format: s{subject}_{imageID}_{eyeState}_{gender}_{glasses}_{reflections}_{lighting}_{sensorID}.png

    Args:
        filename: The image filename (basename only).

    Returns:
        Dictionary with parsed metadata, or None if parsing fails.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')

    if len(parts) != 8:
        return None

    try:
        return {
            'subject': parts[0],           # e.g., 's0001'
            'image_id': parts[1],          # e.g., '01842'
            'eye_state': int(parts[2]),    # 0 or 1 (original encoding)
            'gender': int(parts[3]),       # 0=male, 1=female
            'glasses': int(parts[4]),      # 0=no, 1=yes
            'reflections': int(parts[5]),  # 0=none, 1=small, 2=big
            'lighting': int(parts[6]),     # 0=bad, 1=good
            'sensor_id': parts[7],         # 01, 02, 03
        }
    except (ValueError, IndexError):
        return None


def discover_images(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Recursively discover images in a directory organized as:
        data_dir/{class_name}/image_files

    For MRL: awake=0 (open), sleepy=1 (closed)
    Label is determined by folder name, NOT filename metadata.

    Args:
        data_dir: Path to data split directory (e.g., train/).

    Returns:
        Tuple of (image_paths, labels, subjects)
    """
    image_paths = []
    labels = []
    subjects = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: Directory not found: {class_dir}")
            continue

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                image_paths.append(img_path)
                labels.append(class_idx)

                # Extract subject from filename
                metadata = parse_mrl_filename(os.path.basename(img_path))
                if metadata:
                    subjects.append(metadata['subject'])
                else:
                    subjects.append('unknown')

    return image_paths, labels, subjects


def discover_subject_aware_splits(seed: int = config.RANDOM_SEED) -> Dict[str, Tuple[List[str], List[int], List[str]]]:
    """Create deterministic train/validation/test partitions with no subject overlap.

    All files from the existing MRL folders are inventoried without moving or
    editing them, then reassigned in memory by filename subject identifier.
    Unparseable filenames are kept as individual groups so they cannot silently
    merge into one synthetic subject.
    """
    grouped = {}
    for source_split in ("train", "val", "test"):
        paths, labels, subjects = discover_images(getattr(config, f"MRL_{source_split.upper()}_DIR"))
        for path, label, subject in zip(paths, labels, subjects):
            group = subject if subject != "unknown" else f"unknown:{os.path.abspath(path)}"
            grouped.setdefault(group, []).append((path, label, subject))

    subject_ids = sorted(grouped)
    if len(subject_ids) < 3:
        raise ValueError(f"Subject-aware splitting requires at least 3 identifiable groups; found {len(subject_ids)}.")
    train_end = max(1, int(len(subject_ids) * 0.70))
    val_end = max(train_end + 1, int(len(subject_ids) * 0.85))
    val_end = min(val_end, len(subject_ids) - 1)
    sizes = (train_end, val_end - train_end, len(subject_ids) - val_end)
    fractions = tuple(size / len(subject_ids) for size in sizes)
    group_counts = {
        subject: (sum(row[1] == 0 for row in rows), sum(row[1] == 1 for row in rows))
        for subject, rows in grouped.items()
    }
    class_totals = [sum(counts[label] for counts in group_counts.values()) for label in (0, 1)]

    # Seek a deterministic group assignment whose class proportions stay close
    # to the full dataset while preserving the requested subject counts.
    rng = random.Random(seed)
    best_score = float("inf")
    best_groups = None
    for _ in range(20_000):
        candidate = list(subject_ids)
        rng.shuffle(candidate)
        partitions = (candidate[:sizes[0]], candidate[sizes[0]:sizes[0] + sizes[1]], candidate[sizes[0] + sizes[1]:])
        score = 0.0
        for partition, fraction in zip(partitions, fractions):
            for label in (0, 1):
                actual = sum(group_counts[subject][label] for subject in partition)
                expected = fraction * class_totals[label]
                score += ((actual - expected) / max(1, class_totals[label])) ** 2
        if score < best_score:
            best_score, best_groups = score, partitions
    assigned = {name: set(groups) for name, groups in zip(("train", "val", "test"), best_groups)}
    result = {}
    for split, groups in assigned.items():
        rows = [
            (path, label, subject if subject != "unknown" else group)
            for group in sorted(groups)
            for path, label, subject in grouped[group]
        ]
        rows.sort(key=lambda row: row[0])
        result[split] = tuple([list(values) for values in zip(*rows)]) if rows else ([], [], [])
    return result


def get_class_distribution(labels: List[int]) -> Dict[str, int]:
    """
    Report class distribution.

    Args:
        labels: List of integer labels.

    Returns:
        Dictionary mapping class name to count.
    """
    counts = Counter(labels)
    return {config.CLASS_NAMES[k]: v for k, v in sorted(counts.items())}


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
    # Create dataset from paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000),
                        seed=config.RANDOM_SEED)

    # Map: load and preprocess images
    def _load_and_preprocess(path, label):
        image = decode_and_preprocess(path, augment=augment)
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        # Cache to disk to avoid re-reading - better for large datasets on CPU
        ds = ds.cache()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_mrl_split(split_dir: str, batch_size: int = config.BATCH_SIZE,
                   augment: bool = False, shuffle: bool = False,
                   cache: bool = True) -> Tuple[tf.data.Dataset, List[str], List[int], List[str]]:
    """
    Load a complete MRL dataset split.

    Args:
        split_dir: Path to the split directory (train/val/test).
        batch_size: Batch size.
        augment: Whether to apply augmentation.
        shuffle: Whether to shuffle.
        cache: Whether to cache.

    Returns:
        Tuple of (dataset, image_paths, labels, subjects)
    """
    print(f"Loading MRL split from: {split_dir}")
    paths, labels, subjects = discover_images(split_dir)

    if len(paths) == 0:
        raise ValueError(f"No images found in {split_dir}")

    dist = get_class_distribution(labels)
    print(f"  Found {len(paths)} images: {dist}")
    print(f"  Unique subjects: {len(set(subjects))}")

    ds = create_dataset(paths, labels, batch_size=batch_size,
                        augment=augment, shuffle=shuffle, cache=cache)

    return ds, paths, labels, subjects


def get_subject_split_info(train_subjects: List[str],
                           val_subjects: List[str],
                           test_subjects: List[str]) -> Dict:
    """
    Report subject overlap between splits (for validation).

    Args:
        train_subjects, val_subjects, test_subjects: Subject lists per split.

    Returns:
        Dictionary with subject overlap information.
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    return {
        'train_subjects': sorted(train_set),
        'val_subjects': sorted(val_set),
        'test_subjects': sorted(test_set),
        'train_val_overlap': sorted(train_set & val_set),
        'train_test_overlap': sorted(train_set & test_set),
        'val_test_overlap': sorted(val_set & test_set),
        'all_overlap': sorted(train_set & val_set & test_set),
    }
