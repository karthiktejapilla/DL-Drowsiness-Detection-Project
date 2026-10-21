"""
UTA-RLDD Dataset loader and video/frame sequence utilities.

Handles:
- Parsing UTA-RLDD annotation files (train.txt, val.txt, test.txt)
- Subject extraction based on file prefixes (e.g., img_a, img_b)
- Face-level active (0) vs fatigue (1) dataset construction
- Configurable frame sampling for evaluation
"""
import os
import glob
from typing import Dict, List, Tuple, Optional
from collections import Counter
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def parse_uta_annotation_file(annotation_path: str, base_dir: str = config.UTA_RLDD_DIR) -> Tuple[List[str], List[int], List[str]]:
    """
    Parse an annotation file (e.g., train.txt, test.txt, val.txt).
    Each line format: relative/path/to/image.jpg label

    Args:
        annotation_path: Path to annotation text file.
        base_dir: Base directory of UTA-RLDD dataset.

    Returns:
        Tuple of (image_paths, labels, subject_ids)
    """
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    image_paths = []
    labels = []
    subjects = []

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                rel_path = parts[0]
                label = int(parts[1])
                full_path = os.path.join(base_dir, rel_path.replace("/", os.sep).replace("\\", os.sep))

                image_paths.append(full_path)
                labels.append(label)

                # Extract subject prefix (e.g. img_f_... -> subject 'f', image_... -> 'generic')
                fname = os.path.basename(rel_path)
                if fname.startswith("img_"):
                    subj = fname.split("_")[1]
                else:
                    subj = "common"
                subjects.append(subj)

    return image_paths, labels, subjects


def load_uta_rldd_split(
    split_name: str = "test",
    sample_limit: Optional[int] = None,
    base_dir: str = config.UTA_RLDD_DIR
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load UTA-RLDD images and labels for a specific split ('train', 'val', or 'test').

    Args:
        split_name: 'train', 'val', or 'test'
        sample_limit: Optional max number of samples to load (useful for fast evaluation)
        base_dir: Base folder path

    Returns:
        Tuple of (image_paths, labels, subjects)
    """
    txt_file = os.path.join(base_dir, f"{split_name}.txt")
    if os.path.exists(txt_file):
        paths, labels, subjects = parse_uta_annotation_file(txt_file, base_dir=base_dir)
    else:
        # Fallback to directory scan
        split_dir = os.path.join(base_dir, split_name)
        paths, labels, subjects = [], [], []
        for label_idx, folder_name in enumerate(["active", "fatigue"]):
            folder_path = os.path.join(split_dir, folder_name)
            if os.path.isdir(folder_path):
                for f in glob.glob(os.path.join(folder_path, "*.jpg")):
                    paths.append(f)
                    labels.append(label_idx)
                    fname = os.path.basename(f)
                    subj = fname.split("_")[1] if fname.startswith("img_") else "common"
                    subjects.append(subj)

    # Filter to only existing files
    valid_data = [(p, l, s) for p, l, s in zip(paths, labels, subjects) if os.path.isfile(p)]
    if not valid_data:
        return [], [], []

    paths, labels, subjects = zip(*valid_data)
    paths, labels, subjects = list(paths), list(labels), list(subjects)

    if sample_limit and sample_limit < len(paths):
        # Evenly sample to maintain class ratio
        paths = paths[:sample_limit]
        labels = labels[:sample_limit]
        subjects = subjects[:sample_limit]

    return paths, labels, subjects
