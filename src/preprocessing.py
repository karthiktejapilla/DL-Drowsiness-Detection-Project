"""
Image preprocessing and computer vision utilities for Human Drowsiness Detection.

Handles:
- Image loading, decoding, corruption checking, and MobileNetV3 scaling
- Robust multi-cascade face detection with CLAHE equalization and orientation support
- Dual-eye localization with cascade ensemble, ocular symmetry, and anatomical fallback
- Direct eye-crop verification for MRL-style cropped inputs
- MediaPipe landmarks backward compatibility
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Global Haar Cascade Cache ───
_CASCADES: Dict[str, cv2.CascadeClassifier] = {}


def get_cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    """Load and cache an OpenCV Haar cascade from cv2.data.haarcascades."""
    global _CASCADES
    if name in _CASCADES:
        return _CASCADES[name]
    path = os.path.join(cv2.data.haarcascades, name)
    if os.path.isfile(path):
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            _CASCADES[name] = cascade
            return cascade
    return None


def decode_and_preprocess(image_path: str, augment: bool = False) -> tf.Tensor:
    """
    Load, decode, resize, and preprocess a single image for model training.
    Scales pixels to [-1, 1] as expected by MobileNetV3.
    """
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])

    if augment:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        if tf.random.uniform(()) > 0.5:
            zoom_size = tf.random.uniform((), minval=0.85, maxval=1.0)
            crop_h = tf.cast(tf.cast(config.IMG_HEIGHT, tf.float32) * zoom_size, tf.int32)
            crop_w = tf.cast(tf.cast(config.IMG_WIDTH, tf.float32) * zoom_size, tf.int32)
            image = tf.image.random_crop(image, [crop_h, crop_w, 3])
            image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])

    image = tf.clip_by_value(image, 0.0, 255.0)
    image = (image / 127.5) - 1.0
    return image


def preprocess_for_inference(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess an arbitrary numpy image (BGR or Grayscale) for model inference.
    MRL training images are grayscale decoded into 3 identical channels.
    Produces float32 array of shape (1, IMG_HEIGHT, IMG_WIDTH, 3) normalized to [-1, 1].
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid empty image passed to preprocess_for_inference")

    if image_array.ndim == 2:
        gray = image_array
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        gray = image_array[:, :, 0]
    else:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)
    preprocessed = (img_float / 127.5) - 1.0
    return np.expand_dims(preprocessed, axis=0)


def check_image_readable(path: str) -> bool:
    """Verify an image file exists and can be decoded."""
    try:
        raw = tf.io.read_file(path)
        tf.image.decode_image(raw, channels=3)
        return True
    except Exception:
        return False


def is_likely_eye_crop(image_array: np.ndarray) -> bool:
    """
    Heuristically checks if an input image is already a standalone eye crop.
    MRL eye images and close-up ocular crops are typically small (e.g. <=320px)
    and have an aspect ratio roughly between 0.6 and 2.6.
    """
    if image_array is None or image_array.size == 0:
        return False
    h, w = image_array.shape[:2]
    aspect_ratio = w / float(max(1, h))
    if 0.6 <= aspect_ratio <= 2.6 and (w <= 360 and h <= 300):
        return True
    return False


def detect_faces_robust(frame_bgr: np.ndarray, min_size: Tuple[int, int] = (40, 40)) -> List[Tuple[int, int, int, int]]:
    """
    Robust multi-detector face localization.
    Uses an ensemble of:
      1. haarcascade_frontalface_alt2.xml (robust to pose & closed eyes)
      2. haarcascade_frontalface_default.xml (standard frontal)
      3. haarcascade_frontalface_alt.xml (tree-structured classifier)
      4. haarcascade_profileface.xml (for turned driver head)
    Applies CLAHE histogram equalization when standard detection fails.

    Returns:
        List of bounding boxes sorted by area descending: [(x, y, w, h), ...]
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

    cascades = [
        get_cascade("haarcascade_frontalface_alt2.xml"),
        get_cascade("haarcascade_frontalface_default.xml"),
        get_cascade("haarcascade_frontalface_alt.xml"),
    ]
    cascades = [c for c in cascades if c is not None]

    # Pass 1: Standard grayscale search
    for fc in cascades:
        faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=min_size)
        if len(faces) > 0:
            return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)

    # Pass 2: CLAHE for low-light, shadow, or washed-out images
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    eq_gray = clahe.apply(gray)
    for fc in cascades:
        faces = fc.detectMultiScale(eq_gray, scaleFactor=1.1, minNeighbors=3, minSize=min_size)
        if len(faces) > 0:
            return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)

    # Pass 3: Profile face (head turned slightly away from camera)
    prof_cascade = get_cascade("haarcascade_profileface.xml")
    if prof_cascade is not None:
        p_faces = prof_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=min_size)
        if len(p_faces) > 0:
            return sorted(p_faces, key=lambda b: b[2] * b[3], reverse=True)
        # Check flipped for opposite profile
        flipped = cv2.flip(gray, 1)
        pf_faces = prof_cascade.detectMultiScale(flipped, scaleFactor=1.15, minNeighbors=3, minSize=min_size)
        if len(pf_faces) > 0:
            w_orig = gray.shape[1]
            unflipped = [(w_orig - (x + w), y, w, h) for (x, y, w, h) in pf_faces]
            return sorted(unflipped, key=lambda b: b[2] * b[3], reverse=True)

    return []


def extract_eyes_from_face(
    frame_bgr: np.ndarray,
    face_bbox: Tuple[int, int, int, int]
) -> Tuple[
    Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]],
    Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]],
    str
]:
    """
    Extract Left Eye and Right Eye regions from a detected face bounding box.
    Uses a 3-tier strategy:
      1. Cascade detection (eyeglass-specialized + standard ocular cascades)
      2. Ocular symmetry fallback if exactly one eye is detected
      3. Anatomical ocular extraction if cascades find 0 eyes (e.g. tightly closed eyes!)

    Args:
        frame_bgr: Input BGR image.
        face_bbox: (fx, fy, fw, fh) face coordinates.

    Returns:
        ((left_crop, (x1, y1, x2, y2)), (right_crop, (x1, y1, x2, y2)), method_name)
    """
    h_img, w_img = frame_bgr.shape[:2]
    fx, fy, fw, fh = face_bbox

    # Define upper face ocular search zone: y: 15% to 65% of face height
    roi_y1 = max(0, fy + int(0.12 * fh))
    roi_y2 = min(h_img, fy + int(0.65 * fh))
    roi_x1 = max(0, fx)
    roi_x2 = min(w_img, fx + fw)

    roi_gray = cv2.cvtColor(frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
    roi_color = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

    # Try eye cascades: check eyeglasses cascade first, then standard eye cascade
    eye_cascades = [
        get_cascade("haarcascade_eye_tree_eyeglasses.xml"),
        get_cascade("haarcascade_eye.xml"),
        get_cascade("haarcascade_lefteye_2splits.xml"),
        get_cascade("haarcascade_righteye_2splits.xml"),
    ]
    eye_cascades = [c for c in eye_cascades if c is not None]

    detected_candidate_eyes: List[Tuple[int, int, int, int]] = []
    min_eye_size = (max(14, int(fw * 0.12)), max(14, int(fh * 0.12)))
    max_eye_size = (int(fw * 0.45), int(fh * 0.45))

    for ec in eye_cascades:
        eyes = ec.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=3,
                                  minSize=min_eye_size, maxSize=max_eye_size)
        if len(eyes) >= 2:
            detected_candidate_eyes = sorted(eyes, key=lambda b: b[0])
            break
        elif len(eyes) == 1 and not detected_candidate_eyes:
            detected_candidate_eyes = list(eyes)

    left_result: Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]] = (None, None)
    right_result: Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]] = (None, None)
    method_used = "anatomical"

    # Separate candidates into Left Eye (x < 0.52 of face) and Right Eye (x >= 0.48 of face)
    left_cands = [b for b in detected_candidate_eyes if b[0] + b[2] / 2.0 < (roi_x2 - roi_x1) * 0.52]
    right_cands = [b for b in detected_candidate_eyes if b[0] + b[2] / 2.0 >= (roi_x2 - roi_x1) * 0.48]

    if left_cands and right_cands:
        method_used = "cascade_dual"
        lb = left_cands[0]
        rb = right_cands[0]

        # Add 25% padding around detected eye box
        def make_crop_and_box(b):
            ex, ey, ew, eh = b
            pad_x = int(ew * 0.25)
            pad_y = int(eh * 0.25)
            x1 = max(0, roi_x1 + ex - pad_x)
            y1 = max(0, roi_y1 + ey - pad_y)
            x2 = min(w_img, roi_x1 + ex + ew + pad_x)
            y2 = min(h_img, roi_y1 + ey + eh + pad_y)
            crop = frame_bgr[y1:y2, x1:x2]
            return crop, (x1, y1, x2, y2)

        left_result = make_crop_and_box(lb)
        right_result = make_crop_and_box(rb)

    elif left_cands and not right_cands:
        # Symmetrical reflection of left eye to produce right eye
        method_used = "cascade_symmetry"
        lb = left_cands[0]
        ex, ey, ew, eh = lb
        pad_x, pad_y = int(ew * 0.25), int(eh * 0.25)
        lx1 = max(0, roi_x1 + ex - pad_x)
        ly1 = max(0, roi_y1 + ey - pad_y)
        lx2 = min(w_img, roi_x1 + ex + ew + pad_x)
        ly2 = min(h_img, roi_y1 + ey + eh + pad_y)
        left_result = (frame_bgr[ly1:ly2, lx1:lx2], (lx1, ly1, lx2, ly2))

        # Right eye symmetric across face center
        face_center_x = fx + fw // 2
        offset_from_center = face_center_x - (roi_x1 + ex + ew // 2)
        rx_center = face_center_x + offset_from_center
        rx1 = max(0, rx_center - (ew + 2 * pad_x) // 2)
        rx2 = min(w_img, rx1 + (ew + 2 * pad_x))
        right_result = (frame_bgr[ly1:ly2, rx1:rx2], (rx1, ly1, rx2, ly2))

    elif right_cands and not left_cands:
        method_used = "cascade_symmetry"
        rb = right_cands[0]
        ex, ey, ew, eh = rb
        pad_x, pad_y = int(ew * 0.25), int(eh * 0.25)
        rx1 = max(0, roi_x1 + ex - pad_x)
        ry1 = max(0, roi_y1 + ey - pad_y)
        rx2 = min(w_img, roi_x1 + ex + ew + pad_x)
        ry2 = min(h_img, roi_y1 + ey + eh + pad_y)
        right_result = (frame_bgr[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2))

        face_center_x = fx + fw // 2
        offset_from_center = (roi_x1 + ex + ew // 2) - face_center_x
        lx_center = face_center_x - offset_from_center
        lx1 = max(0, lx_center - (ew + 2 * pad_x) // 2)
        lx2 = min(w_img, lx1 + (ew + 2 * pad_x))
        left_result = (frame_bgr[ry1:ry2, lx1:lx2], (lx1, ry1, lx2, ry2))

    else:
        # Anatomical ocular estimation (Essential when eyes are closed or in shadows!)
        method_used = "anatomical_proportions"
        lx1 = max(0, fx + int(0.16 * fw))
        lx2 = min(w_img, fx + int(0.48 * fw))
        ly1 = max(0, fy + int(0.20 * fh))
        ly2 = min(h_img, fy + int(0.52 * fh))

        rx1 = max(0, fx + int(0.52 * fw))
        rx2 = min(w_img, fx + int(0.84 * fw))
        ry1 = max(0, fy + int(0.20 * fh))
        ry2 = min(h_img, fy + int(0.52 * fh))

        if lx2 > lx1 and ly2 > ly1:
            left_result = (frame_bgr[ly1:ly2, lx1:lx2], (lx1, ly1, lx2, ly2))
        if rx2 > rx1 and ry2 > ry1:
            right_result = (frame_bgr[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2))

    return left_result, right_result, method_used


def extract_eyes_haar(frame: np.ndarray, face_cascade, eye_cascade) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Backward-compatible eye extractor using OpenCV cascades.
    Returns list of tuples: [(eye_crop, (x1, y1, x2, y2)), ...]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))
    crops_and_coords = []

    for (fx, fy, fw, fh) in faces:
        (cl, bl), (cr, br), _ = extract_eyes_from_face(frame, (fx, fy, fw, fh))
        if cl is not None and cl.size > 0 and bl is not None:
            crops_and_coords.append((cl, bl))
        if cr is not None and cr.size > 0 and br is not None:
            crops_and_coords.append((cr, br))
    return crops_and_coords


def get_eye_region_and_coords(
    frame: np.ndarray, landmarks, eye_indices: list, padding: float = 0.3
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Extract eye region and coordinates using MediaPipe face mesh landmarks.
    Preserved for backward compatibility.
    """
    h, w = frame.shape[:2]
    x_coords = [landmarks[i].x * w for i in eye_indices if i < len(landmarks)]
    y_coords = [landmarks[i].y * h for i in eye_indices if i < len(landmarks)]

    if not x_coords or not y_coords:
        return None, None

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None, None

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None, None

    return eye_crop, (x1, y1, x2, y2)


def get_eye_region_from_landmarks(
    frame: np.ndarray, landmarks, eye_indices: list, padding: float = 0.3
) -> Optional[np.ndarray]:
    """Preserved for backward compatibility."""
    crop, _ = get_eye_region_and_coords(frame, landmarks, eye_indices, padding=padding)
    return crop
