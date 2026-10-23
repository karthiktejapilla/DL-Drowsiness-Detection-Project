"""
Image preprocessing utilities for Human Drowsiness Detection.

Handles:
- Image loading, decoding, and MobileNetV3 scaling to [-1, 1]
- Training-time augmentation (brightness, contrast, flip, zoom)
- Inference preprocessing for face images (RGB, resize, normalize)
- Robust multi-cascade face detection with CLAHE equalization
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

    Images are loaded as RGB (3 channels).
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
    Preprocess a numpy image (BGR or RGB or Grayscale) for model inference.

    The DPD model is trained on RGB face images.
    Produces float32 array of shape (1, IMG_HEIGHT, IMG_WIDTH, 3) normalized to [-1, 1].
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid empty image passed to preprocess_for_inference")

    # Convert to RGB
    if image_array.ndim == 2:
        # Grayscale
        rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        rgb = cv2.cvtColor(image_array[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # Assume BGR from OpenCV
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
    else:
        rgb = image_array

    resized = cv2.resize(rgb, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)
    preprocessed = (img_float / 127.5) - 1.0
    return np.expand_dims(preprocessed, axis=0)


def detect_faces_robust(frame_bgr: np.ndarray, min_size: Tuple[int, int] = None) -> List[Tuple[int, int, int, int]]:
    """
    Robust multi-detector face localization.
    Uses an ensemble of:
      1. haarcascade_frontalface_alt2.xml (robust to frontal-face variation)
      2. haarcascade_frontalface_default.xml (standard frontal)
      3. haarcascade_frontalface_alt.xml (tree-structured classifier)
      4. haarcascade_profileface.xml (for turned driver head)
    Applies CLAHE histogram equalization when standard detection fails.

    Returns:
        List of bounding boxes sorted by area descending: [(x, y, w, h), ...]
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    if min_size is None:
        min_size = config.FACE_DETECTION_MIN_SIZE

    original_h, original_w = frame_bgr.shape[:2]
    detection_scale = min(1.0, 640.0 / max(original_h, original_w))
    if detection_scale < 1.0:
        detection_image = cv2.resize(
            frame_bgr,
            (max(1, int(round(original_w * detection_scale))), max(1, int(round(original_h * detection_scale)))),
            interpolation=cv2.INTER_AREA,
        )
        scaled_min_size = (
            max(24, int(round(min_size[0] * detection_scale))),
            max(24, int(round(min_size[1] * detection_scale))),
        )
    else:
        detection_image = frame_bgr
        scaled_min_size = min_size

    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY) if detection_image.ndim == 3 else detection_image

    def restore_original_scale(faces):
        if detection_scale == 1.0:
            return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return sorted(
            [tuple(int(round(value / detection_scale)) for value in face) for face in faces],
            key=lambda b: b[2] * b[3],
            reverse=True,
        )

    cascades = [
        get_cascade("haarcascade_frontalface_alt2.xml"),
        get_cascade("haarcascade_frontalface_default.xml"),
        get_cascade("haarcascade_frontalface_alt.xml"),
    ]
    cascades = [c for c in cascades if c is not None]

    # Pass 1: Standard grayscale search
    for fc in cascades:
        faces = fc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=scaled_min_size)
        if len(faces) > 0:
            return restore_original_scale(faces)

    # Pass 2: CLAHE for low-light, shadow, or washed-out images
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    eq_gray = clahe.apply(gray)
    for fc in cascades:
        faces = fc.detectMultiScale(eq_gray, scaleFactor=1.05, minNeighbors=2, minSize=scaled_min_size)
        if len(faces) > 0:
            return restore_original_scale(faces)

    # Pass 3: Profile face (head turned slightly away from camera)
    prof_cascade = get_cascade("haarcascade_profileface.xml")
    if prof_cascade is not None:
        p_faces = prof_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=scaled_min_size)
        if len(p_faces) > 0:
            return restore_original_scale(p_faces)
        # Check flipped for opposite profile
        flipped = cv2.flip(gray, 1)
        pf_faces = prof_cascade.detectMultiScale(flipped, scaleFactor=1.15, minNeighbors=3, minSize=scaled_min_size)
        if len(pf_faces) > 0:
            w_orig = gray.shape[1]
            unflipped = [(w_orig - (x + w), y, w, h) for (x, y, w, h) in pf_faces]
            return restore_original_scale(unflipped)

    return []


def crop_face_from_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str]:
    """
    Detect and crop the primary face from a frame.

    For the DPD fatigue model, we classify the whole face crop rather than
    individual eye regions.

    Returns:
        Tuple of (face_crop_bgr, face_bbox, detection_method)
        face_crop_bgr is None if no face was detected.
    """
    faces = detect_faces_robust(frame_bgr)

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        h_img, w_img = frame_bgr.shape[:2]

        # Add padding around face (15%)
        pad_x = int(fw * 0.15)
        pad_y = int(fh * 0.15)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w_img, fx + fw + pad_x)
        y2 = min(h_img, fy + fh + pad_y)

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size > 0:
            return face_crop, (x1, y1, x2, y2), "haar_cascade"

    return None, None, "none"
