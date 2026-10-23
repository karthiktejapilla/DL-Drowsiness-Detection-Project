"""
Unified Inference Pipeline for Human Drowsiness Detection.

Provides:
- Shared inference engine across Image Upload, Video Upload, and Live Webcam
- Face-level fatigue classification (Active vs Fatigue)
- Robust face detection with Haar cascade ensemble + CLAHE
- Professional visual annotation rendering
- Execution latency measurement
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import load_trained_model
from src.preprocessing import (
    preprocess_for_inference,
    detect_faces_robust,
    crop_face_from_frame,
)


@dataclass
class FaceResult:
    """Individual face analysis result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_bgr: np.ndarray             # Cropped face region
    pred_class: int                   # 0 = ACTIVE, 1 = FATIGUE
    pred_label: str                   # 'Active' or 'Fatigue'
    confidence: float                 # Confidence in predicted class [0.0 - 1.0]
    raw_probability: float            # Sigmoid output [0.0 = Active, 1.0 = Fatigue]


@dataclass
class ImageAnalysisResult:
    """Complete result from processing an image or single video frame."""
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_result: Optional[FaceResult] = None
    overall_state: str = "NO FACE DETECTED"
    overall_class: Optional[int] = None    # 0 = ACTIVE, 1 = FATIGUE, None = Undetected
    overall_confidence: float = 0.0
    annotated_image_bgr: np.ndarray = field(default_factory=lambda: np.zeros((100, 100, 3), dtype=np.uint8))
    latency_ms: float = 0.0
    is_direct_face_image: bool = False
    detection_method: str = "none"         # 'haar_cascade', 'direct_face', 'none'
    message: str = ""


class InferenceEngine:
    """
    Unified Inference Engine for Face-Level Drowsiness Classification.

    Classifies face images as Active (alert) or Fatigue (drowsy).
    """

    def __init__(self, model_path: str = config.MODEL_PATH, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = load_trained_model(model_path)

    def predict_face(self, face_crop_bgr: np.ndarray) -> Tuple[int, float, float]:
        """
        Classify a single face crop.

        Args:
            face_crop_bgr: BGR face crop array.

        Returns:
            Tuple of (pred_class [0 or 1], confidence [0.0 - 1.0], raw_prob [0.0 - 1.0])
            pred_class: 0 = Active, 1 = Fatigue
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return 0, 0.0, 0.0

        inp = preprocess_for_inference(face_crop_bgr)
        raw_prob = float(self.model(inp, training=False).numpy()[0][0])
        pred_class = 1 if raw_prob >= config.CONFIDENCE_THRESHOLD else 0
        confidence = raw_prob if pred_class == 1 else (1.0 - raw_prob)
        return pred_class, confidence, raw_prob

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        allow_direct_face_fallback: bool = True,
        draw_annotations: bool = True,
        is_direct_face_hint: bool = False,
    ) -> ImageAnalysisResult:
        """
        Full inference pipeline for a single image or video frame.

        Pipeline:
        1. If the image appears to be a direct face crop (small dimensions or hint),
           classify it directly.
        2. Otherwise, detect faces using Haar cascades, crop the largest face,
           and classify it.
        3. If no face is detected and fallback is allowed, classify the entire image.

        Args:
            frame_bgr: Input BGR image.
            allow_direct_face_fallback: If True, classify the whole image when no face is detected.
            draw_annotations: Whether to draw bounding boxes and status overlays.
            is_direct_face_hint: If True, treat the entire image as a face crop.

        Returns:
            ImageAnalysisResult dataclass.
        """
        start_time = time.time()

        if frame_bgr is None or frame_bgr.size == 0:
            return ImageAnalysisResult(
                face_detected=False,
                annotated_image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                latency_ms=0.0,
                message="Input frame is empty or invalid.",
            )

        annotated = frame_bgr.copy() if draw_annotations else frame_bgr
        h, w = frame_bgr.shape[:2]
        face_result: Optional[FaceResult] = None
        face_detected = False
        face_bbox = None
        detection_method = "none"
        is_direct_face = False

        # ─── Mode 1: Direct Face Image (hint provided) ───
        if is_direct_face_hint:
            p_cls, conf, prob = self.predict_face(frame_bgr)
            detection_method = "direct_face"
            is_direct_face = True
            face_detected = True
            face_bbox = (0, 0, w, h)
            face_result = FaceResult(
                bbox=(0, 0, w, h),
                crop_bgr=frame_bgr,
                pred_class=p_cls,
                pred_label=config.CLASS_NAMES[p_cls],
                confidence=conf,
                raw_probability=prob,
            )
        elif w <= 200 and h <= 200 and allow_direct_face_fallback:
            # Image is already face-crop resolution (e.g. DPD 128x128)
            p_cls, conf, prob = self.predict_face(frame_bgr)
            detection_method = "direct_face"
            is_direct_face = True
            face_detected = True
            face_bbox = (0, 0, w, h)
            face_result = FaceResult(
                bbox=(0, 0, w, h),
                crop_bgr=frame_bgr,
                pred_class=p_cls,
                pred_label=config.CLASS_NAMES[p_cls],
                confidence=conf,
                raw_probability=prob,
            )

        # ─── Mode 2: Automatic Face Detection ───
        if face_result is None:
            face_crop, face_box, method = crop_face_from_frame(frame_bgr)
            if face_crop is not None and face_box is not None:
                face_detected = True
                detection_method = method
                p_cls, conf, prob = self.predict_face(face_crop)
                face_bbox = face_box
                face_result = FaceResult(
                    bbox=face_box,
                    crop_bgr=face_crop,
                    pred_class=p_cls,
                    pred_label=config.CLASS_NAMES[p_cls],
                    confidence=conf,
                    raw_probability=prob,
                )

        # ─── Mode 3: Fallback — classify entire image as face ───
        if face_result is None and allow_direct_face_fallback:
            p_cls, conf, prob = self.predict_face(frame_bgr)
            detection_method = "direct_face_fallback"
            is_direct_face = True
            face_detected = True
            face_bbox = (0, 0, w, h)
            face_result = FaceResult(
                bbox=(0, 0, w, h),
                crop_bgr=frame_bgr,
                pred_class=p_cls,
                pred_label=config.CLASS_NAMES[p_cls],
                confidence=conf,
                raw_probability=prob,
            )

        # ─── Resolve Overall State ───
        overall_state = "NO FACE DETECTED"
        overall_class = None
        overall_confidence = 0.0

        if face_result is not None:
            overall_class = face_result.pred_class
            overall_confidence = face_result.confidence
            if face_result.pred_class == 0:
                overall_state = "ACTIVE"
            else:
                overall_state = "FATIGUE"

        # ─── Draw Professional Visual Overlays ───
        if draw_annotations and face_result is not None:
            x1, y1, x2, y2 = face_result.bbox

            # Don't draw outer border for whole-image direct classification
            if not (x1 == 0 and y1 == 0 and x2 == w and y2 == h):
                # Green for Active, Red for Fatigue
                box_color = (60, 200, 40) if face_result.pred_class == 0 else (40, 50, 235)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                badge_text = f"{face_result.pred_label} ({face_result.confidence:.0%})"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y1 = max(0, y1 - 28)
                tag_y2 = max(28, y1)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 10, tag_y2), box_color, -1)
                cv2.putText(annotated, badge_text, (x1 + 5, tag_y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        latency = (time.time() - start_time) * 1000.0

        message = (
            "Analysis successful."
            if face_result is not None
            else "No face detected. Ensure adequate lighting and a visible face in the frame."
        )

        return ImageAnalysisResult(
            face_detected=face_detected,
            face_bbox=face_bbox,
            face_result=face_result,
            overall_state=overall_state,
            overall_class=overall_class,
            overall_confidence=round(overall_confidence, 3),
            annotated_image_bgr=annotated,
            latency_ms=round(latency, 1),
            is_direct_face_image=is_direct_face,
            detection_method=detection_method,
            message=message,
        )
