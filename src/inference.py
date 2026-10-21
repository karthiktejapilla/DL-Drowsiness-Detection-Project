"""
Unified Inference Pipeline for Human Drowsiness Detection.

Provides:
- Shared inference engine across Image Upload, Video Upload, and Live Webcam
- Robust multi-cascade face detection with CLAHE enhancement
- Multi-tier dual-eye localization (cascade ensemble, bilateral symmetry, anatomical fallback)
- Direct eye-crop fallback for standalone eye images (MRL dataset images)
- Interactive manual ROI support
- Bilateral eye state resolution (Left Eye, Right Eye, Combined State)
- Professional visual annotation rendering with high-visibility overlays
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
    extract_eyes_from_face,
    is_likely_eye_crop,
    get_eye_region_and_coords,
    extract_eyes_haar,
)


@dataclass
class EyeResult:
    """Individual eye analysis result."""
    name: str                       # 'Left Eye', 'Right Eye', 'Manual Eye ROI', 'Direct Crop'
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    crop_bgr: np.ndarray            # Cropped eye region
    pred_class: int                 # 0 = OPEN / AWAKE, 1 = CLOSED / SLEEPY
    pred_label: str                 # 'OPEN' or 'CLOSED'
    confidence: float               # Confidence in the predicted class [0.0 - 1.0]
    raw_probability: float          # Sigmoid output [0.0 = Open, 1.0 = Closed]


@dataclass
class ImageAnalysisResult:
    """Complete result from processing an image or single video frame."""
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    eyes_detected_count: int = 0
    left_eye: Optional[EyeResult] = None
    right_eye: Optional[EyeResult] = None
    extra_eyes: List[EyeResult] = field(default_factory=list)
    overall_state: str = "NO FACE/EYES DETECTED"
    overall_class: Optional[int] = None # 0 = OPEN, 1 = CLOSED, None = Undetected
    overall_confidence: float = 0.0
    annotated_image_bgr: np.ndarray = field(default_factory=lambda: np.zeros((100, 100, 3), dtype=np.uint8))
    latency_ms: float = 0.0
    is_direct_eye_crop: bool = False
    detection_method: str = "none"      # 'cascade_dual', 'cascade_symmetry', 'anatomical_proportions', 'direct_crop', 'manual_roi'
    message: str = ""


class InferenceEngine:
    """
    Unified Inference Engine for Eye-State Classification and Ocular Tracking.
    """

    def __init__(self, model_path: str = config.EYE_STATE_MODEL_PATH, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = load_trained_model(model_path)

        # Retain backward-compatible references
        self.face_mesh = None
        self.mp_available = False
        self.haar_face_cascade = None
        self.haar_eye_cascade = None
        self._init_legacy_cascades()

    def _init_legacy_cascades(self):
        try:
            cascade_dir = cv2.data.haarcascades
            face_xml = os.path.join(cascade_dir, 'haarcascade_frontalface_alt2.xml')
            if not os.path.isfile(face_xml):
                face_xml = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
            eye_xml = os.path.join(cascade_dir, 'haarcascade_eye.xml')
            if os.path.isfile(face_xml) and os.path.isfile(eye_xml):
                self.haar_face_cascade = cv2.CascadeClassifier(face_xml)
                self.haar_eye_cascade = cv2.CascadeClassifier(eye_xml)
        except Exception:
            pass

    def predict_crop(self, eye_crop_bgr: np.ndarray) -> Tuple[int, float, float]:
        """
        Classify a single cropped eye region.

        Args:
            eye_crop_bgr: BGR or Grayscale crop array.

        Returns:
            Tuple of (pred_class [0 or 1], confidence [0.0 - 1.0], raw_prob [0.0 - 1.0])
        """
        if eye_crop_bgr is None or eye_crop_bgr.size == 0:
            return 0, 0.0, 0.0

        inp = preprocess_for_inference(eye_crop_bgr)
        raw_prob = float(self.model(inp, training=False).numpy()[0][0])
        pred_class = 1 if raw_prob >= config.CONFIDENCE_THRESHOLD else 0
        confidence = raw_prob if pred_class == 1 else (1.0 - raw_prob)
        return pred_class, confidence, raw_prob

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        allow_direct_crop_fallback: bool = True,
        manual_roi: Optional[Tuple[int, int, int, int]] = None,
        draw_annotations: bool = True,
        is_direct_crop_hint: bool = False,
    ) -> ImageAnalysisResult:
        """
        Full inference pipeline for a single image or video frame.

        Args:
            frame_bgr: Input BGR image.
            allow_direct_crop_fallback: If True, small/macro eye images are classified directly.
            manual_roi: Optional (x1, y1, x2, y2) bounding coordinates for manual eye analysis.
            draw_annotations: Whether to draw bounding boxes and status overlays on returned frame.
            is_direct_crop_hint: If user explicitly designates the image as an eye crop.

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

        left_eye_res: Optional[EyeResult] = None
        right_eye_res: Optional[EyeResult] = None
        extra_eyes: List[EyeResult] = []
        face_detected = False
        primary_face_box: Optional[Tuple[int, int, int, int]] = None
        detection_method = "none"
        is_direct_crop = False

        # ─── Mode 1: Manual Eye ROI Selection ───
        if manual_roi is not None:
            rx1, ry1, rx2, ry2 = manual_roi
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            if rx2 > rx1 and ry2 > ry1:
                crop = frame_bgr[ry1:ry2, rx1:rx2]
                if crop.size > 0:
                    p_cls, conf, prob = self.predict_crop(crop)
                    detection_method = "manual_roi"
                    is_direct_crop = True
                    left_eye_res = EyeResult(
                        name="Manual Eye ROI",
                        bbox=(rx1, ry1, rx2, ry2),
                        crop_bgr=crop,
                        pred_class=p_cls,
                        pred_label="CLOSED" if p_cls == 1 else "OPEN",
                        confidence=conf,
                        raw_probability=prob,
                    )

        # ─── Mode 2: Explicit or Detected Direct Eye Crop ───
        if left_eye_res is None and (is_direct_crop_hint or (allow_direct_crop_fallback and is_likely_eye_crop(frame_bgr))):
            # Check if image has faces; if not, treat as eye crop
            faces = detect_faces_robust(frame_bgr)
            if len(faces) == 0:
                p_cls, conf, prob = self.predict_crop(frame_bgr)
                detection_method = "direct_crop"
                is_direct_crop = True
                left_eye_res = EyeResult(
                    name="Direct Eye Image",
                    bbox=(0, 0, w, h),
                    crop_bgr=frame_bgr,
                    pred_class=p_cls,
                    pred_label="CLOSED" if p_cls == 1 else "OPEN",
                    confidence=conf,
                    raw_probability=prob,
                )

        # ─── Mode 3: Automatic Face and Dual-Eye Detection ───
        if left_eye_res is None:
            faces = detect_faces_robust(frame_bgr)
            if len(faces) > 0:
                face_detected = True
                primary_face_box = faces[0]
                fx, fy, fw, fh = primary_face_box

                # Extract left and right eyes using cascade + symmetry + anatomical fallback
                (crop_l, box_l), (crop_r, box_r), method = extract_eyes_from_face(frame_bgr, primary_face_box)
                detection_method = method

                if crop_l is not None and crop_l.size > 0 and box_l is not None:
                    p_cls_l, conf_l, prob_l = self.predict_crop(crop_l)
                    left_eye_res = EyeResult(
                        name="Left Eye",
                        bbox=box_l,
                        crop_bgr=crop_l,
                        pred_class=p_cls_l,
                        pred_label="CLOSED" if p_cls_l == 1 else "OPEN",
                        confidence=conf_l,
                        raw_probability=prob_l,
                    )

                if crop_r is not None and crop_r.size > 0 and box_r is not None:
                    p_cls_r, conf_r, prob_r = self.predict_crop(crop_r)
                    right_eye_res = EyeResult(
                        name="Right Eye",
                        bbox=box_r,
                        crop_bgr=crop_r,
                        pred_class=p_cls_r,
                        pred_label="CLOSED" if p_cls_r == 1 else "OPEN",
                        confidence=conf_r,
                        raw_probability=prob_r,
                    )

        # ─── Mode 4: Standalone Crop Fallback if explicitly allowed ───
        if left_eye_res is None and right_eye_res is None and allow_direct_crop_fallback:
            p_cls, conf, prob = self.predict_crop(frame_bgr)
            detection_method = "direct_crop_fallback"
            is_direct_crop = True
            left_eye_res = EyeResult(
                name="Direct Eye Crop",
                bbox=(0, 0, w, h),
                crop_bgr=frame_bgr,
                pred_class=p_cls,
                pred_label="CLOSED" if p_cls == 1 else "OPEN",
                confidence=conf,
                raw_probability=prob,
            )

        # ─── Count Detected Eyes ───
        detected_eyes = [e for e in [left_eye_res, right_eye_res] + extra_eyes if e is not None]
        eyes_count = len(detected_eyes)

        # ─── Resolve Overall State & Class ───
        overall_state = "NO FACE/EYES DETECTED"
        overall_class: Optional[int] = None
        overall_confidence = 0.0

        if is_direct_crop and left_eye_res is not None:
            overall_class = left_eye_res.pred_class
            overall_confidence = left_eye_res.confidence
            overall_state = f"DIRECT CROP: {left_eye_res.pred_label}"
        elif eyes_count >= 2 and left_eye_res and right_eye_res:
            overall_confidence = (left_eye_res.confidence + right_eye_res.confidence) / 2.0
            if left_eye_res.pred_class == 1 and right_eye_res.pred_class == 1:
                overall_state = "BOTH EYES CLOSED"
                overall_class = 1
            elif left_eye_res.pred_class == 0 and right_eye_res.pred_class == 0:
                overall_state = "BOTH EYES OPEN"
                overall_class = 0
            else:
                overall_state = "ASYMMETRIC (ONE CLOSED)"
                overall_class = 1  # Conservative caution for driver safety
        elif eyes_count == 1:
            single = detected_eyes[0]
            overall_class = single.pred_class
            overall_confidence = single.confidence
            overall_state = f"SINGLE EYE {single.pred_label}"
        else:
            overall_state = "NO FACE/EYES DETECTED"
            overall_class = None
            overall_confidence = 0.0

        # ─── Draw Professional Visual Overlays ───
        if draw_annotations:
            # 1. Draw Face Box if detected
            if primary_face_box is not None:
                fx, fy, fw, fh = primary_face_box
                # Subtle slate/cyan face bounding box
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (200, 160, 40), 1)
                tag_face = f"Face: Driver ({fw}x{fh})"
                cv2.putText(annotated, tag_face, (fx + 4, max(16, fy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 40), 1)

            # 2. Draw Eye Bounding Boxes & Badges
            for eye in detected_eyes:
                x1, y1, x2, y2 = eye.bbox
                if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
                    continue  # Don't draw outer border for whole direct crop
                # Emerald Green for OPEN, Crimson Red for CLOSED
                box_color = (40, 50, 235) if eye.pred_class == 1 else (60, 200, 40)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                badge_text = f"{eye.name}: {eye.pred_label} ({eye.confidence:.0%})"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, y1 - 20)
                tag_y2 = max(20, y1)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), box_color, -1)
                cv2.putText(annotated, badge_text, (x1 + 4, tag_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        latency = (time.time() - start_time) * 1000.0

        message = (
            "Analysis successful."
            if eyes_count > 0
            else "No face or eye landmarks could be reliably identified. Ensure adequate lighting or use manual ROI crop."
        )

        return ImageAnalysisResult(
            face_detected=face_detected or is_direct_crop,
            face_bbox=primary_face_box,
            eyes_detected_count=eyes_count,
            left_eye=left_eye_res,
            right_eye=right_eye_res,
            extra_eyes=extra_eyes,
            overall_state=overall_state,
            overall_class=overall_class,
            overall_confidence=round(overall_confidence, 3),
            annotated_image_bgr=annotated,
            latency_ms=round(latency, 1),
            is_direct_eye_crop=is_direct_crop,
            detection_method=detection_method,
            message=message,
        )
