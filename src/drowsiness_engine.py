"""
Temporal Decision Engine for Drowsiness Detection.

Tracks continuous predicted facial fatigue over time, maintaining a rolling
history of predictions to distinguish momentary lapses or expressions from sustained
drowsiness and fatigue.
"""
import time
from collections import deque
from typing import Dict, Any, Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DrowsinessEngine:
    """
    Temporal Drowsiness Decision Engine.
    
    States:
        - ALERT: Subject is active, alert, and responsive.
        - FATIGUE DETECTED: Fatigue signs detected for brief duration (< threshold).
        - DROWSINESS ALERT: Sustained fatigue exceeding warning threshold (alarm triggers).
        - FACE NOT DETECTED: Face missing from frame.
    """

    STATE_ALERT = "ALERT"
    STATE_FATIGUE_DETECTED = "FATIGUE DETECTED"
    STATE_DROWSINESS_ALERT = "DROWSINESS ALERT"
    STATE_NOT_DETECTED = "NO FACE DETECTED"

    def __init__(
        self,
        fatigue_threshold_sec: float = config.FATIGUE_THRESHOLD_SEC,
        blink_max_duration_sec: float = config.BLINK_MAX_DURATION_SEC,
        smoothing_window: int = config.PREDICTION_SMOOTHING_WINDOW,
        cooldown_sec: float = config.WARNING_COOLDOWN_SEC,
        min_confidence: float = config.MIN_INFERENCE_CONFIDENCE,
    ):
        self.fatigue_threshold_sec = fatigue_threshold_sec
        self.blink_max_duration_sec = blink_max_duration_sec
        self.smoothing_window = smoothing_window
        self.cooldown_sec = cooldown_sec
        self.min_confidence = min_confidence

        # Rolling history of raw states: 0 = ACTIVE, 1 = FATIGUE
        self.history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)

        # Rolling window for fatigue ratio (percentage of fatigue frames over past 60 frames)
        self.fatigue_window = deque(maxlen=60)

        # Timing tracking
        self.fatigue_start_time: Optional[float] = None
        self.first_timestamp: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.current_fatigue_duration: float = 0.0

        # Warning trigger tracking
        self.last_warning_time: float = -999.0
        self.total_transient_events: int = 0
        self.total_drowsy_events: int = 0
        self.total_frames_processed: int = 0
        self.total_fatigue_frames: int = 0
        self.current_state: str = self.STATE_ALERT

    def reset(self):
        """Reset internal temporal state (e.g. when video changes or stream restarts)."""
        self.history.clear()
        self.confidence_history.clear()
        self.fatigue_window.clear()
        self.fatigue_start_time = None
        self.first_timestamp = None
        self.last_timestamp = None
        self.current_fatigue_duration = 0.0
        self.total_transient_events = 0
        self.total_drowsy_events = 0
        self.total_frames_processed = 0
        self.total_fatigue_frames = 0
        self.current_state = self.STATE_ALERT

    def update(
        self,
        pred_class: Optional[int],
        confidence: float = 1.0,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update the engine with a new frame's prediction.

        Args:
            pred_class: 0 for ACTIVE, 1 for FATIGUE, None if face undetected.
            confidence: Float confidence of the model prediction [0.0 - 1.0].
            current_time: Optional explicit timestamp (in seconds). If None, uses time.time().

        Returns:
            Dictionary with decision output:
                - state: Current categorized state string
                - fatigue_duration: Duration (sec) of current continuous fatigue
                - smoothed_class: Smoothed 0 (ACTIVE) or 1 (FATIGUE) or None
                - alert_triggered: Boolean if audible/visual alarm should fire
                - fatigue_ratio: Percentage (0-100%) of fatigue frames in rolling window
                - total_drowsy_events: Total count of sustained drowsy episodes
        """
        now = time.time() if current_time is None else current_time
        if self.first_timestamp is None:
            self.first_timestamp = now
        alert_triggered = False

        # Current fatigue ratio in rolling window
        curr_fatigue_pct = (
            round((sum(self.fatigue_window) / len(self.fatigue_window) * 100.0), 1)
            if self.fatigue_window
            else 0.0
        )

        # Handle face not detected
        if pred_class is None or float(confidence) < self.min_confidence:
            self.history.clear()
            self.confidence_history.clear()
            self.fatigue_start_time = None
            self.current_fatigue_duration = 0.0
            self.current_state = self.STATE_NOT_DETECTED
            self.last_timestamp = now
            return {
                "state": self.current_state,
                "fatigue_duration": 0.0,
                "closure_duration": 0.0,  # For backward-compatibility with UI
                "smoothed_state": None,
                "smoothed_class": None,
                "confidence": 0.0,
                "uncertain": pred_class is not None,
                "alert_triggered": False,
                "fatigue_ratio": curr_fatigue_pct,
                "perclos": curr_fatigue_pct,  # Alias for backward-compatibility
                "total_drowsy_events": self.total_drowsy_events,
            }

        # Valid detection: update counters
        self.total_frames_processed += 1
        self.fatigue_window.append(pred_class)
        if pred_class == 1:
            self.total_fatigue_frames += 1

        curr_fatigue_pct = round((sum(self.fatigue_window) / len(self.fatigue_window) * 100.0), 1)

        # Add to rolling history
        self.history.append(pred_class)
        self.confidence_history.append(confidence)

        # Smoothed state: majority vote over recent frames
        smoothed_state = 1 if (sum(self.history) / len(self.history)) >= 0.5 else 0
        avg_confidence = float(np.mean(self.confidence_history))

        # Check fatigue duration
        if smoothed_state == 1:
            # Subject is showing fatigue
            if self.fatigue_start_time is None:
                self.fatigue_start_time = now
                self.current_fatigue_duration = 0.0
            else:
                self.current_fatigue_duration = max(0.0, now - self.fatigue_start_time)

            if self.current_fatigue_duration >= self.fatigue_threshold_sec:
                self.current_state = self.STATE_DROWSINESS_ALERT
                # Check cooldown for warning alarm
                if (now - self.last_warning_time) >= self.cooldown_sec:
                    alert_triggered = True
                    self.last_warning_time = now
                    self.total_drowsy_events += 1
            else:
                self.current_state = self.STATE_FATIGUE_DETECTED
        else:
            # Subject is active/alert
            if self.fatigue_start_time is not None:
                past_duration = max(0.0, now - self.fatigue_start_time)
                if 0.05 <= past_duration <= self.blink_max_duration_sec:
                    self.total_transient_events += 1
                self.fatigue_start_time = None

            self.current_fatigue_duration = 0.0
            self.current_state = self.STATE_ALERT

        self.last_timestamp = now

        return {
            "state": self.current_state,
            "fatigue_duration": round(self.current_fatigue_duration, 2),
            "closure_duration": round(self.current_fatigue_duration, 2),  # Compatibility
            "smoothed_state": smoothed_state,
            "smoothed_class": smoothed_state,
            "confidence": round(avg_confidence, 3),
            "uncertain": False,
            "alert_triggered": alert_triggered,
            "fatigue_ratio": curr_fatigue_pct,
            "perclos": curr_fatigue_pct,  # Compatibility
            "total_drowsy_events": self.total_drowsy_events,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return overall session statistics."""
        session_duration = 0.0
        if self.first_timestamp is not None and self.last_timestamp is not None:
            session_duration = max(0.0, self.last_timestamp - self.first_timestamp)

        overall_fatigue_pct = round(
            (self.total_fatigue_frames / max(1, self.total_frames_processed)) * 100.0, 1
        )

        return {
            "total_frames": self.total_frames_processed,
            "total_fatigue_frames": self.total_fatigue_frames,
            "total_drowsy_events": self.total_drowsy_events,
            "overall_fatigue_pct": overall_fatigue_pct,
            "session_duration_s": round(session_duration, 2),
        }
