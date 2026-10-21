"""
Temporal Decision Engine for Drowsiness Detection.

Tracks continuous predicted eye closure using elapsed time, maintaining a rolling
history of eye-state predictions to distinguish natural brief blinks from sustained
drowsy eye closure.
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
        - ALERT: Eyes open or normal blink pattern.
        - EYES CLOSED: Eyes closed for brief duration (< threshold).
        - POSSIBLE DROWSINESS: Sustained eye closure exceeding warning threshold.
        - FACE/EYES NOT DETECTED: Face or eyes missing from frame.
    """

    STATE_ALERT = "ALERT"
    STATE_EYES_CLOSED = "EYES CLOSED"
    STATE_POSSIBLE_DROWSINESS = "POSSIBLE DROWSINESS"
    STATE_NOT_DETECTED = "FACE/EYES NOT DETECTED"

    def __init__(
        self,
        closure_threshold_sec: float = config.EYE_CLOSURE_THRESHOLD_SEC,
        blink_max_duration_sec: float = config.BLINK_MAX_DURATION_SEC,
        smoothing_window: int = config.PREDICTION_SMOOTHING_WINDOW,
        cooldown_sec: float = config.WARNING_COOLDOWN_SEC,
    ):
        self.closure_threshold_sec = closure_threshold_sec
        self.blink_max_duration_sec = blink_max_duration_sec
        self.smoothing_window = smoothing_window
        self.cooldown_sec = cooldown_sec

        # Rolling history of raw eye states: 0 = OPEN, 1 = CLOSED
        self.history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)

        # PERCLOS tracking window (percentage of eye closure over past N frames)
        self.perclos_window = deque(maxlen=60)

        # Timing tracking
        self.closure_start_time: Optional[float] = None
        self.first_timestamp: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.current_closure_duration: float = 0.0

        # Warning trigger tracking
        self.last_warning_time: float = -999.0
        self.total_blinks: int = 0
        self.total_drowsy_events: int = 0
        self.total_frames_processed: int = 0
        self.total_closed_frames: int = 0
        self.current_state: str = self.STATE_ALERT

    def reset(self):
        """Reset internal temporal state (e.g. when video changes or face lost)."""
        self.history.clear()
        self.confidence_history.clear()
        self.perclos_window.clear()
        self.closure_start_time = None
        self.first_timestamp = None
        self.last_timestamp = None
        self.current_closure_duration = 0.0
        self.total_blinks = 0
        self.total_drowsy_events = 0
        self.total_frames_processed = 0
        self.total_closed_frames = 0
        self.current_state = self.STATE_ALERT

    def update(
        self,
        eye_state_pred: Optional[int],
        confidence: float = 1.0,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update the engine with a new frame's eye-state prediction.

        Args:
            eye_state_pred: 0 for OPEN, 1 for CLOSED, None if face/eyes undetected.
            confidence: Float confidence of the eye state model prediction.
            current_time: Optional explicit timestamp (in seconds). If None, uses time.time().

        Returns:
            Dictionary with decision output:
                - state: Current categorized state string
                - closure_duration: Duration (sec) of current continuous eye closure
                - smoothed_eye_state: Smoothed 0 (OPEN) or 1 (CLOSED) or None
                - alert_triggered: Boolean if audible/visual alarm should fire
                - perclos: Percentage (0-100%) of frames closed in rolling window
                - total_blinks: Total count of detected blinks
                - total_drowsy_events: Total count of sustained drowsy episodes
        """
        now = time.time() if current_time is None else current_time
        if self.first_timestamp is None:
            self.first_timestamp = now
        alert_triggered = False

        # Current PERCLOS value
        curr_perclos = round((sum(self.perclos_window) / len(self.perclos_window) * 100.0), 1) if self.perclos_window else 0.0

        # Handle face or eyes not detected
        if eye_state_pred is None:
            self.history.clear()
            self.confidence_history.clear()
            self.closure_start_time = None
            self.current_closure_duration = 0.0
            self.current_state = self.STATE_NOT_DETECTED
            self.last_timestamp = now
            return {
                "state": self.current_state,
                "closure_duration": 0.0,
                "smoothed_eye_state": None,
                "confidence": 0.0,
                "alert_triggered": False,
                "perclos": curr_perclos,
                "total_blinks": self.total_blinks,
                "total_drowsy_events": self.total_drowsy_events,
            }

        # Valid detection: update counters
        self.total_frames_processed += 1
        self.perclos_window.append(eye_state_pred)
        if eye_state_pred == 1:
            self.total_closed_frames += 1

        curr_perclos = round((sum(self.perclos_window) / len(self.perclos_window) * 100.0), 1)

        # Add to rolling history
        self.history.append(eye_state_pred)
        self.confidence_history.append(confidence)

        # Smoothed state: majority vote over recent frames
        smoothed_state = 1 if (sum(self.history) / len(self.history)) >= 0.5 else 0
        avg_confidence = float(np.mean(self.confidence_history))

        # Check eye closure duration
        if smoothed_state == 1:
            # Eyes are closed
            if self.closure_start_time is None:
                self.closure_start_time = now
                self.current_closure_duration = 0.0
            else:
                self.current_closure_duration = max(0.0, now - self.closure_start_time)

            if self.current_closure_duration >= self.closure_threshold_sec:
                self.current_state = self.STATE_POSSIBLE_DROWSINESS
                # Check cooldown for warning alarm
                if (now - self.last_warning_time) >= self.cooldown_sec:
                    alert_triggered = True
                    self.last_warning_time = now
                    self.total_drowsy_events += 1
            else:
                self.current_state = self.STATE_EYES_CLOSED
        else:
            # Eyes are open
            if self.closure_start_time is not None:
                # Eyes just opened! Check if the past closure was a normal blink
                past_duration = max(0.0, now - self.closure_start_time)
                if 0.05 <= past_duration <= self.blink_max_duration_sec:
                    self.total_blinks += 1
                self.closure_start_time = None

            self.current_closure_duration = 0.0
            self.current_state = self.STATE_ALERT

        self.last_timestamp = now

        return {
            "state": self.current_state,
            "closure_duration": round(self.current_closure_duration, 2),
            "smoothed_eye_state": smoothed_state,
            "confidence": round(avg_confidence, 3),
            "alert_triggered": alert_triggered,
            "perclos": curr_perclos,
            "total_blinks": self.total_blinks,
            "total_drowsy_events": self.total_drowsy_events,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return overall session statistics."""
        session_duration = 0.0
        if self.first_timestamp is not None and self.last_timestamp is not None:
            session_duration = max(0.0, self.last_timestamp - self.first_timestamp)

        blinks_per_min = round((self.total_blinks / (session_duration / 60.0)), 1) if session_duration > 10 else 0.0
        overall_perclos = round((self.total_closed_frames / max(1, self.total_frames_processed)) * 100.0, 1)

        return {
            "total_frames": self.total_frames_processed,
            "total_closed_frames": self.total_closed_frames,
            "total_blinks": self.total_blinks,
            "blinks_per_min": blinks_per_min,
            "total_drowsy_events": self.total_drowsy_events,
            "overall_perclos": overall_perclos,
            "session_duration_s": round(session_duration, 2),
        }

