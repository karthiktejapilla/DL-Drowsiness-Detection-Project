"""
Video processing and sequence dataset utilities for Human Drowsiness Detection.

Provides:
- Video frame extraction with uniform downsampling and temporal timestamping
- Extraction of face sequences from video files for drowsiness analysis
- Video generation from annotated frames for reporting and download
"""
import os
import cv2
from typing import List, Tuple, Optional, Generator, Dict, Any
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_frames_from_video(
    video_path: str,
    sample_rate: int = config.VIDEO_FRAME_SAMPLE_RATE,
    max_frames: int = config.MAX_VIDEO_FRAMES,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Extract frames from a video file with sampling rate.

    Args:
        video_path: Path to the input video file.
        sample_rate: Sample every Nth frame.
        max_frames: Maximum total frames to extract.

    Yields:
        Tuple of (frame_index, timestamp_sec, frame_bgr)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    extracted_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or extracted_count >= max_frames:
                break

            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                yield frame_idx, timestamp, frame
                extracted_count += 1

            frame_idx += 1
    finally:
        cap.release()


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Retrieve video metadata (dimensions, fps, total frames, duration).

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary of video properties.
    """
    if not os.path.isfile(video_path):
        return {"exists": False}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"exists": False, "readable": False}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "exists": True,
        "readable": True,
        "filename": os.path.basename(video_path),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration_s, 2),
    }


def write_annotated_video(
    frames_bgr: List[np.ndarray],
    output_path: str,
    fps: float = 15.0,
) -> bool:
    """
    Save a list of annotated BGR frames into an MP4 video.

    Args:
        frames_bgr: List of BGR images.
        output_path: Path to output video.
        fps: Playback frame rate.

    Returns:
        True if video was saved successfully.
    """
    if not frames_bgr:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = frames_bgr[0].shape[:2]

    # Try mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_bgr:
        out.write(frame)

    out.release()
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 0
