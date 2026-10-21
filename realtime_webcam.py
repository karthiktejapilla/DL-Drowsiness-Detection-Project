"""Local continuous webcam monitor using the same inference path as Streamlit."""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.drowsiness_engine import DrowsinessEngine
from src.inference import InferenceEngine

try:
    import winsound

    def play_alert_sound():
        winsound.Beep(1200, 400)
except ImportError:
    def play_alert_sound():
        print("\a", end="", flush=True)


def run_webcam(model_path=config.EYE_STATE_MODEL_PATH, camera_index=config.CAMERA_INDEX,
               closure_threshold=config.EYE_CLOSURE_THRESHOLD_SEC, sound_alert=True):
    try:
        detector = InferenceEngine(model_path=model_path)
    except Exception as exc:
        print(f"Could not load model: {exc}")
        return

    temporal = DrowsinessEngine(closure_threshold_sec=closure_threshold)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Could not open camera {camera_index}. Check Windows camera permission and close other camera apps.")
        return

    title = "Drowsiness Monitor — Q to quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    previous_time = time.time()
    fps = 0.0
    try:
        print("Camera running. Press q to stop.")
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a camera frame; stopping.")
                break
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 1e-5)
            previous_time = now

            result = detector.process_frame(frame, allow_direct_crop_fallback=False)
            decision = temporal.update(result.overall_class, confidence=result.overall_confidence, current_time=now)
            output = result.annotated_image_bgr.copy()
            state = decision["state"]
            color = (0, 0, 220) if state == DrowsinessEngine.STATE_POSSIBLE_DROWSINESS else (25, 55, 65)
            cv2.rectangle(output, (0, 0), (output.shape[1], 78), color, -1)
            cv2.putText(output, f"{state} | {result.overall_state}", (16, 30), cv2.FONT_HERSHEY_DUPLEX, .7, (255,255,255), 2)
            cv2.putText(output, f"Closure {decision['closure_duration']:.1f}s / {closure_threshold:.1f}s   FPS {fps:.1f}",
                        (16, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (230,230,230), 1)
            if decision["alert_triggered"] and sound_alert:
                try:
                    play_alert_sound()
                except Exception:
                    pass
            cv2.imshow(title, output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if detector.face_mesh is not None:
            detector.face_mesh.close()
        print("Camera stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous local webcam drowsiness monitor")
    parser.add_argument("--model-path", default=config.EYE_STATE_MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=config.EYE_CLOSURE_THRESHOLD_SEC)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    run_webcam(args.model_path, args.camera_index, args.threshold, not args.no_sound)
