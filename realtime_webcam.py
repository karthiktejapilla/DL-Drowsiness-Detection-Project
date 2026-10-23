"""Local continuous webcam monitor using the unified DPD face-level inference pipeline."""
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


def run_webcam(
    model_path=config.MODEL_PATH,
    camera_index=config.CAMERA_INDEX,
    fatigue_threshold=config.FATIGUE_THRESHOLD_SEC,
    sound_alert=True,
    capture_width=config.WEBCAM_CAPTURE_WIDTH,
    capture_height=config.WEBCAM_CAPTURE_HEIGHT,
):
    try:
        detector = InferenceEngine(model_path=model_path)
    except Exception as exc:
        print(f"Could not load model: {exc}")
        return

    temporal = DrowsinessEngine(fatigue_threshold_sec=fatigue_threshold)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Could not open camera {camera_index}. Check Windows camera permission and close other camera apps.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

    title = "DPD Drowsiness Monitor — Press 'q' to quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    previous_time = time.time()
    fps = 0.0

    try:
        print("Camera running. Press 'q' to stop.")
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a camera frame; stopping.")
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 1e-5)
            previous_time = now

            inference_start = time.perf_counter()
            result = detector.process_frame(frame, allow_direct_face_fallback=False)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            decision = temporal.update(result.overall_class, confidence=result.overall_confidence, current_time=now)
            output = result.annotated_image_bgr.copy()

            state = decision["state"]
            if state == DrowsinessEngine.STATE_DROWSINESS_ALERT:
                header_color = (0, 0, 220)  # Red alert
            elif state == DrowsinessEngine.STATE_FATIGUE_DETECTED:
                header_color = (0, 140, 255)  # Orange warning
            else:
                header_color = (30, 60, 40)   # Green/dark alert

            cv2.rectangle(output, (0, 0), (output.shape[1], 75), header_color, -1)
            cv2.putText(
                output,
                f"{state} | {result.overall_state}",
                (16, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output,
                f"Fatigue: {decision['fatigue_duration']:.1f}s / {fatigue_threshold:.1f}s | Ratio: {decision['fatigue_ratio']}% | FPS: {fps:.1f} | Inference: {inference_ms:.1f} ms",
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )

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
        print("Camera stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous local webcam drowsiness monitor (DPD model)")
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=config.FATIGUE_THRESHOLD_SEC)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    run_webcam(args.model_path, args.camera_index, args.threshold, not args.no_sound)
