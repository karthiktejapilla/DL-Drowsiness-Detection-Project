"""
Automated unit and integration tests for Drowsiness Detection System.

Tests cover:
1. Dataset filename parsing (MRL metadata)
2. Label mapping (0=open/awake, 1=closed/sleepy)
3. Image preprocessing shapes and value ranges
4. Model input/output shapes
5. Subject-wise split integrity (no data leakage)
6. Temporal closure logic (blinks vs sustained drowsiness)
7. Missing face and missing eye behavior
8. Invalid dataset paths and invalid video file handling
"""
import os
import sys
import unittest
import numpy as np
import tempfile
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import parse_mrl_filename, get_class_distribution, get_subject_split_info, discover_subject_aware_splits
from src.drowsiness_engine import DrowsinessEngine
from src.preprocessing import preprocess_for_inference
from src.model import _build_fallback_model


class TestDatasetParsing(unittest.TestCase):
    def test_mrl_filename_parsing(self):
        filename = "s0001_01842_0_0_1_0_0_01.png"
        meta = parse_mrl_filename(filename)
        self.assertIsNotNone(meta)
        self.assertEqual(meta['subject'], "s0001")
        self.assertEqual(meta['image_id'], "01842")
        self.assertEqual(meta['eye_state'], 0)
        self.assertEqual(meta['gender'], 0)
        self.assertEqual(meta['glasses'], 1)
        self.assertEqual(meta['reflections'], 0)
        self.assertEqual(meta['lighting'], 0)
        self.assertEqual(meta['sensor_id'], "01")

    def test_invalid_filename_parsing(self):
        self.assertIsNone(parse_mrl_filename("corrupted_name.png"))
        self.assertIsNone(parse_mrl_filename("abc.jpg"))

    def test_label_mapping_consistency(self):
        labels = [0, 0, 1, 1, 1]
        dist = get_class_distribution(labels)
        self.assertEqual(dist["awake"], 2)
        self.assertEqual(dist["sleepy"], 3)


class TestSubjectSplitIntegrity(unittest.TestCase):
    def test_no_leakage_across_splits(self):
        train_sub = ["s0001", "s0002", "s0003"]
        val_sub = ["s0004"]
        test_sub = ["s0005"]
        info = get_subject_split_info(train_sub, val_sub, test_sub)
        self.assertEqual(len(info["train_val_overlap"]), 0)
        self.assertEqual(len(info["train_test_overlap"]), 0)
        self.assertEqual(len(info["val_test_overlap"]), 0)

    def test_local_mrl_subject_aware_split_has_no_leakage(self):
        splits = discover_subject_aware_splits(seed=config.RANDOM_SEED)
        info = get_subject_split_info(splits["train"][2], splits["val"][2], splits["test"][2])
        self.assertEqual(len(info["train_val_overlap"]), 0)
        self.assertEqual(len(info["train_test_overlap"]), 0)
        self.assertEqual(len(info["val_test_overlap"]), 0)
        self.assertEqual(sum(len(splits[name][0]) for name in ("train", "val", "test")), 84898)


class TestImagePreprocessing(unittest.TestCase):
    def test_preprocess_for_inference_shape(self):
        dummy_bgr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        preprocessed = preprocess_for_inference(dummy_bgr)
        self.assertEqual(preprocessed.shape, (1, config.IMG_HEIGHT, config.IMG_WIDTH, 3))
        self.assertEqual(preprocessed.dtype, np.float32)


class TestModelArchitecture(unittest.TestCase):
    def test_fallback_cnn_shapes(self):
        model = _build_fallback_model(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            dropout_rate=0.3,
            learning_rate=1e-3,
        )
        self.assertEqual(model.input_shape, (None, config.IMG_HEIGHT, config.IMG_WIDTH, 3))
        self.assertEqual(model.output_shape, (None, 1))

        # Test forward pass with dummy tensor
        dummy_input = np.zeros((2, config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.float32)
        output = model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue((output >= 0.0).all() and (output <= 1.0).all())


class TestTemporalDrowsinessEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DrowsinessEngine(
            closure_threshold_sec=1.5,
            blink_max_duration_sec=0.3,
            smoothing_window=3,
            cooldown_sec=2.0
        )

    def test_alert_state_with_open_eyes(self):
        t = 0.0
        for _ in range(5):
            res = self.engine.update(eye_state_pred=0, confidence=0.95, current_time=t)
            t += 0.1
        self.assertEqual(res["state"], DrowsinessEngine.STATE_ALERT)
        self.assertEqual(res["closure_duration"], 0.0)
        self.assertFalse(res["alert_triggered"])

    def test_natural_blink_registered(self):
        t = 0.0
        # Start open (fill smoothing window with open)
        for _ in range(3):
            self.engine.update(0, current_time=t)
            t += 0.1
        # Close eyes (2 frames = 0.2s)
        self.engine.update(1, current_time=t)
        t += 0.1
        self.engine.update(1, current_time=t)
        t += 0.1
        # Open eyes again (2 frames to clear smoothing window)
        self.engine.update(0, current_time=t)
        t += 0.1
        res = self.engine.update(0, current_time=t)
        self.assertEqual(res["total_blinks"], 1)
        self.assertEqual(res["state"], DrowsinessEngine.STATE_ALERT)

    def test_sustained_closure_triggers_drowsiness(self):
        t = 0.0
        # Start closing eyes
        for _ in range(25): # 2.5 seconds (> 1.5s threshold)
            res = self.engine.update(eye_state_pred=1, confidence=0.98, current_time=t)
            t += 0.1

        self.assertEqual(res["state"], DrowsinessEngine.STATE_POSSIBLE_DROWSINESS)
        self.assertGreaterEqual(res["closure_duration"], 1.5)
        self.assertEqual(res["total_drowsy_events"], 1)

    def test_missing_face_detection_behavior(self):
        t = 0.0
        # If face is missing (None)
        res = self.engine.update(eye_state_pred=None, confidence=0.0, current_time=t)
        self.assertEqual(res["state"], DrowsinessEngine.STATE_NOT_DETECTED)
        self.assertIsNone(res["smoothed_eye_state"])
        self.assertFalse(res["alert_triggered"])


    def test_perclos_and_summary_metrics(self):
        t = 0.0
        # 10 frames open (0), then 10 frames closed (1)
        for _ in range(10):
            res = self.engine.update(eye_state_pred=0, confidence=0.9, current_time=t)
            t += 0.1
        for _ in range(10):
            res = self.engine.update(eye_state_pred=1, confidence=0.9, current_time=t)
            t += 0.1

        # PERCLOS should be 50%
        self.assertAlmostEqual(res["perclos"], 50.0, delta=1.0)
        summary = self.engine.get_summary()
        self.assertEqual(summary["total_frames"], 20)
        self.assertEqual(summary["total_closed_frames"], 10)
        self.assertAlmostEqual(summary["overall_perclos"], 50.0, delta=1.0)


class TestDatasetInspector(unittest.TestCase):
    def test_mrl_and_uta_statistics(self):
        from src.dataset_inspector import DatasetInspector
        mrl_stats = DatasetInspector.get_mrl_statistics()
        self.assertEqual(mrl_stats["total_images"], 84898)
        self.assertEqual(mrl_stats["unique_subjects"], 37)
        self.assertEqual(mrl_stats["splits"]["train"]["total"], 50937)

        uta_stats = DatasetInspector.get_uta_rldd_statistics()
        self.assertEqual(uta_stats["total_frames"], 11787)
        self.assertEqual(uta_stats["splits"]["train"]["total"], 9054)


class TestInferenceAndPreprocessing(unittest.TestCase):
    def test_get_eye_region_and_coords(self):
        from src.preprocessing import get_eye_region_and_coords
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Test with empty landmarks
        crop, coords = get_eye_region_and_coords(dummy_frame, [], [0, 1])
        self.assertIsNone(crop)
        self.assertIsNone(coords)

    def test_direct_crop_inference(self):
        from src.inference import InferenceEngine
        if os.path.exists(config.EYE_STATE_MODEL_PATH):
            engine = InferenceEngine(config.EYE_STATE_MODEL_PATH)
            # Create a 128x128 synthetic eye crop
            synth_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            result = engine.process_frame(synth_crop, allow_direct_crop_fallback=True)
            self.assertIsNotNone(result)
            self.assertTrue(result.is_direct_eye_crop)
            self.assertIn(result.overall_class, [0, 1])
            self.assertGreaterEqual(result.overall_confidence, 0.0)

    def test_sample_bundled_eye_crops(self):
        from src.inference import InferenceEngine
        if os.path.exists(config.EYE_STATE_MODEL_PATH):
            engine = InferenceEngine(config.EYE_STATE_MODEL_PATH)
            awake_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_eye_awake.png")
            sleepy_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_eye_sleepy.png")
            if os.path.exists(awake_path) and os.path.exists(sleepy_path):
                img_awake = cv2.imread(awake_path)
                res_awake = engine.process_frame(img_awake, allow_direct_crop_fallback=True)
                self.assertEqual(res_awake.overall_class, 0)
                self.assertGreaterEqual(res_awake.overall_confidence, 0.90)

                img_sleepy = cv2.imread(sleepy_path)
                res_sleepy = engine.process_frame(img_sleepy, allow_direct_crop_fallback=True)
                self.assertEqual(res_sleepy.overall_class, 1)
                self.assertGreaterEqual(res_sleepy.overall_confidence, 0.90)

    def test_sample_driver_face_detection(self):
        from src.inference import InferenceEngine
        from src.preprocessing import detect_faces_robust
        if os.path.exists(config.EYE_STATE_MODEL_PATH):
            engine = InferenceEngine(config.EYE_STATE_MODEL_PATH)
            for fname in ["sample_face_active.jpg", "sample_face_fatigue.jpg"]:
                p = os.path.join(config.PROJECT_ROOT, "sample_data", fname)
                if os.path.exists(p):
                    img = cv2.imread(p)
                    faces = detect_faces_robust(img)
                    self.assertGreaterEqual(len(faces), 1)
                    res = engine.process_frame(img, allow_direct_crop_fallback=False)
                    self.assertTrue(res.face_detected)
                    self.assertGreaterEqual(res.eyes_detected_count, 1)

    def test_manual_roi_crop_inference(self):
        from src.inference import InferenceEngine
        if os.path.exists(config.EYE_STATE_MODEL_PATH):
            engine = InferenceEngine(config.EYE_STATE_MODEL_PATH)
            test_img = np.ones((400, 400, 3), dtype=np.uint8) * 120
            res = engine.process_frame(test_img, manual_roi=(50, 50, 150, 150))
            self.assertTrue(res.is_direct_eye_crop)
            self.assertEqual(res.detection_method, "manual_roi")
            self.assertIsNotNone(res.left_eye)
            self.assertEqual(res.left_eye.name, "Manual Eye ROI")


class TestPathAndVideoValidation(unittest.TestCase):
    def test_nonexistent_dataset_path(self):
        from src.data_loader import discover_images
        paths, labels, _ = discover_images("non/existent/path/12345")
        self.assertEqual(len(paths), 0)
        self.assertEqual(len(labels), 0)

    def test_invalid_video_file_handling(self):
        invalid_path = "non_existent_video_file_123.mp4"
        cap = cv2.VideoCapture(invalid_path)
        self.assertFalse(cap.isOpened())
        cap.release()


if __name__ == "__main__":
    unittest.main()


