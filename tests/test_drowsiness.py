"""
Automated unit and integration tests for Drowsiness Detection System (DPD Migration).

Tests cover:
1. DPD dataset discovery and label mapping (0=Active, 1=Fatigue)
2. Stratified train/val/test splitting integrity and balance
3. Face image preprocessing shapes and value normalization
4. Model architecture input/output shapes and forward pass
5. Temporal decision engine logic (Alert vs Transient Fatigue vs Sustained Drowsiness Alert)
6. Missing face detection handling and reset behavior
7. DatasetInspector DPD querying and sample retrieval
8. Video processing and sequence extraction utilities
9. Full-frame and direct-face inference pipeline execution
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import (
    discover_dpd_images,
    stratified_split,
    get_class_distribution,
    compute_class_weights,
    filter_splits_for_face_crops,
)
from src.preprocessing import (
    preprocess_for_inference,
    detect_faces_robust,
    crop_face_from_frame,
)
from src.model import _build_fallback_model
from src.drowsiness_engine import DrowsinessEngine
from src.dataset_inspector import DatasetInspector
from src.video_dataset import get_video_metadata


class TestDPDDatasetLoading(unittest.TestCase):
    def test_split_filter_excludes_images_without_cached_face_crops(self):
        splits = {
            "train": (["a.jpg", "b.jpg"], [0, 1]),
            "val": (["c.jpg"], [1]),
            "test": (["d.jpg"], [0]),
        }
        filtered = filter_splits_for_face_crops(
            splits,
            {"a.jpg": "a_crop.jpg", "c.jpg": "c_crop.jpg", "d.jpg": "d_crop.jpg"},
        )
        self.assertEqual(filtered["train"], (["a.jpg"], [0]))
        self.assertEqual(filtered["val"], (["c.jpg"], [1]))
        self.assertEqual(filtered["test"], (["d.jpg"], [0]))

    def test_discovery_uses_requested_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "custom_faces"
            (root / "Active Subjects").mkdir(parents=True)
            (root / "Fatigue Subjects").mkdir(parents=True)
            (root / "Active Subjects" / "a.jpg").touch()
            (root / "Fatigue Subjects" / "f.png").touch()

            paths, labels = discover_dpd_images(str(root))

            self.assertEqual(len(paths), 2)
            self.assertEqual(
                {Path(path).name: label for path, label in zip(paths, labels)},
                {"a.jpg": 0, "f.png": 1},
            )

    def test_discovery_rejects_missing_class_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "Active Subjects").mkdir()
            with self.assertRaises(FileNotFoundError):
                discover_dpd_images(temp_dir)

    def test_dpd_discovery_and_balance(self):
        paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
        self.assertGreater(len(paths), 0, "DPD dataset should have images")
        dist = get_class_distribution(labels)
        self.assertIn("Active", dist)
        self.assertIn("Fatigue", dist)
        self.assertEqual(dist["Active"], dist["Fatigue"], "DPD classes should be balanced")

    def test_stratified_split_integrity(self):
        paths, labels = discover_dpd_images(config.DPD_DATASET_DIR)
        splits = stratified_split(paths, labels, train_ratio=0.70, val_ratio=0.15, seed=42)

        train_p, train_l = splits["train"]
        val_p, val_l = splits["val"]
        test_p, test_l = splits["test"]

        # Ensure no overlap
        train_set = set(train_p)
        val_set = set(val_p)
        test_set = set(test_p)

        self.assertEqual(len(train_set.intersection(val_set)), 0, "Train and Val splits must not overlap")
        self.assertEqual(len(train_set.intersection(test_set)), 0, "Train and Test splits must not overlap")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "Val and Test splits must not overlap")
        self.assertEqual(len(train_p) + len(val_p) + len(test_p), len(paths))

    def test_class_weights_computation(self):
        labels = [0, 0, 0, 0, 1]
        weights = compute_class_weights(labels)
        self.assertGreater(weights[1], weights[0], "Underrepresented class should receive higher weight")


class TestImagePreprocessing(unittest.TestCase):
    def test_preprocess_for_inference_shape_and_range(self):
        dummy_bgr = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
        preprocessed = preprocess_for_inference(dummy_bgr)
        self.assertEqual(preprocessed.shape, (1, config.IMG_HEIGHT, config.IMG_WIDTH, 3))
        self.assertEqual(preprocessed.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(preprocessed)), -1.0)
        self.assertLessEqual(float(np.max(preprocessed)), 1.0)

    def test_crop_face_from_frame(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        crop, bbox, method = crop_face_from_frame(frame)
        # Empty frame has no detected faces
        self.assertIsNone(crop)
        self.assertIsNone(bbox)
        self.assertEqual(method, "none")


class TestModelArchitecture(unittest.TestCase):
    def test_fallback_cnn_shapes_and_prediction(self):
        model = _build_fallback_model(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            dropout_rate=0.3,
            learning_rate=1e-3,
        )
        self.assertEqual(model.input_shape, (None, config.IMG_HEIGHT, config.IMG_WIDTH, 3))
        self.assertEqual(model.output_shape, (None, 1))

        dummy_input = np.zeros((2, config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.float32)
        output = model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue((output >= 0.0).all() and (output <= 1.0).all())


class TestTemporalDrowsinessEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DrowsinessEngine(
            fatigue_threshold_sec=1.5,
            blink_max_duration_sec=0.3,
            smoothing_window=3,
            cooldown_sec=2.0,
        )

    def test_alert_state_with_active_subject(self):
        t = 0.0
        for _ in range(5):
            res = self.engine.update(pred_class=0, confidence=0.95, current_time=t)
            t += 0.1
        self.assertEqual(res["state"], DrowsinessEngine.STATE_ALERT)
        self.assertEqual(res["fatigue_duration"], 0.0)
        self.assertFalse(res["alert_triggered"])

    def test_low_confidence_is_uncertain(self):
        engine = DrowsinessEngine(smoothing_window=3, min_confidence=0.8)
        res = engine.update(pred_class=1, confidence=0.6, current_time=0.0)
        self.assertEqual(res["state"], DrowsinessEngine.STATE_NOT_DETECTED)
        self.assertTrue(res["uncertain"])

    def test_transient_fatigue_registered(self):
        t = 0.0
        # Active baseline
        for _ in range(3):
            self.engine.update(0, current_time=t)
            t += 0.1
        # Brief fatigue (2 frames = 0.2s <= 0.3s)
        self.engine.update(1, current_time=t)
        t += 0.1
        self.engine.update(1, current_time=t)
        t += 0.1
        # Back to active
        self.engine.update(0, current_time=t)
        t += 0.1
        res = self.engine.update(0, current_time=t)
        self.assertEqual(res["state"], DrowsinessEngine.STATE_ALERT)
        self.assertEqual(self.engine.total_transient_events, 1)

    def test_sustained_fatigue_triggers_drowsiness_alert(self):
        t = 0.0
        # 25 consecutive fatigue frames (2.5 seconds > 1.5s threshold)
        for _ in range(25):
            res = self.engine.update(pred_class=1, confidence=0.98, current_time=t)
            t += 0.1

        self.assertEqual(res["state"], DrowsinessEngine.STATE_DROWSINESS_ALERT)
        self.assertGreaterEqual(res["fatigue_duration"], 1.5)
        self.assertEqual(res["total_drowsy_events"], 1)

    def test_missing_face_behavior(self):
        t = 0.0
        res = self.engine.update(pred_class=None, confidence=0.0, current_time=t)
        self.assertEqual(res["state"], DrowsinessEngine.STATE_NOT_DETECTED)
        self.assertIsNone(res["smoothed_class"])
        self.assertFalse(res["alert_triggered"])


class TestDatasetInspector(unittest.TestCase):
    def test_dpd_statistics(self):
        stats = DatasetInspector.get_dpd_statistics()
        self.assertTrue(stats["exists"])
        self.assertEqual(stats["total_images"], 9120)
        self.assertEqual(stats["total_active"], 4560)
        self.assertEqual(stats["total_fatigue"], 4560)
        self.assertEqual(stats["active_pct"], 50.0)
        self.assertEqual(stats["fatigue_pct"], 50.0)

    def test_get_sample_images(self):
        samples = DatasetInspector.get_sample_images(count_per_class=2)
        self.assertIn("Active", samples)
        self.assertIn("Fatigue", samples)
        self.assertEqual(len(samples["Active"]), 2)
        self.assertEqual(len(samples["Fatigue"]), 2)
        for p in samples["Active"] + samples["Fatigue"]:
            self.assertTrue(os.path.isfile(p))


class TestVideoUtilities(unittest.TestCase):
    def test_video_metadata_nonexistent(self):
        meta = get_video_metadata("non_existent_video_path.mp4")
        self.assertFalse(meta["exists"])


if __name__ == "__main__":
    unittest.main()
