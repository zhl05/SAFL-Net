"""Synthetic checks for data handling, losses, metrics and command-line interfaces."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys

import cv2
import numpy as np
import torch

from safl.backend import build_model, build_optimizer, ModelBackendUnavailable
from safl.data import UltrasoundDataset, fixed_bbox, fixed_click, read_manifest
from safl.losses import SegmentationLoss, ClassificationLoss, joint_loss
from safl.metrics import boundary_distances, summarize


class WorkflowTests(unittest.TestCase):
    @patch.dict('os.environ', {'SAFL_BACKEND': 'safl_test_backend_not_installed'})
    def test_missing_backend_reports_configuration_error(self):
        with self.assertRaises(ModelBackendUnavailable):
            build_model()
        with self.assertRaises(ModelBackendUnavailable):
            build_optimizer(None)

    def test_prompt_coordinates_and_empty_mask(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        point, labels = fixed_click(mask)
        self.assertEqual(labels, [1])
        self.assertEqual(mask[point[0, 1], point[0, 0]], 1)
        np.testing.assert_array_equal(fixed_bbox(mask), [3, 2, 6, 4])
        point, labels = fixed_click(np.zeros_like(mask))
        self.assertEqual(labels, [0])
        np.testing.assert_array_equal(fixed_bbox(np.zeros_like(mask), img_size=8), [-1, -1, 8, 8])

    def test_synthetic_dataset_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for subfolder in ("Image", "Label"):
                (root / "BUSBRA" / subfolder).mkdir(parents=True)
            image = np.arange(256, dtype=np.uint8).reshape(16, 16)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[3:10, 4:12] = 255
            cv2.imwrite(str(root / "BUSBRA/Image/synthetic.png"), image)
            cv2.imwrite(str(root / "BUSBRA/Label/synthetic.png"), mask)
            manifest = root / "cases.txt"
            manifest.write_text("1/BUSBRA/synthetic/1\n", encoding="utf-8")
            dataset = UltrasoundDataset(root, manifest)
            first, second = dataset[0], dataset[0]
            self.assertEqual(first["image"].shape, (1, 256, 256))
            self.assertEqual(first["low_mask"].shape, (1, 128, 128))
            self.assertEqual(first["cls_label"], 1)
            self.assertTrue(torch.equal(first["image"], second["image"]))
            self.assertTrue(set(first["label"].unique().tolist()).issubset({0, 1}))
            for invalid in ("1/../synthetic/1", "1/BUSBRA/synthetic/2", "", "1/BUSBRA/synthetic/1\n1/BUSBRA/synthetic/0"):
                manifest.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_manifest(manifest)

    def test_boundary_distance_policy(self):
        empty = np.zeros((8, 8), dtype=bool)
        positive = empty.copy()
        positive[2:5, 2:5] = True
        self.assertEqual(boundary_distances(empty, empty), (0.0, 0.0, False))
        self.assertEqual(boundary_distances(positive, empty), (None, None, True))
        self.assertEqual(boundary_distances(positive, positive), (0.0, 0.0, False))

    def test_classification_metrics_and_single_class_auc(self):
        rows = [dict(label=0, score=0.1, dice=1.0, iou=1.0),
                dict(label=1, score=0.9, dice=0.5, iou=1 / 3)]
        result = summarize(rows)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["dice"], 0.75)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertIsNone(summarize(rows[:1])["auc"])
        with self.assertRaises(ValueError):
            summarize([])

    def test_fp32_joint_loss_and_gradient(self):
        mask_logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
        class_logits = torch.zeros((2, 2), requires_grad=True)
        total, seg, cls = joint_loss({"low_res_logits": mask_logits}, class_logits,
                                      torch.ones_like(mask_logits), torch.tensor([0, 1]),
                                      SegmentationLoss(), ClassificationLoss())
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.allclose(total, seg + cls))
        total.backward()
        self.assertIsNotNone(mask_logits.grad)
        self.assertIsNotNone(class_logits.grad)

    def test_entrypoint_help(self):
        root = Path(__file__).resolve().parents[1]
        for entry in ("train.py", "test.py", "evaluate.py"):
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, str(root / entry), "--help"],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--data-root", result.stdout)

    @patch.dict('os.environ', {'SAFL_BACKEND': 'safl_test_backend_not_installed'})
    def test_entrypoints_stop_before_loading_data_without_backend(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            commands = {
                "train.py": ["--train-manifest", "train.txt", "--val-manifest", "val.txt"],
                "test.py": ["--manifest", "test.txt", "--checkpoint", "checkpoint.pth"],
                "evaluate.py": ["--manifest", "test.txt", "--checkpoint", "checkpoint.pth"],
            }
            for entry, options in commands.items():
                with self.subTest(entry=entry):
                    result = subprocess.run(
                        [sys.executable, str(root / entry), "--data-root", "data",
                         "--output", str(output), "--device", "cpu"] + options,
                        cwd=root, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Model backend", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
