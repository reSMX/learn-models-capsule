from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from galar_dual_model.labels import ANATOMY_LABELS, PATHOLOGY_LABELS
from galar_dual_model.tools.extract_pathology_frames import Candidate, select_frames
from galar_dual_model.tools.video_inference import make_transform, predict_scores


class FixedLogits(nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([0.0, 2.0, -2.0], dtype=torch.float32)
        return logits.repeat(len(batch), 1)


class VideoToolTests(unittest.TestCase):
    def test_canonical_class_counts(self) -> None:
        self.assertEqual(len(ANATOMY_LABELS), 9)
        self.assertEqual(len(PATHOLOGY_LABELS), 12)

    def test_validation_transform_shape(self) -> None:
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        tensor = make_transform()(frame)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_sigmoid_scores_are_independent(self) -> None:
        batch = torch.zeros(2, 3, 224, 224)
        scores = predict_scores(FixedLogits(), batch, torch.device("cpu"))
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())
        self.assertFalse(np.allclose(scores.sum(axis=1), 1.0))

    def test_frame_selection_respects_requested_gap(self) -> None:
        candidates = [
            Candidate(0.99, 0, 0, (0.99,)),
            Candidate(0.98, 5, 0, (0.98,)),
            Candidate(0.97, 25, 0, (0.97,)),
        ]
        selected = select_frames(candidates, count=2, fps=10.0, min_gap_seconds=2.0)
        self.assertEqual([item.frame_index for item in selected], [0, 25])


if __name__ == "__main__":
    unittest.main()
