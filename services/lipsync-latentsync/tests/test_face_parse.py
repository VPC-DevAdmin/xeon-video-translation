"""Occlusion handling: per-frame paste weights and the parse-to-mask math."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
from latentsync_driver import face_parse as fp  # noqa: E402


def test_alpha_is_one_when_every_frame_has_a_face():
    assert fp.occlusion_alpha([True] * 5).tolist() == [1.0] * 5


def test_alpha_zero_through_the_gap_and_its_margin_then_ramps():
    visible = [True] * 6 + [False] * 3 + [True] * 6
    alpha = fp.occlusion_alpha(visible, margin=1, ramp=3)
    # gap at 6..8, margin widens it to 5..9
    assert alpha[5:10].tolist() == [0.0] * 5
    assert alpha[4] == pytest.approx(1 / 3) and alpha[3] == pytest.approx(2 / 3) and alpha[2] == 1.0
    assert alpha[10] == pytest.approx(1 / 3) and alpha[11] == pytest.approx(2 / 3) and alpha[12] == 1.0


def test_alpha_handles_edges_and_empty():
    assert fp.occlusion_alpha([]).tolist() == []
    alpha = fp.occlusion_alpha([False, True, True, True, True])
    assert alpha[0] == 0.0 and alpha[1] == 0.0 and alpha[4] == 1.0


def test_face_mask_keeps_face_classes_and_drops_objects():
    torch = pytest.importorskip("torch")
    parsing = torch.zeros((3, 64, 64), dtype=torch.uint8)
    parsing[:, 8:56, 8:56] = 1          # skin
    parsing[:, 40:48, 24:40] = 12       # lower lip
    parsing[:, 30:56, 30:56] = 0        # an object covering the lower right of the face
    mask = fp.face_mask_from_parsing(parsing, dilate=3, feather=3)
    assert mask.shape == (3, 1, 64, 64)
    assert float(mask[1, 0, 20, 20]) > 0.99      # skin
    assert float(mask[1, 0, 44, 26]) > 0.99      # lip
    assert float(mask[1, 0, 50, 50]) < 0.01      # object
    assert float(mask[1, 0, 2, 2]) < 0.01        # background


def test_face_mask_is_temporally_averaged():
    torch = pytest.importorskip("torch")
    parsing = torch.zeros((3, 32, 32), dtype=torch.uint8)
    parsing[1] = 1  # only the middle frame has face pixels
    mask = fp.face_mask_from_parsing(parsing, dilate=1, feather=1)
    assert float(mask[1, 0, 16, 16]) == pytest.approx(1 / 3)
    assert float(mask[0, 0, 16, 16]) == pytest.approx(1 / 3)
