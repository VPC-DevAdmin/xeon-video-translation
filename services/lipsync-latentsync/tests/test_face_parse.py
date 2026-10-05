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


def test_alpha_fades_on_low_detection_confidence():
    conf = np.array([0.95, 0.95, 0.95, 0.6, 0.95, 0.95, 0.95], np.float32)
    alpha = fp.occlusion_alpha(conf)
    assert alpha[0] == 1.0 and alpha[6] == 1.0
    expected = (0.6 - fp.CONF_LOW) / (fp.CONF_HIGH - fp.CONF_LOW)
    assert alpha[3] == pytest.approx(expected, abs=1e-5)
    assert alpha[2] == pytest.approx(expected, abs=1e-5) and alpha[4] == pytest.approx(expected, abs=1e-5)  # neighbours too
    assert alpha[1] == 1.0


def test_alpha_confidence_gap_still_zero_with_ramp():
    conf = np.array([0.9] * 5 + [0.0] * 2 + [0.9] * 5, np.float32)
    alpha = fp.occlusion_alpha(conf, margin=1, ramp=3)
    assert alpha[4:8].tolist() == [0.0] * 4
    assert alpha[0] == 1.0 and alpha[11] == 1.0


def _face_sequence(torch, n=40, size=64):
    crops = torch.full((n, 3, size, size), 150, dtype=torch.uint8)
    crops[:, :, 20:44, 20:44] = 90  # darker face centre
    return crops


def test_occluder_entering_from_the_border_is_masked_and_mouth_motion_is_not():
    torch = pytest.importorskip("torch")
    crops = _face_sequence(torch)
    # a bright object slides in from the right over frames 18..23
    for k, i in enumerate(range(18, 24)):
        crops[i, :, 10:54, 64 - 8 * (k + 1):] = 250
    # the "mouth" opens on frames 30..33: a compact dark blob in the middle
    crops[30:34, :, 36:42, 28:36] = 20
    visible = np.ones(len(crops), bool)
    occ = fp.occluder_masks(crops, visible, size=64, window=12, threshold=32, min_area=0.02, dilate=3)
    assert occ.shape == (40, 1, 64, 64)
    assert float(occ[21, 0, 30, 60]) == 1.0           # object pixels flagged
    assert float(occ[21, 0, 30, 10]) == 0.0           # face left of it untouched
    assert float(occ[31, 0, 39, 32]) == 0.0           # mouth motion is not an occluder
    assert float(occ[5].sum()) == 0.0                 # clean frames stay clean


def test_pose_change_that_persists_is_not_an_occluder():
    torch = pytest.importorskip("torch")
    crops = _face_sequence(torch)
    crops[20:, :, :, 40:] = 200  # from frame 20 on the right side is lit differently and stays so
    occ = fp.occluder_masks(crops, np.ones(40, bool), size=64, window=12, threshold=32)
    assert float(occ[25:35].sum()) == 0.0


def test_default_threshold_catches_a_pale_object_crossing_the_mouth():
    torch = pytest.importorskip("torch")
    crops = _face_sequence(torch)
    # a pale object only 25 levels brighter than the face slides over the lower half
    for k, i in enumerate(range(18, 24)):
        crops[i, :, 32:, 64 - 10 * (k + 1):] = 175
    occ = fp.occluder_masks(crops, np.ones(40, bool), size=64, window=12)
    assert float(occ[21, 0, 50, 50]) == 1.0
    assert float(occ[5].sum()) == 0.0


def test_ranges_compacts_runs():
    assert fp.ranges([]) == ""
    assert fp.ranges([3, 4, 5, 9, 12, 13]) == "3-5, 9, 12-13"


def test_hand_mask_is_hand_shaped_not_a_hull():
    pytest.importorskip("cv2")
    # an open hand: wrist at the bottom, five fingers fanning up, in a 512 crop
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (256, 480)
    bases = {1: (200, 420), 5: (220, 380), 9: (256, 370), 13: (292, 380), 17: (312, 400)}
    for b, (x, y) in bases.items():
        for k in range(4):
            pts[b + k] = (x + (x - 256) * 0.25 * k, y - 50 * k)
    mask = fp.hand_mask_from_landmarks(pts, size=512)
    assert mask.shape == (512, 512)
    assert mask[int(pts[8, 1]), int(pts[8, 0])] == 1          # fingertip covered
    assert mask[int(pts[9, 1]) - 30, int(pts[9, 0])] == 1      # along the middle finger
    gap = (pts[8] + pts[12]) / 2 - np.array([0, 60])          # between index and middle tips
    assert mask[int(gap[1]), int(gap[0])] == 0                 # the hull would have covered this


def test_silent_frames_marks_long_runs_minus_their_lead():
    sr, fps = 16000, 25
    audio = np.zeros(sr * 4, np.float32)                 # 4 s = 100 frames
    audio[: int(2.0 * sr)] = 0.3                         # speech for 2 s (frames 0-49)
    audio[int(2.6 * sr):int(2.8 * sr)] = 0.3             # a 0.2 s blip (frames 65-69)
    silent = fp.silent_frames(audio, sr, fps, 100, min_run=10, lead=3)
    assert not silent[:50].any()
    assert not silent[50] and not silent[51] and not silent[52]   # lead frames stay
    assert silent[54] and silent[60]
    assert not silent[67]                                # the blip is voiced
    assert silent[75] and silent[99]                     # trailing silence to the end


def test_silent_frames_short_pause_is_ignored():
    sr, fps = 16000, 25
    audio = np.full(sr * 2, 0.3, np.float32)
    audio[int(1.0 * sr):int(1.2 * sr)] = 0.0             # 0.2 s pause = 5 frames
    assert not fp.silent_frames(audio, sr, fps, 50, min_run=10).any()


def test_pick_closed_mouth_prefers_clear_frames():
    mouth_open = np.array([0.00, 0.02, 0.01, 0.05], np.float32)
    alpha = np.array([0.0, 1.0, 1.0, 1.0], np.float32)       # frame 0 is gated
    covered = np.array([0.0, 0.0, 0.4, 0.0], np.float32)     # frame 2 has a hand
    assert fp.pick_closed_mouth(mouth_open, alpha, covered) == 1
    assert fp.pick_closed_mouth(mouth_open, np.zeros(4)) is None
