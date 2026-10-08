"""Per-person face tracks: mouth measure, clustering, frame-to-identity linking."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
from latentsync_driver import identities as ids  # noqa: E402


def _det(x, emb=None, mouth=0.1, score=0.9):
    return ids.Detection(np.array([x, 100, x + 100, 230], np.float32), np.full((3, 2), x, np.float32), score, mouth,
                         None if emb is None else np.asarray(emb, np.float32))


def test_mouth_opening_is_scale_free():
    pts = np.zeros((106, 2))
    pts[52], pts[61] = (0, 0), (40, 0)
    for upper, lower in ids.MOUTH_PAIRS:
        pts[upper], pts[lower] = (20, -2), (20, 6)
    assert ids.mouth_opening(pts) == pytest.approx(8 / 40)
    assert ids.mouth_opening(pts * 3) == pytest.approx(8 / 40)


def test_two_people_cluster_into_two_identities():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    a, b = np.eye(512)[0], np.eye(512)[1]
    emb = np.stack([a + 0.03 * rng.standard_normal(512) for _ in range(20)] + [b + 0.03 * rng.standard_normal(512) for _ in range(15)])  # same-person cosine ~0.8, like ArcFace
    labels, centroids = ids.cluster_embeddings(emb)
    assert len(centroids) == 2
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]


def test_identities_follow_people_who_swap_sides():
    pytest.importorskip("scipy")
    a, b = np.eye(4)[0], np.eye(4)[1]
    centroids = np.stack([a, b])
    frames = []
    for i in range(30):
        # A walks from x=100 to x=700, B from 700 to 100; embeddings every 10 frames
        xa, xb = 100 + 20 * i, 700 - 20 * i
        embed = i % 10 == 0
        frames.append([_det(xa, a if embed else None), _det(xb, b if embed else None)])
    assigned = ids.assign(frames, centroids, sample_every=10)
    for i, found in enumerate(assigned):
        assert found[0].bbox[0] == 100 + 20 * i and found[1].bbox[0] == 700 - 20 * i


def test_finalize_numbers_people_left_to_right_and_marks_absence():
    pytest.importorskip("scipy")
    right, left = _det(800), _det(100)
    assigned = [{0: right, 1: left}, {0: right}, {0: right, 1: left}]
    out = ids.finalize(assigned, 2, 25, smooth_window=1)
    assert out.centers[0][0] < out.centers[1][0]              # identity 0 is now the left person
    assert out.visible[0].tolist() == pytest.approx([0.9, 0.0, 0.9])
    assert np.isnan(out.mouth[0][1])
    assert out.landmarks[0][1, 0, 0] == 100                    # gap carried over
    assert out.presence == pytest.approx([2 / 3, 1.0])
