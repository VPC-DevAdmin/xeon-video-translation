import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import jobs
from app.config import settings
from app.main import app
from app.pipeline import orchestrator as o
from app.pipeline._lipsync import latentsync_client


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_artifacts_dir", tmp_path)
    monkeypatch.setattr(settings, "min_free_disk_mb", 0)
    monkeypatch.setattr(settings, "auth_tokens_json", "{}")
    monkeypatch.setattr(jobs, "_kickoff", lambda *a: None)
    o._shutting_down = False
    for values in (o._jobs, o._queues, o._lanes, o._tasks):
        values.clear()
    yield
    for values in (o._jobs, o._queues, o._lanes, o._tasks):
        values.clear()


@pytest.mark.parametrize("mode,steps,tier", [("fast", 10, None), ("quality", 40, None)])
def test_submission_persists_render_settings(mode, steps, tier):
    client = TestClient(app)
    response = client.post(
        "/jobs",
        data={"target_language": "es", "mode": mode},
        files={"video": ("clip.mp4", b"test")},
    )
    assert response.status_code == 201, response.text
    state = o.get_job(response.json()["job_id"])
    assert state.lipsync_backend == "latentsync"
    assert state.lipsync_quality["num_inference_steps"] == steps
    assert state.lipsync_quality.get("service_tier") == tier


@pytest.mark.parametrize("value", ["0", "101", "1.5", "abc"])
def test_invalid_steps_rejected(value):
    response = TestClient(app).post(
        "/jobs",
        data={"target_language": "es", "mode": "quality", "latentsync_steps": value},
        files={"video": ("clip.mp4", b"test")},
    )
    assert response.status_code == 422


def test_explicit_settings_override_preset():
    client = TestClient(app)
    response = client.post(
        "/jobs",
        data={
            "target_language": "es",
            "mode": "quality",
            "latentsync_steps": "50",
            "latentsync_guidance": "0",
            "latentsync_seed": "0",
        },
        files={"video": ("clip.mp4", b"test")},
    )
    state = o.get_job(response.json()["job_id"])
    assert state.lipsync_quality == {"num_inference_steps": 50, "guidance_scale": 0.0, "seed": 0}


@pytest.mark.parametrize(
    "fast_url, tier, expected",
    [
        ("http://fast", "fast", "http://fast/lipsync"),
        ("", "fast", "http://batch/lipsync"),
        ("http://fast", None, "http://batch/lipsync"),
    ],
)
def test_independent_renderer_routing(tmp_path, monkeypatch, fast_url, tier, expected):
    monkeypatch.setattr(settings, "latentsync_service_url", "http://batch")
    monkeypatch.setattr(settings, "latentsync_fast_service_url", fast_url)
    result = MagicMock()
    result.__enter__.return_value.read.return_value = b'{"status":"ok"}'
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        return result

    monkeypatch.setattr(latentsync_client.urllib.request, "urlopen", urlopen)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"rendered")
    latentsync_client.run(
        tmp_path / "video",
        tmp_path / "audio",
        out,
        quality_overrides={"service_tier": tier, "num_inference_steps": 10},
    )
    assert calls[0].full_url == expected
    assert json.loads(calls[0].data)["num_inference_steps"] == 10
    assert "service_tier" not in json.loads(calls[0].data)


def test_face_track_keys_forwarded_to_latentsync(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "latentsync_service_url", "http://batch")
    result = MagicMock()
    result.__enter__.return_value.read.return_value = b'{"status":"ok"}'
    calls = []
    monkeypatch.setattr(latentsync_client.urllib.request, "urlopen", lambda request, **kw: calls.append(request) or result)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"rendered")
    latentsync_client.run(tmp_path / "video", tmp_path / "audio", out,
                          quality_overrides={"face_track_source": "/jobs/x/input.mov", "face_track_offset_frames": 400})
    payload = json.loads(calls[0].data)
    assert payload["face_track_source"] == "/jobs/x/input.mov" and payload["face_track_offset_frames"] == 400
