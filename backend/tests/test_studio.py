import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import state_store, storage, checkpoints
from app.config import settings
from app.pipeline import orchestrator as o
from app.api import studio, jobs
from app.main import app


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_artifacts_dir", tmp_path)
    monkeypatch.setattr(settings, "auth_tokens_json", "{}")
    monkeypatch.setattr(settings, "min_free_disk_mb", 0)
    o._shutting_down = False
    o._jobs.clear()
    o._queues.clear()
    o._lanes.clear()
    o._tasks.clear()
    o._speech_lock = None
    monkeypatch.setattr(jobs, "_kickoff", lambda *a: None)
    monkeypatch.setattr(studio, "_kickoff", lambda *a: None)
    yield
    o._jobs.clear()
    o._queues.clear()
    o._speech_lock = None


def job(identifier="a" * 32, owner="local", status="failed", failed="translate"):
    state = o.JobState(
        identifier, owner_id=owner, status=status, input_filename="clip.mp4", target_language="es"
    )
    state.stages = [
        o.StageResult(
            name=n,
            status=o.StageStatus.DONE if i < o.STAGE_NAMES.index(failed) else o.StageStatus.FAILED,
        )
        for i, n in enumerate(o.STAGE_NAMES)
    ]
    o.register_job(state)
    root = storage.job_dir(identifier)
    (root / "input.mp4").write_bytes(b"input")
    for stage, names in checkpoints.ARTIFACTS.items():
        for name in names:
            if name.endswith(".json"):
                (root / name).write_text(
                    json.dumps(
                        {
                            "language": "en",
                            "text": "Hello",
                            "segments": [{"start": 0, "end": 1, "text": "Hello"}],
                        }
                    )
                )
            else:
                (root / name).write_bytes(b"artifact")
        checkpoints.record(root, stage)
    return state


def test_transactional_lease_single_winner():
    state_store.save("a" * 32, {"job_id": "a" * 32, "status": "queued", "owner_id": "local"})
    with ThreadPoolExecutor(8) as pool:
        wins = list(pool.map(lambda i: state_store.claim("a" * 32, str(i)), range(8)))
    assert sum(wins) == 1
    assert not state_store.heartbeat("a" * 32, "wrong")
    state_store.release("a" * 32, "wrong")
    assert not state_store.claim("a" * 32, "other")
    state_store.recover_interrupted()
    assert state_store.claim("a" * 32, "restart")


def test_sqlite_authoritative_and_checksum(tmp_path):
    state = job()
    root = storage.job_dir(state.job_id)
    (root / "meta.json").write_text("broken")
    assert storage.read_meta(state.job_id)["status"] == "failed"
    assert checkpoints.valid(root, "audio")
    (root / "audio.wav").write_bytes(b"corrupt")
    assert not checkpoints.valid(root, "audio")


def test_revision_preserves_edit_and_parent():
    original = job()
    with TestClient(app) as client:
        response = client.post(
            "/jobs/" + original.job_id + "/revise", json={"translation": ["Hola"]}
        )
        assert response.status_code == 201, response.text
        new = response.json()["job_id"]
        root = storage.job_dir(new)
        assert json.loads((root / "translation.json").read_text())["segments"][0]["text"] == "Hola"
        state = o.get_job(new)
        assert state.stages[o.STAGE_NAMES.index("translate")].status == o.StageStatus.DONE
        assert state.stages[o.STAGE_NAMES.index("tts")].status == o.StageStatus.PENDING
        assert not (root / "translated_audio.wav").exists()
        assert original.job_id == "a" * 32 and original.status == "failed"
        assert (
            json.loads((storage.job_dir(original.job_id) / "translation.json").read_text())["text"]
            == "Hello"
        )
        assert (
            client.post(
                "/jobs/" + original.job_id + "/revise",
                json={"translation": ["Hola"], "from_stage": "audio"},
            ).status_code
            == 400
        )


def test_ownership_all_download_and_mutation_routes(monkeypatch):
    job(owner="alice")
    monkeypatch.setattr(
        settings, "auth_tokens_json", json.dumps({"alice-token": "alice", "bob-token": "bob"})
    )
    with TestClient(app) as client:
        assert client.get("/jobs").status_code == 401
        client.headers["Authorization"] = "Bearer bob-token"
        assert client.get("/jobs").json()["jobs"] == []
        for path in (
            "",
            "/artifacts",
            "/artifacts/input.mp4",
            "/bundle",
            "/subtitles/srt",
            "/stream",
        ):
            assert client.get("/jobs/" + "a" * 32 + path).status_code == 404, path
        assert client.post("/jobs/" + "a" * 32 + "/revise", json={}).status_code == 404
        assert client.delete("/jobs/" + "a" * 32).status_code == 404
        client.headers["Authorization"] = "Bearer alice-token"
        assert client.get("/jobs/" + "a" * 32 + "/bundle").status_code == 200
        assert "Hello" in client.get("/jobs/" + "a" * 32 + "/subtitles/vtt").text


def test_idempotent_retry_at_capacity(monkeypatch):
    job(status="queued")
    monkeypatch.setattr(settings, "max_user_jobs", 1)
    # Do not start recovery for the intentionally queued fixture.
    client = TestClient(app)
    response = client.post(
        "/jobs",
        data={"request_id": "a" * 32, "target_language": "es"},
        files={"video": ("x.mp4", b"x")},
    )
    assert response.status_code == 201, response.text
    assert (
        client.post(
            "/jobs", data={"target_language": "es"}, files={"video": ("x.mp4", b"x")}
        ).status_code
        == 429
    )


@pytest.mark.asyncio
async def test_scheduler_priority_cancellation_and_aging():
    from app.scheduler import SpeechScheduler

    scheduler = SpeechScheduler()
    order = []

    async def worker(label, priority):
        async with scheduler.lease(priority):
            order.append(label)

    async with scheduler.lease():
        batch = asyncio.create_task(worker("batch", 20))
        cancelled = asyncio.create_task(worker("cancelled", 0))
        fast = asyncio.create_task(worker("avatar", 0))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
    await asyncio.gather(batch, fast)
    assert order == ["avatar", "batch"] and not scheduler.busy
    async with scheduler.lease():
        batch = asyncio.create_task(worker("aged", 20))
        await asyncio.sleep(0)
        priority, index, future, at = scheduler.waiters[0]
        scheduler.waiters[0] = (priority, index, future, at - 900)
        fast = asyncio.create_task(worker("new", 0))
        await asyncio.sleep(0)
    await asyncio.gather(batch, fast)
    assert order[-2:] == ["aged", "new"]


def test_metrics_owner_percentiles():
    job()
    job("b" * 32, owner="bob")
    for value in range(1, 21):
        state_store.metric("a" * 32, "tts_seconds", value)
    state_store.metric("b" * 32, "tts_seconds", 10000)
    metric = state_store.metrics("local")[0]
    assert metric["p50"] == 10 and metric["p95"] == 19 and metric["count"] == 20


def test_short_spans_are_never_deleted(tmp_path, monkeypatch):
    from app.pipeline import tts
    from unittest.mock import Mock

    monkeypatch.setattr(tts, "_non_silent_spans", lambda _: [(0.1, 0.2), (1, 2), (3, 3.1)])
    monkeypatch.setattr(tts, "_probe_duration", lambda _: 4)
    trim = Mock()
    monkeypatch.setattr(tts, "_ffmpeg_atrim", trim)
    tts._trim_to_speech(tmp_path / "voice.wav")
    assert trim.call_args.args[2] <= 0.1 and trim.call_args.args[3] >= 3.1


def test_quality_rules_and_subtitle_injection():
    from app.pipeline.quality import issues

    assert not issues("Cost 12 Xeon", "Precio ١٢ Xeon", {"Xeon": "Xeon"})
    assert len(issues("12 Xeon", "13 server", {"Xeon": "Xeon"})) == 2
    text = studio.subtitles([dict(start=0, end=1.25, text="a\n\n00 --> b")], "vtt")
    assert text.startswith("WEBVTT") and text.count("-->") == 1 and "00:00:01.250" in text
