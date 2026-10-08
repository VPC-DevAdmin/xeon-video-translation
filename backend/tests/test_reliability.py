import asyncio
import threading
from unittest.mock import Mock
import pytest
from app import storage
from app.config import settings
from app.events import EventLog
from app.pipeline import orchestrator as o, tts


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_artifacts_dir", tmp_path)
    o._shutting_down = False
    o._jobs.clear()
    o._queues.clear()
    o._tasks.clear()
    o._lanes.clear()
    o._speech_lock = None
    yield
    o._jobs.clear()
    o._queues.clear()
    o._tasks.clear()
    o._lanes.clear()
    o._speech_lock = None


def test_storage_read_does_not_create_and_rejects_traversal(tmp_path):
    assert storage.read_meta("a" * 32) is None
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        storage.job_dir("..")
    storage.write_meta("a" * 32, {"status": "queued"})
    assert storage.read_meta("a" * 32) == {"status": "queued"}
    assert not list((tmp_path / ("a" * 32)).glob(".meta-*"))


@pytest.mark.asyncio
async def test_sse_broadcast_and_replay():
    log = EventLog()
    for name in ["job_started", "job_completed", "stream_end"]:
        await log.put({"event": name, "data": {}})

    async def read(cursor=0):
        return [event["event"] async for _, event in log.subscribe(cursor)]

    left, right = await asyncio.gather(read(), read())
    assert left == right == ["job_started", "job_completed", "stream_end"]
    assert await read(1) == ["job_completed", "stream_end"]


@pytest.mark.asyncio
async def test_cancel_waits_for_native_inference_to_exit():
    started, finish = threading.Event(), threading.Event()

    def native():
        started.set()
        finish.wait(2)

    task = asyncio.create_task(o.blocking_call(native))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_batch_does_not_block_fast_lane(monkeypatch, tmp_path):
    batch_entered, release = asyncio.Event(), asyncio.Event()

    async def stage(*args):
        pass

    async def stabilize(state, queue, path):
        return path

    async def lipsync(state, queue, path):
        if state.lipsync_backend == "latentsync":
            batch_entered.set()
            await release.wait()

    for name in ["audio", "transcribe", "translate", "tts", "poststabilize", "mux"]:
        monkeypatch.setattr(o, "_run_stage_" + name, stage)
    monkeypatch.setattr(o, "_run_stage_stabilize", stabilize)
    monkeypatch.setattr(o, "_run_stage_lipsync", lipsync)
    batch = o.JobState("b" * 32, lipsync_backend="latentsync")
    fast = o.JobState("f" * 32, lipsync_backend="musetalk")
    o.register_job(batch)
    o.register_job(fast)
    task = asyncio.create_task(o.run_pipeline(batch, tmp_path / "input.mp4"))
    await batch_entered.wait()
    await asyncio.wait_for(o.run_pipeline(fast, tmp_path / "input.mp4"), 1)
    assert fast.status == "completed" and batch.status == "running"
    release.set()
    await task
    assert not o._jobs and not o._queues and not o._tasks


def test_failed_tts_segment_is_retried_and_never_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_select_reference", lambda *a: (tmp_path / "ref.wav", "ref"))
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 0.1)
    attempts = []

    def synth(text, reference, language, path):
        attempts.append(text)
        if text == "second":
            raise RuntimeError("failed")
        path.write_bytes(b"wav")

    monkeypatch.setattr(tts, "_xtts_to_file", synth)
    assemble = Mock()
    monkeypatch.setattr(tts, "_assemble_timeline", assemble)
    segments = [{"text": "Si", "start": 0, "end": 1}, {"text": "second", "start": 1, "end": 2}]
    with pytest.raises(tts.TTSError, match="segment 2 failed"):
        tts._synthesize_per_segment(
            segments, segments, tmp_path / "ref.wav", "es", tmp_path / "out.wav"
        )
    assert attempts == ["Si", "second", "second"]
    assemble.assert_not_called()


def test_short_nonempty_translations_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_select_reference", lambda *a: (tmp_path / "ref.wav", "ref"))
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 0.1)
    monkeypatch.setattr(
        tts, "_xtts_to_file", lambda text, ref, lang, path: path.write_bytes(b"wav")
    )
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    segments = [{"text": "是", "start": 0, "end": 1}]
    assert (
        tts._synthesize_per_segment(
            segments, segments, tmp_path / "ref.wav", "zh", tmp_path / "out.wav"
        )
        == 1
    )


def test_tts_overflow_fails_without_cutting_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_select_reference", lambda *a: (tmp_path / "ref.wav", "ref"))
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 3)
    monkeypatch.setattr(
        tts, "_xtts_to_file", lambda text, ref, lang, path, **k: path.write_bytes(b"wav")
    )
    segments = [{"text": "long speech", "start": 0, "end": 1}]
    monkeypatch.setattr(tts.settings, "tts_max_speed_last_resort", 1.7)  # 3x is beyond even the last resort
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
        tts._synthesize_per_segment(
            segments, segments, tmp_path / "ref.wav", "en", tmp_path / "out.wav"
        )


def test_mode_and_idempotent_submission(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import jobs

    monkeypatch.setattr(jobs, "_kickoff", lambda *a: None)
    with TestClient(app) as client:
        data = {"target_language": "es", "mode": "quality", "request_id": "c" * 32}
        first = client.post(
            "/jobs", data=data, files={"video": ("clip.mp4", b"placeholder", "video/mp4")}
        )
        second = client.post(
            "/jobs", data=data, files={"video": ("clip.mp4", b"placeholder", "video/mp4")}
        )
        assert first.status_code == second.status_code == 201
        assert first.json()["job_id"] == second.json()["job_id"]
        state = o.get_job("c" * 32)
        assert state.lipsync_backend == "latentsync"
        assert state.enable_output_stabilization is False
        assert client.get("/jobs/bad").status_code == 400


def test_timeline_preserves_second_onset(tmp_path):
    import subprocess, wave, array

    files = []
    for i, frequency in enumerate([400, 800]):
        path = tmp_path / f"{i}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:duration=0.2",
                "-ar",
                "24000",
                str(path),
            ],
            check=True,
        )
        files.append(path)
    output = tmp_path / "timeline.wav"
    tts._assemble_timeline(files, [{"start": 1, "end": 2}, {"start": 4, "end": 5}], output)
    with wave.open(str(output)) as audio:
        samples = array.array("h", audio.readframes(audio.getnframes()))
    assert max(abs(s) for s in samples[24000 : 2 * 24000]) == 0
    assert max(abs(s) for s in samples[3 * 24000 : 3 * 24000 + 2000]) > 100


@pytest.mark.asyncio
async def test_restart_marks_interrupted_job_failed(tmp_path):
    state = o.JobState(job_id="e" * 32, status="running")
    storage.write_meta(state.job_id, state.to_dict())
    await o.recover_jobs()
    recovered = storage.read_meta(state.job_id)
    assert recovered["status"] == "failed"
    assert "restarted" in recovered["error"]


def test_empty_upload_can_retry_same_identifier(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import jobs

    async def no_work(*args):
        pass

    monkeypatch.setattr(jobs, "_kickoff", no_work)
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            data={"target_language": "es", "request_id": "f" * 32},
            files={"video": ("input.mp4", b"", "video/mp4")},
        )
        assert response.status_code == 422
        response = client.post(
            "/jobs",
            data={"target_language": "es", "request_id": "f" * 32},
            files={"video": ("input.mp4", b"media", "video/mp4")},
        )
        assert response.status_code == 201
    assert not jobs._uploading
