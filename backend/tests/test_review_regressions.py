"""Reviewer regressions: real lifecycle/media and mocked model boundaries."""

import array
import asyncio
import copy
import json
import math
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from app import checkpoints, main, operations, state_store, storage
from app.api import studio
from app.config import settings
from app.pipeline import orchestrator as o, quality, transcribe, tts, windowed as w


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_artifacts_dir", tmp_path)
    monkeypatch.setattr(settings, "min_free_disk_mb", 0)
    monkeypatch.setattr(settings, "warmup_models", False)
    monkeypatch.setattr(settings, "recover_jobs", True)
    monkeypatch.setattr(o, "_shutting_down", False)
    for registry in (o._jobs, o._tasks, o._queues, o._cancel_signals, o._lanes):
        registry.clear()
    o._speech_lock = None
    yield
    for registry in (o._jobs, o._tasks, o._queues, o._cancel_signals, o._lanes):
        registry.clear()
    o._speech_lock = None


def job(identifier="a" * 32, owner="local"):
    state = o.JobState(identifier, owner_id=owner, lipsync_backend="none")
    queue = o.register_job(state)
    path = storage.job_dir(identifier) / "input.mp4"
    path.write_bytes(b"source")
    return state, path, queue


@pytest.mark.asyncio
async def test_shutdown_drains_native_and_recovers_running_and_queued(monkeypatch):
    started, finish = threading.Event(), threading.Event()

    def native():
        started.set()
        assert finish.wait(5)

    async def blocked(state, queue, *args):
        await o.blocking_call(native)

    monkeypatch.setattr(o, "_run_stage_audio", blocked)
    a, ap, _ = job()
    b, bp, _ = job("b" * 32)
    active = asyncio.create_task(o.run_pipeline(a, ap))
    while not started.is_set():
        await asyncio.sleep(0.001)
    waiting = asyncio.create_task(o.run_pipeline(b, bp))
    await asyncio.sleep(0)
    assert [a.status, b.status] == ["running", "queued"]
    stopping = asyncio.create_task(main._shutdown())
    await asyncio.sleep(0.02)
    assert not stopping.done()
    finish.set()
    await asyncio.wait_for(stopping, 5)
    await asyncio.gather(active, waiting)
    assert [storage.read_meta(s.job_id)["status"] for s in (a, b)] == ["queued", "queued"]
    recovered = []

    async def resume(state, path):
        recovered.append(state.job_id)

    monkeypatch.setattr(o, "run_pipeline", resume)
    await main._startup()
    await asyncio.sleep(0)
    assert set(recovered) == {a.job_id, b.job_id}
    await main._shutdown()


@pytest.mark.asyncio
async def test_cancel_before_dispatch_finishes_stream_and_remains_cancelled():
    state, path, queue = job()
    from app.api import jobs

    assert (await jobs.cancel(state.job_id))["status"] == "cancelled"
    await o.run_pipeline(state, path)
    assert o.pending_count() == 0 and not o._queues
    assert storage.read_meta(state.job_id)["status"] == "cancelled"
    events = [message["event"] async for _, message in queue.subscribe()]
    assert events[-1] == "stream_end"
    await main._shutdown()
    assert storage.read_meta(state.job_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_duplicate_dispatch_does_not_close_winner(monkeypatch):
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(o, "_run_stage_audio", blocked)
    state, path, queue = job()
    task = asyncio.create_task(o.run_pipeline(state, path))
    await entered.wait()
    await o.run_pipeline(state, path)
    assert o._tasks[state.job_id] is task and o.get_queue(state.job_id) is queue
    assert all(m["event"] != "stream_end" for _, m in queue.messages)
    o.cancel_job(state.job_id)
    task.cancel()
    await task
    assert storage.read_meta(state.job_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_failed_foreign_claim_detaches_without_overwriting_owner():
    state, path, queue = job()
    assert state_store.claim(state.job_id, "other-worker")
    await o.run_pipeline(state, path)
    assert not o._jobs and not o._queues and o.pending_count() == 0
    assert [m["event"] async for _, m in queue.subscribe()][-1] == "stream_end"
    assert state_store.heartbeat(state.job_id, "other-worker")
    assert storage.read_meta(state.job_id)["status"] == "queued"


@pytest.mark.parametrize("budget,fps", [(4096, None), (8192, 25)])
def test_52_second_portrait_clip_is_windowed_under_worker_budget(monkeypatch, budget, fps):
    monkeypatch.setattr(
        w.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=json.dumps(
                {"streams": [{"width": 1080, "height": 1920, "avg_frame_rate": "30/1"}]}
            )
        ),
    )
    monkeypatch.setattr(w, "duration", lambda p: 52)
    size, overlap = w.bounded_plan("video", "audio", budget, 8, 0.4, fps=fps)
    windows = list(w.windows(52, size, overlap))
    assert windows[0][0] == 0 and windows[-1][1] == 1300
    assert max(right - left for _, _, left, right in windows) * 1080 * 1920 * 3 < budget * 1024**2
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))


def test_adaptive_windows_include_context_even_with_small_budget(monkeypatch):
    monkeypatch.setattr(
        w.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=json.dumps(
                {"streams": [{"width": 3840, "height": 2160, "avg_frame_rate": "60/1"}]}
            )
        ),
    )
    monkeypatch.setattr(w, "duration", lambda p: 10)
    size, overlap = w.bounded_plan("v", "a", 64, 8, 0.4)
    assert all(
        (right - left) * 3840 * 2160 * 3 <= 64 * 1024**2 * 0.8
        for _, _, left, right in w.windows(10, size, overlap)
    )


def test_small_clip_keeps_single_render(tmp_path):
    video, audio = tmp_path / "v.mp4", tmp_path / "a.wav"
    w.run_ffmpeg(["-f", "lavfi", "-i", "color=size=160x120:duration=1", video])
    w.run_ffmpeg(["-f", "lavfi", "-i", "sine=duration=1", audio])
    assert w.bounded_plan(video, audio, 4096, 8, 0.4) is None
    assert w.bounded_plan(video, audio, 4096, 8, 0.4, force=True) == (8, 0.4)


@pytest.mark.parametrize("backend", ["f5tts", "indicf5"])
def test_first_segment_cache_hit_can_rewrite_and_regenerate(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(settings, "tts_segment_retries", 0)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    monkeypatch.setattr(
        tts, "_probe_duration", lambda p: 2 if p.read_text() == "Long phrase" else 0.5
    )
    monkeypatch.setattr(quality, "rewrite", lambda *a: "Short")
    calls = []

    def generate(text, language, reference, output, segments):
        calls.append(text)
        output.write_text(text)

    monkeypatch.setattr(tts, "_synthesize_" + backend + "_single_shot", generate)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"reference")
    segments = [dict(start=0, end=1, text="Long phrase")]
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
        tts._synthesize_per_segment(
            copy.deepcopy(segments), segments, ref, "en", tmp_path / "out.wav", backend=backend
        )
    assert calls == ["Long phrase"]
    tts._synthesize_per_segment(
        copy.deepcopy(segments),
        segments,
        ref,
        "en",
        tmp_path / "out.wav",
        backend=backend,
        options={"rewrite_overruns": True},
    )
    assert calls == ["Long phrase", "Short"]


def recognize(monkeypatch, words):
    model = SimpleNamespace(
        transcribe=lambda *a, **k: (
            [
                SimpleNamespace(
                    words=[
                        SimpleNamespace(word=text, start=start, end=end, probability=prob)
                        for text, start, end, prob in words
                    ]
                )
            ],
            None,
        )
    )
    monkeypatch.setattr(transcribe, "_get_model", lambda: model)


def signal(path, spans, seconds=1.7):
    rate = 24000
    samples = array.array(
        "h",
        (
            int(5000 * math.sin(2 * math.pi * 440 * i / rate))
            if any(start <= i / rate < end for start, end in spans)
            else 0
            for i in range(round(rate * seconds))
        ),
    )
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        out.writeframes(samples.tobytes())


def test_text_aligned_click_cleanup_fits_without_dropping_word(tmp_path, monkeypatch):
    path = tmp_path / "click.wav"
    signal(path, [(0, 0.04), (0.6, 1.4)])
    recognize(monkeypatch, [("Hola", 0.6, 1.4, 0.99)])
    assert tts._trim_tail_via_whisper(path, "es", "Hola") is True
    tts._trim_to_speech(path)
    assert 0.8 <= tts._probe_duration(path) <= 1


def test_short_first_word_is_retained(tmp_path, monkeypatch):
    path = tmp_path / "short.wav"
    signal(path, [(0, 0.04), (0.6, 1.4)])
    recognize(monkeypatch, [("Sí", 0, 0.04, 0.99), ("amigo", 0.6, 1.4, 0.99)])
    assert tts._trim_tail_via_whisper(path, "es", "Sí amigo") is True
    tts._trim_to_speech(path)
    assert tts._probe_duration(path) >= 1.4


def test_repeated_tail_removed_after_complete_expected_phrase(tmp_path, monkeypatch):
    path = tmp_path / "repeat.wav"
    signal(path, [(0, 0.3), (0.4, 0.7), (1, 1.4)])
    recognize(monkeypatch, [("no", 0, 0.3, 0.99), ("no", 0.4, 0.7, 0.99), ("no", 1, 1.4, 0.99)])
    assert tts._trim_tail_via_whisper(path, "es", "no no") is True
    assert 0.7 <= tts._probe_duration(path) <= 0.81


@pytest.mark.parametrize(
    "words,expected,result",
    [
        ([("hello", 0, 0.3, 0.99)], "hello world", False),
        ([("hello", 0, 0.3, 0.4), ("world", 0.4, 0.7, 0.4)], "hello world", None),
        ([("no", 0, 0.5, 0.99), ("no", 0.51, 0.8, 0.99)], "no", False),
    ],
)
def test_incomplete_or_uncertain_alignment_never_cuts_audio(
    tmp_path, monkeypatch, words, expected, result
):
    path = tmp_path / "speech.wav"
    signal(path, [(0, 1.4)])
    original = path.read_bytes()
    recognize(monkeypatch, words)
    assert tts._trim_tail_via_whisper(path, "en", expected) is result
    assert path.read_bytes() == original


def test_usage_uses_sql_totals_without_walking_history(monkeypatch):
    a, ap, _ = job()
    b, bp, _ = job("b" * 32, owner="someone-else")
    ap.write_bytes(b"x" * 1000)
    bp.write_bytes(b"x" * 2000)
    size = operations.refresh_usage(a.job_id)
    operations.refresh_usage(b.job_id)
    monkeypatch.setattr(Path, "rglob", lambda *a: pytest.fail("admission walked artifact history"))
    assert operations.usage("local") == {"bytes": size, "active": 1}
    assert operations.usage("someone-else")["bytes"] > size
    monkeypatch.setattr(settings, "max_user_storage_mb", 0.0001)
    with pytest.raises(HTTPException) as exc:
        operations.check_capacity("local")
    assert exc.value.status_code == 413
    a.status = "failed"
    storage.write_meta(a.job_id, a.to_dict())
    state_store.remove(a.job_id)
    assert operations.usage("local") == {"bytes": 0, "active": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("stages", [[], [o.StageResult(name="translate")]])
async def test_legacy_revision_returns_409_for_missing_prerequisites(stages):
    state, path, _ = job()
    state.status, state.stages = "failed", stages
    (path.parent / "translation.json").write_text(json.dumps({"segments": [{"text": "Hola"}]}))
    storage.write_meta(state.job_id, state.to_dict())
    with pytest.raises(HTTPException) as exc:
        await studio.revise(
            state.job_id, studio.Revision(translation=["Buenos días"]), BackgroundTasks()
        )
    assert exc.value.status_code == 409
    assert len([p for p in settings.job_artifacts_dir.iterdir() if p.is_dir()]) == 1


@pytest.mark.asyncio
async def test_legacy_reordered_stages_normalized_and_edits_preserved():
    state, path, _ = job()
    state.status = "failed"
    state.stages = [o.StageResult(name="translate"), o.StageResult(name="audio")]
    for stage in ("audio", "stabilize", "transcribe", "translate"):
        for name in checkpoints.ARTIFACTS[stage]:
            (path.parent / name).write_text(json.dumps({"segments": [{"text": "Hola"}]}))
        checkpoints.record(path.parent, stage)
    storage.write_meta(state.job_id, state.to_dict())
    result = await studio.revise(
        state.job_id, studio.Revision(translation=["Buenos días"]), BackgroundTasks()
    )
    revised = o.get_job(result["job_id"])
    assert [s.name for s in revised.stages] == o.STAGE_NAMES
    assert revised.stages[3].status == o.StageStatus.DONE
    assert revised.stages[4].status == o.StageStatus.PENDING
    assert (
        json.loads((storage.job_dir(revised.job_id) / "translation.json").read_text())["segments"][
            0
        ]["text"]
        == "Buenos días"
    )


def test_accounting_migrates_legacy_database(tmp_path):
    import sqlite3

    with sqlite3.connect(tmp_path / ".state.sqlite3") as db:
        db.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL, created TEXT NOT NULL, meta TEXT NOT NULL, lease TEXT, expires REAL, attempt INTEGER NOT NULL DEFAULT 0)"
        )
    state, path, _ = job()
    operations.reconcile_usage()
    assert operations.usage("local")["bytes"] >= len(path.read_bytes())
    assert state_store.read(state.job_id)["status"] == "queued"


@pytest.mark.asyncio
async def test_slow_capacity_check_does_not_block_health(monkeypatch):
    import httpx
    from app.api import jobs

    entered, finish = threading.Event(), threading.Event()

    def slow_check(owner):
        entered.set()
        assert finish.wait(5)

    async def no_work(*args):
        pass

    monkeypatch.setattr(operations, "check_capacity", slow_check)
    monkeypatch.setattr(jobs, "_kickoff", no_work)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/jobs",
                data={"target_language": "es"},
                files={"video": ("clip.mp4", b"media", "video/mp4")},
            )
        )
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            response = await asyncio.wait_for(client.get("/health"), 0.5)
            assert response.status_code == 200
        finally:
            finish.set()
        assert (await request).status_code == 201


def test_bad_speech_is_resynthesized_with_same_words_and_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_segment_retries", 1)
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda p: 0.5)
    validation = iter([False, True])
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: next(validation))
    calls = []

    def generate(text, ref, language, output, voice=None):
        calls.append((text, voice))
        output.write_bytes(b"take")

    monkeypatch.setattr(tts, "_xtts_to_file", generate)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"source")
    segments = [dict(start=0, end=1, text="Hola")]
    tts._synthesize_per_segment(
        segments, segments, ref, "es", tmp_path / "out.wav", options={"voice": "example"}
    )
    assert calls == [("Hola", "example"), ("Hola", "example")]


def test_speech_validation_failure_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_segment_retries", 1)
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda p: 0.5)
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: False)
    calls = []

    def generate(text, ref, language, output, **kwargs):
        calls.append(text)
        output.write_bytes(b"bad-take")

    monkeypatch.setattr(tts, "_xtts_to_file", generate)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"source")
    segments = [dict(start=0, end=1, text="Hola")]
    with pytest.raises(tts.TTSError, match="complete translation"):
        tts._synthesize_per_segment(segments, segments, ref, "es", tmp_path / "out.wav")
    assert calls == ["Hola", "Hola"]


def test_excessive_time_stretch_preserves_audio_without_error(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/test/rubberband")
    monkeypatch.setattr(tts, "_probe_duration", lambda p: 3)
    path = tmp_path / "speech.wav"
    path.write_bytes(b"speech")
    tts._maybe_time_stretch(path, 1)
    assert path.read_bytes() == b"speech"


def test_quiet_recognized_first_word_survives_segment_cleanup(tmp_path, monkeypatch):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"reference")
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    recognize(monkeypatch, [("Sí", 0, 0.05, 0.99), ("amigo", 0.6, 1.4, 0.99)])

    def generate(text, ref, language, output, **kwargs):
        rate = 24000
        samples = array.array(
            "h",
            (
                int((70 if i / rate < 0.05 else 5000) * math.sin(2 * math.pi * 440 * i / rate))
                if i / rate < 0.05 or 0.6 <= i / rate < 1.4
                else 0
                for i in range(round(rate * 1.7))
            ),
        )
        with wave.open(str(output), "wb") as out:
            out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            out.writeframes(samples.tobytes())

    monkeypatch.setattr(tts, "_xtts_to_file", generate)
    segments = [dict(start=0, end=2, text="Sí amigo")]
    output = tmp_path / "out.wav"
    tts._synthesize_per_segment(segments, segments, reference, "es", output)
    with wave.open(str(output), "rb") as audio:
        first_word = array.array("h", audio.readframes(1200))
    assert max(map(abs, first_word)) > 50


def test_periodic_accounting_skips_immutable_history(monkeypatch):
    active, _, _ = job()
    finished, _, _ = job("b" * 32)
    finished.status = "completed"
    storage.write_meta(finished.job_id, finished.to_dict())
    refreshed = []
    monkeypatch.setattr(operations, "refresh_usage", refreshed.append)
    operations.reconcile_usage(active_only=True)
    assert refreshed == [active.job_id]
