import io
import pytest
from app import llm
from app.config import settings
from app.pipeline import windowed


def test_nvenc_failure_does_not_fall_back_or_poison_later_requests(monkeypatch):
    monkeypatch.setattr(settings, "device", "cuda")
    monkeypatch.setattr(settings, "video_encoder", "h264_nvenc")
    calls = []

    def run(args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("no encoder session")

    monkeypatch.setattr(windowed, "run_ffmpeg", run)
    with pytest.raises(RuntimeError, match="no encoder"):
        windowed.encode([], "one.mp4")
    windowed.encode([], "two.mp4")
    assert len(calls) == 2 and all("h264_nvenc" in c and "libx264" not in c for c in calls)


def test_llm_size_limits_reject_before_network(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_input_chars", 5)
    with pytest.raises(llm.LLMError, match="input limit"):
        llm.chat([{"content": "too much text"}])
    with pytest.raises(llm.LLMError, match="output token"):
        llm.chat([], max_tokens=settings.llm_max_output_tokens + 1)


def test_llm_full_queue_releases_after_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_concurrent", 1)
    monkeypatch.setattr(settings, "llm_max_pending", 0)
    llm._limits.cache_clear()
    with llm._admission():
        with pytest.raises(llm.LLMError, match="queue is full"):
            with llm._admission():
                pass
    with pytest.raises(ValueError):
        with llm._admission():
            raise ValueError()
    with llm._admission():
        pass


def test_stream_close_releases_admission(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "http://test/v1")
    monkeypatch.setattr(settings, "llm_max_concurrent", 1)
    monkeypatch.setattr(settings, "llm_max_pending", 0)
    monkeypatch.setattr(
        llm.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'),
    )
    monkeypatch.setattr(settings, "llm_interactive_max_concurrent", 1)
    monkeypatch.setattr(settings, "llm_interactive_max_pending", 0)
    llm._limits.cache_clear()
    stream = llm.stream([])
    assert next(stream) == "a"
    # Streams occupy the interactive lane only; batch translation keeps going.
    with pytest.raises(llm.LLMError, match="queue is full"):
        with llm._admission("interactive"):
            pass
    with llm._admission("batch"):
        pass
    stream.close()
    with llm._admission("interactive"):
        pass


def test_llm_rejects_large_or_malformed_response(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "http://test/v1")
    monkeypatch.setattr(settings, "llm_max_response_bytes", 10)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"x" * 11))
    with pytest.raises(llm.LLMError, match="size limit"):
        llm.chat([])
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"{"))
    with pytest.raises(llm.LLMError, match="request failed"):
        llm.chat([])


def test_window_gpu_decode_is_explicit(monkeypatch):
    from gpu_runtime import media

    monkeypatch.setattr(settings, "device", "cuda")
    monkeypatch.setattr(settings, "video_encoder", "h264_nvenc")
    monkeypatch.setattr(media, "probe_stream", lambda path: {"pix_fmt": "yuv420p"})
    calls = []
    monkeypatch.setattr(windowed, "run_ffmpeg", calls.append)
    windowed.encode(["-i", "input.mp4", "-vf", "fps=25"], "out.mp4")
    assert calls[0][:4] == ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    assert "hwdownload,format=nv12,fps=25" in calls[0]


@pytest.mark.parametrize("body,expected", [
    (b'{"data":[{"id":"served-model"}]}', True),
    (b'{"data":[{"id":"different-model"}]}', False),
    (b'{"data":null}', False),
    (b'[]', False),
    (b'{', False),
    (b'x' * 65537, False),
])
def test_llm_readiness_checks_model_and_bounds_response(monkeypatch, body, expected):
    monkeypatch.setattr(settings, "llm_base_url", "http://test/v1")
    monkeypatch.setattr(settings, "llm_model", "served-model")
    def open_request(request, timeout):
        assert request.full_url == "http://test/v1/models" and timeout == 3
        return io.BytesIO(body)
    monkeypatch.setattr(llm.urllib.request, "urlopen", open_request)
    assert llm.ready() is expected


def test_llm_readiness_unconfigured_or_offline(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "")
    assert not llm.ready()
    monkeypatch.setattr(settings, "llm_base_url", "http://test/v1")
    def offline(*args, **kwargs):
        raise TimeoutError("offline")
    monkeypatch.setattr(llm.urllib.request, "urlopen", offline)
    assert not llm.ready()


def test_failed_model_warmup_is_not_reported_done(monkeypatch):
    from app import main
    from app.pipeline import transcribe, tts
    monkeypatch.setattr(main, "_warmup_state", {"status": "running"})
    monkeypatch.setattr(settings, "translate_backend", "llm")
    monkeypatch.setattr(settings, "tts_backend", "xtts")
    monkeypatch.setattr(transcribe, "_get_model", lambda: None)
    def fail():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(tts, "_get_xtts", fail)
    main._warmup()
    assert main._warmup_state["status"] == "failed"
    monkeypatch.setattr(tts, "_get_xtts", lambda: None)
    main._warmup()
    assert main._warmup_state["status"] == "done"


def test_llm_readiness_ttl_reuses_last_answer(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "http://test/v1")
    monkeypatch.setattr(settings, "llm_model", "served-model")
    llm._ready_cache.clear()
    calls = []

    def open_request(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b'{"data":[{"id":"served-model"}]}')

    monkeypatch.setattr(llm.urllib.request, "urlopen", open_request)
    assert llm.ready(ttl=60) and llm.ready(ttl=60) and llm.ready(ttl=60)
    assert len(calls) == 1
    assert llm.ready() and len(calls) == 2  # ttl=0 always asks the server


@pytest.mark.parametrize("stream,expected", [
    ({"tags": {"rotate": "90"}}, -90.0),                      # clockwise tag -> ccw matrix
    ({"side_data_list": [{"rotation": -90}]}, -90.0),
    ({"tags": {"rotate": "90"}, "side_data_list": [{"rotation": 90}]}, 90.0),  # matrix wins
    ({}, 0.0),
])
def test_display_rotation_sign_convention(stream, expected):
    from gpu_runtime.media import display_rotation
    assert display_rotation(stream) == expected


def test_manual_transpose_clears_rotation_metadata(monkeypatch):
    from gpu_runtime import media
    monkeypatch.setattr(media, "probe_stream", lambda path: {"pix_fmt": "yuv420p", "side_data_list": [{"rotation": -90}]})
    flags, prefix = media.cuda_filter_input("in.mp4")
    assert flags[-2:] == ["-display_rotation", "0"]
    assert prefix == "hwdownload,format=nv12,transpose=clock"
