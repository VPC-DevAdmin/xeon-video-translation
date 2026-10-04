import shutil
import threading
import pytest
from app.pipeline import windowed as w
from app.pipeline.enhancements import remix


def fixture_media(tmp_path, video_seconds=2.4, audio_seconds=3.2):
    video = tmp_path / "input.mp4"
    audio = tmp_path / "audio.wav"
    w.run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=25:duration={video_seconds}",
            "-c:v",
            "libx264",
            video,
        ]
    )
    w.run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds}", audio])
    return video, audio


def test_real_window_render_resume_corruption_and_tail(tmp_path):
    video, audio = fixture_media(tmp_path)
    calls = []
    output = tmp_path / "result.mp4"

    def renderer(source, sound, result):
        calls.append(w.duration(source))
        shutil.copyfile(source, result)
        assert abs(w.duration(source) - w.duration(sound)) < 0.05

    w.render(video, audio, output, renderer, size=1, overlap=0.2)
    assert len(calls) == 4 and max(calls) <= 1.41
    assert abs(w.duration(output) - 3.2) < 0.05
    w.render(video, audio, output, renderer, size=1, overlap=0.2)
    assert len(calls) == 4
    (tmp_path / "render-windows/part-000001.mp4").write_bytes(b"corrupt")
    w.render(video, audio, output, renderer, size=1, overlap=0.2)
    assert len(calls) == 5


def test_window_cancel_drains_then_resume(tmp_path):
    video, audio = fixture_media(tmp_path)
    cancel = threading.Event()
    calls = []

    def renderer(source, sound, result):
        calls.append(source)
        shutil.copyfile(source, result)
        cancel.set()

    with pytest.raises(w.RenderCancelled):
        w.render(video, audio, tmp_path / "out.mp4", renderer, size=1, cancel=cancel)
    assert len(calls) == 1 and not (tmp_path / "out.mp4").exists()


def test_background_remix_real_ffmpeg(tmp_path):
    _, voice = fixture_media(tmp_path, audio_seconds=1)
    background = tmp_path / "bed.wav"
    output = tmp_path / "mix.wav"
    w.run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=220:duration=2", background])
    remix(voice, background, output, 0.35)
    assert abs(w.duration(output) - 2) < 0.05


def test_window_offset_passed_only_to_renderers_that_accept_it():
    from app.pipeline import windowed as w

    seen = []
    w._call_renderer(lambda s, a, r, window_offset_frames=None: seen.append(window_offset_frames), "s", "a", "r", 200)
    w._call_renderer(lambda s, a, r, **kw: seen.append(kw["window_offset_frames"]), "s", "a", "r", 400)
    w._call_renderer(lambda s, a, r: seen.append("plain"), "s", "a", "r", 600)
    assert seen == [200, 400, "plain"]


def test_next_window_is_cut_and_prepared_while_current_renders(tmp_path):
    """The preparer for window N+1 must be invoked (with its 25 fps offset and an
    existing cut file) before the renderer for window N returns."""
    import subprocess, threading, time
    from pathlib import Path
    from app.pipeline import windowed as w

    video = tmp_path / "v.mp4"; audio = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=25", "-t", "3", "-pix_fmt", "yuv420p", str(video)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000", "-t", "3", str(audio)], check=True)
    events = []
    lock = threading.Lock()

    def renderer(source, sound, result):
        with lock:
            events.append(("render", Path(source).name))
        time.sleep(0.3)  # give the prefetch thread time to run
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c", "copy", str(result)], check=True)

    def preparer(source, sound, offset):
        assert Path(source).exists() and Path(sound).exists()
        with lock:
            events.append(("prepare", Path(source).name, offset))

    w.render(video, audio, tmp_path / "out.mp4", renderer, size=1, overlap=0.2, preparer=preparer)
    prepared = [e for e in events if e[0] == "prepare"]
    rendered = [e for e in events if e[0] == "render"]
    assert len(rendered) >= 3 and len(prepared) == len(rendered) - 1
    offsets = [e[2] for e in prepared]
    assert offsets == sorted(offsets) and offsets[0] > 0
    # each prepare for window k happened before render of window k started
    for k, (_, name, _) in enumerate(prepared, start=1):
        assert events.index(("prepare", name, offsets[k - 1])) < events.index(("render", name))
    assert not list((tmp_path / "render-windows").glob("source-*.mp4"))
