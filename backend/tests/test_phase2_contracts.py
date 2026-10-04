import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
import pytest
from app.config import settings
from app.pipeline import tts


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_missing_quality_honestly():
    compare = load_script("compare_benchmarks")
    reports = compare.summarize(
        [
            dict(
                tag="baseline",
                mode="fast",
                job={"status": "completed"},
                wall_seconds=t,
                quality_review={"missing_speech": None},
            )
            for t in (1, 2, 8)
        ]
    )
    assert reports[0]["median_seconds"] == 2
    assert reports[0]["p95_seconds"] is None
    assert "insufficient" in reports[0]["p95_status"]
    assert reports[0]["quality_reviewed"] == 0


def test_model_manifest_excludes_credentials_and_detects_changes(tmp_path):
    module = load_script("model_manifest")
    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "token.json").write_text("secret")
    first = module.manifest(tmp_path)
    assert list(first) == ["model.onnx"]
    (tmp_path / "model.onnx").write_bytes(b"changed")
    assert first != module.manifest(tmp_path)


def test_explicit_false_overrides_environment(monkeypatch):
    from app.options import from_request

    monkeypatch.setattr(settings, "enable_alignment", True)
    assert from_request("{}")["alignment"]
    assert not from_request('{"alignment":false}')["alignment"]


def test_tts_cache_reuses_unchanged_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 0.5)
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    calls = []

    def synth(text, ref, lang, path, **kwargs):
        calls.append(text)
        path.write_bytes(b"generated-audio")

    monkeypatch.setattr(tts, "_xtts_to_file", synth)
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"source")
    segments = [dict(start=0, end=1, text="Hola"), dict(start=1, end=2, text="Sí")]
    references = [dict(s) for s in segments]
    for _ in range(2):
        tts._synthesize_per_segment(segments, references, reference, "es", tmp_path / "out.wav")
    assert calls == ["Hola", "Sí"]
    segments[1]["text"] = "No"
    tts._synthesize_per_segment(segments, references, reference, "es", tmp_path / "out.wav")
    assert calls == ["Hola", "Sí", "No"]
    for cached in (tmp_path / "segments").glob("*.wav"):
        cached.write_bytes(b"corrupt")
    tts._synthesize_per_segment(segments, references, reference, "es", tmp_path / "out.wav")
    assert calls[-2:] == ["Hola", "No"]


def test_audio_service_alignment_and_diarization_contract(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "services/audio-quality/app/main.py"
    spec = importlib.util.spec_from_file_location("audio_service", path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    monkeypatch.setattr(service, "ROOT", tmp_path)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"language": "en", "segments": [dict(text="Hello", start=0, end=1)]})
    )
    fake = ModuleType("whisperx")
    fake.load_audio = lambda _: []
    fake.load_align_model = lambda **_: ("model", {})
    fake.align = lambda segments, *a, **k: {
        "segments": [{**segments[0], "words": [dict(word="Hello", start=0.1, end=0.8)]}]
    }
    diarize = ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = lambda **_: lambda _: []

    def assign(_, result):
        result["segments"][0]["speaker"] = "SPEAKER_00"
        return result

    diarize.assign_word_speakers = assign
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)
    result = service.analyze(
        service.Analyze(
            audio_path=str(audio), transcript_path=str(transcript), align=True, diarize=True
        )
    )
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["words"][0]["end"] == 0.8
    assert json.loads(transcript.read_text())["segments"][0].get("speaker") is None
    with pytest.raises(Exception):
        service.path(str(tmp_path.parent / "outside.wav"))
