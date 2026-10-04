import importlib.util
from pathlib import Path
import wave
import pytest

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audio_contract', root / 'experiments/flashhead/audio_contract.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wav(path, rate, frames):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b'\0\0' * frames)
    return path


def test_native_soundtrack_has_same_timeline(tmp_path):
    cond = wav(tmp_path / 'condition.wav', 16000, 16000)
    master = wav(tmp_path / 'master.wav', 24000, 24000)
    result = module.inspect_audio_pair(cond, master)
    assert result['playback']['sample_rate'] == 24000
    assert result['conditioning']['sample_rate'] == 16000


def test_rejects_master_with_different_timing(tmp_path):
    cond = wav(tmp_path / 'condition.wav', 16000, 16000)
    master = wav(tmp_path / 'master.wav', 24000, 24500)
    with pytest.raises(ValueError, match='duration differ'):
        module.inspect_audio_pair(cond, master)


def test_accepts_single_sample_resampling_roundoff(tmp_path):
    cond = wav(tmp_path / 'condition.wav', 16000, 16001)
    master = wav(tmp_path / 'master.wav', 24000, 24001)
    module.inspect_audio_pair(cond, master)


def test_rejects_wrong_conditioning_sample_rate(tmp_path):
    cond = wav(tmp_path / 'condition.wav', 24000, 24000)
    with pytest.raises(ValueError, match='16000 Hz'):
        module.inspect_audio_pair(cond)
