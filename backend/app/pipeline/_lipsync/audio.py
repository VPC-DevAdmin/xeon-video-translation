"""Mel-spectrogram preprocessing for Wav2Lip.

Vendored from https://github.com/Rudrabha/Wav2Lip (MIT-licensed code,
Rudrabha Mukhopadhyay et al., IIIT Hyderabad, 2020). Parameters must match
the training pipeline exactly or the model emits garbage — do not tune.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal

SAMPLE_RATE = 16_000
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
NUM_MELS = 80
FMIN = 55
FMAX = 7600
PREEMPHASIS = 0.97
REF_LEVEL_DB = 20.0
MIN_LEVEL_DB = -100.0

_mel_basis: np.ndarray | None = None


def _get_mel_basis() -> np.ndarray:
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX
        )
    return _mel_basis


def load_wav(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    return wav


def _preemphasis(wav: np.ndarray) -> np.ndarray:
    return signal.lfilter([1.0, -PREEMPHASIS], [1.0], wav)


def _stft(y: np.ndarray) -> np.ndarray:
    return librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE)


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    min_level = np.exp(MIN_LEVEL_DB / 20.0 * np.log(10.0))
    return 20.0 * np.log10(np.maximum(min_level, x))


def _normalize(s: np.ndarray) -> np.ndarray:
    return np.clip((s - MIN_LEVEL_DB) / -MIN_LEVEL_DB, 0.0, 1.0)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Return an (80, T) mel spectrogram matching Wav2Lip's expected input."""
    d = _stft(_preemphasis(wav))
    magnitude = np.abs(d)
    mel = np.dot(_get_mel_basis(), magnitude)
    s = _amp_to_db(mel) - REF_LEVEL_DB
    return _normalize(s)
