"""Who speaks when, and on which face.

The pipeline assumed one person: one cloned voice, and the generated mouth on
the largest face. A two-presenter video breaks both. This module runs after
transcription and rewrites the transcript so every segment belongs to one
speaker who is tied to one face on screen:

1. **Voices.** The XTTS speaker encoder (already loaded for voice cloning)
   embeds 1.5 s windows of the audio every 0.5 s. K-means on those embeddings
   splits the voices; a second voice is accepted only when the split is clear
   (silhouette >= `MIN_SILHOUETTE`). Each word takes the majority voice of the
   windows around it.
2. **Faces.** The LatentSync service tracks every person in the clip and
   reports each one's mouth opening per frame (POST /faces/identities). A voice
   belongs to the face whose mouth moves most while that voice speaks.
3. **Segments.** Words are regrouped into segments that never cross a speaker
   change, end at sentence ends once they are a few seconds long, and are
   capped at `MAX_SEGMENT_SECONDS`. Whisper's 30 s windows otherwise give
   segments that mix both people.

The transcript gains `speakers` ([{id, face_identity, seconds, words}]) and a
`diarization` record; each segment gains `speaker`. TTS already picks a voice
reference per speaker from the transcript, and the span renderer puts each
speaker's lines on that speaker's face (streaming.py).

On a single-speaker clip with one person visible nothing changes. With one
speaker and several people visible, the speaker is still tied to the face that
talks, which fixes renders on the wrong (larger) face.
"""

from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.5
MIN_SILHOUETTE = 0.25
MIN_PRESENCE = 0.15
MAX_SPEAKERS = 4
SENTENCE_MIN_SECONDS = 3.0
MAX_SEGMENT_SECONDS = 15.0
FPS = 25


# ----------------------------------------------------------------- voices
def embed_windows(audio_path: Path, window: float = WINDOW_SECONDS, hop: float = HOP_SECONDS,
                  min_rms: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Window centres (s) and L2-normalised speaker embeddings (XTTS encoder)."""
    import soundfile as sf
    import torch

    from . import tts

    wav, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    model = tts._get_xtts().synthesizer.tts_model
    device = next(model.parameters()).device
    times, embeddings = [], []
    total = len(wav) / rate
    start = 0.0
    with torch.no_grad():
        while start + window <= total:
            chunk = wav[int(start * rate):int((start + window) * rate)]
            if float(np.sqrt(np.mean(chunk**2))) >= min_rms:
                e = model.get_speaker_embedding(torch.from_numpy(chunk).unsqueeze(0).to(device), rate)
                embeddings.append(e.squeeze().float().cpu().numpy())
                times.append(start + window / 2)
            start += hop
    if not embeddings:
        return np.zeros(0), np.zeros((0, 512), np.float32)
    e = np.stack(embeddings)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-9)
    return np.asarray(times), e


def cluster_voices(embeddings: np.ndarray, max_k: int) -> tuple[np.ndarray, int, float]:
    """(labels, k, silhouette). k > 1 only when its silhouette clears MIN_SILHOUETTE;
    of the k that do, the best silhouette wins."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    best = (np.zeros(n, dtype=int), 1, 0.0)
    if n < 10 or max_k < 2:
        return best
    for k in range(2, min(max_k, MAX_SPEAKERS) + 1):
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(embeddings)
        if min(np.bincount(labels)) < max(3, 0.03 * n):
            continue
        score = float(silhouette_score(embeddings, labels, metric="cosine"))
        if score >= MIN_SILHOUETTE and score > best[2]:
            best = (labels, k, score)
    return best


def word_labels(words: list[dict], times: np.ndarray, labels: np.ndarray, pad: float = 0.25) -> list[int]:
    """Majority window label around each word; nearest window when none overlap."""
    out = []
    for w in words:
        lo, hi = float(w["start"]) - pad, float(w["end"]) + pad
        inside = labels[(times >= lo) & (times <= hi)]
        if len(inside):
            out.append(int(np.bincount(inside).argmax()))
        elif len(times):
            out.append(int(labels[np.argmin(np.abs(times - (lo + hi) / 2))]))
        else:
            out.append(0)
    return out


def smooth_labels(words: list[dict], labels: list[int], max_seconds: float = 0.4) -> list[int]:
    """A single short word between two words of the same other speaker is a
    misread window, not a turn."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        w = words[i]
        if out[i - 1] == out[i + 1] != out[i] and float(w["end"]) - float(w["start"]) <= max_seconds:
            out[i] = out[i - 1]
    return out


def snap_to_sentences(words: list[dict], labels: list[int], reach: int = 2) -> list[int]:
    """A turn rarely starts mid-sentence. When a change of speaker lands within
    `reach` words after a sentence start, those words belong to the new
    speaker (the first word of a sentence is often too short for the voice
    windows: "...with VectorPath. | I invited Chris" put "I" on the wrong side)."""
    out = list(labels)
    ends = [bool(_SENTENCE_END.search(str(w["text"]).strip())) for w in words]
    for c in range(1, len(out)):
        if out[c] == out[c - 1]:
            continue
        for j in range(max(1, c - reach), c):
            if ends[j - 1] and not any(ends[j:c]):
                for i in range(j, c):
                    out[i] = out[c]
                break
    return out


# ------------------------------------------------------------------ faces
def mouth_activity(mouth: list[list[float | None]]) -> np.ndarray:
    """(people, frames) movement of each mouth: |frame-to-frame change| of the
    opening, 5-frame average, standardised per person. Absent frames count 0."""
    rows = []
    for series in mouth:
        x = np.array([np.nan if v is None else float(v) for v in series], dtype=np.float64)
        d = np.abs(np.diff(x, prepend=x[:1]))
        d[~np.isfinite(d)] = 0.0
        d = np.convolve(d, np.ones(5) / 5, mode="same")
        seen = np.isfinite(x)
        mu, sd = (d[seen].mean(), d[seen].std()) if seen.any() else (0.0, 1.0)
        z = (d - mu) / (sd if sd > 1e-9 else 1.0)
        z[~seen] = 0.0
        rows.append(z)
    return np.stack(rows) if rows else np.zeros((0, 0))


def voice_face_scores(words: list[dict], labels: list[int], k: int, activity: np.ndarray, fps: int = FPS) -> np.ndarray:
    """(voices, people) mean mouth activity of each person while each voice speaks."""
    people, frames = activity.shape
    scores = np.zeros((k, people))
    counts = np.zeros(k)
    for w, c in zip(words, labels):
        a, b = int(float(w["start"]) * fps), int(np.ceil(float(w["end"]) * fps))
        a, b = max(0, a), min(frames, max(b, a + 1))
        if b <= a:
            continue
        scores[c] += activity[:, a:b].sum(axis=1)
        counts[c] += b - a
    return scores / counts.clip(min=1)[:, None]


def match_voices(scores: np.ndarray) -> list[int]:
    """Voice -> person, one-to-one, maximising total mouth activity."""
    k, people = scores.shape
    if people == 0:
        return [-1] * k
    if people < k:
        # More voices than faces: the best face for each, possibly shared.
        return [int(np.argmax(scores[c])) for c in range(k)]
    best, choice = -np.inf, None
    for perm in itertools.permutations(range(people), k):
        total = sum(scores[c, perm[c]] for c in range(k))
        if total > best:
            best, choice = total, perm
    return list(choice)


# --------------------------------------------------------------- segments
_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
_CLAUSE_END = re.compile(r"[,;:]['\")\]]*$")


def _segment(words: list[dict], speaker: str) -> dict:
    return {"start": float(words[0]["start"]), "end": float(words[-1]["end"]),
            "text": " ".join(str(w["text"]).strip() for w in words).strip(),
            "words": [dict(w) for w in words], "speaker": speaker}


def resegment(words: list[dict], speakers: list[str], sentence_min: float = SENTENCE_MIN_SECONDS,
              max_seconds: float = MAX_SEGMENT_SECONDS) -> list[dict]:
    """Group words into segments that never cross a speaker change, end at a
    sentence end once `sentence_min` long, and split before `max_seconds` at the
    last clause or sentence end (else at the widest pause)."""
    out: list[dict] = []
    current: list[dict] = []
    who = None

    def flush(upto: int | None = None) -> None:
        nonlocal current
        part, rest = (current, []) if upto is None else (current[:upto], current[upto:])
        if part:
            out.append(_segment(part, who))
        current = rest

    for w, s in zip(words, speakers):
        if current and s != who:
            flush()
        who = s
        current.append(w)
        length = float(current[-1]["end"]) - float(current[0]["start"])
        text = str(w["text"]).strip()
        if length >= sentence_min and _SENTENCE_END.search(text):
            flush()
        elif length >= max_seconds:
            cut = None
            for i in range(len(current) - 1, 0, -1):
                if _SENTENCE_END.search(str(current[i - 1]["text"]).strip()) or _CLAUSE_END.search(str(current[i - 1]["text"]).strip()):
                    if float(current[i - 1]["end"]) - float(current[0]["start"]) >= 2.0:
                        cut = i
                        break
            if cut is None:
                gaps = [float(current[i]["start"]) - float(current[i - 1]["end"]) for i in range(1, len(current))]
                cut = int(np.argmax(gaps)) + 1 if gaps else len(current)
            flush(cut)
    if current:
        flush()
    return out


# ------------------------------------------------------------------ main
def analyze(audio_path: Path, transcript: dict, video_path: Path, people: dict | None) -> dict:
    """Return the transcript rewritten per speaker (see module docstring).
    `people` is the /faces/identities response, or None when unavailable."""
    segments = transcript.get("segments") or []
    words = [w for s in segments for w in (s.get("words") or [])]
    if not segments or not words or any(not s.get("words") for s in segments if s.get("text", "").strip()):
        log.info("speakers: no word timings; transcript left as is")
        return transcript

    visible = []
    if people and people.get("people"):
        visible = [p["id"] for p in people["people"] if p.get("presence", 0) >= MIN_PRESENCE]
    times, embeddings = embed_windows(audio_path)
    labels, k, silhouette = cluster_voices(embeddings, max(2, len(visible)) if visible else 2)
    per_word = (snap_to_sentences(words, smooth_labels(words, word_labels(words, times, labels)))
                if k > 1 else [0] * len(words))

    # Name voices S0, S1, ... in order of first appearance.
    order = list(dict.fromkeys(per_word))
    rename = {c: i for i, c in enumerate(order)}
    per_word = [rename[c] for c in per_word]
    k = len(order)

    face_of = [None] * k
    scores = None
    if visible and people.get("mouth"):
        activity = mouth_activity([people["mouth"][i] for i in visible])
        scores = voice_face_scores(words, per_word, k, activity)
        matched = match_voices(scores)
        face_of = [visible[m] if m >= 0 else None for m in matched]

    names = [f"S{c}" for c in per_word]
    new_segments = resegment(words, names)
    seconds = {f"S{c}": 0.0 for c in range(k)}
    counts = {f"S{c}": 0 for c in range(k)}
    for w, n in zip(words, names):
        seconds[n] += float(w["end"]) - float(w["start"])
        counts[n] += 1
    result = dict(transcript)
    result["segments"] = new_segments
    result["speakers"] = [{"id": f"S{c}", "face_identity": face_of[c], "seconds": round(seconds[f"S{c}"], 1),
                           "words": counts[f"S{c}"]} for c in range(k)]
    result["diarization"] = {
        "method": "xtts speaker encoder k-means + face mouth activity",
        "voices": k, "silhouette": round(silhouette, 3), "people_visible": len(visible),
        "people": (people or {}).get("people"),
        "voice_face_scores": None if scores is None else np.round(scores, 3).tolist(),
        "segments_before": len(segments), "segments_after": len(new_segments),
    }
    log.info("speakers: %d voice(s) (silhouette %.2f), %d people visible, faces %s, %d -> %d segments",
             k, silhouette, len(visible), face_of, len(segments), len(new_segments))
    return result


def face_map(transcript: dict) -> dict[str, int]:
    """speaker id -> face identity, for the speakers that have one."""
    return {s["id"]: int(s["face_identity"]) for s in transcript.get("speakers") or []
            if s.get("face_identity") is not None}


def needs_span_render(transcript: dict) -> bool:
    """True when renders must target specific faces: someone is tied to a face
    and more than one person is visible."""
    d = transcript.get("diarization") or {}
    return bool(face_map(transcript)) and int(d.get("people_visible") or 0) >= 2
