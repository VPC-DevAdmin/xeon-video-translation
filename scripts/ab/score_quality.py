#!/usr/bin/env python3
"""Objective quality metrics for a rendered talking-head video (runs in the LatentSync container).

Metrics per video:
  identity      ArcFace (insightface w600k_r50) cosine between each output frame and the
                reference portrait, and against the source footage when given.
  sharpness     Laplacian variance of the 256 px face crop and of its mouth region.
  temporal      mean |frame difference| of consecutive face crops; flicker = std of the
                high-passed mouth-region brightness.
  sync_proxy    peak correlation (and its lag) between mouth openness (MediaPipe lips 13/14,
                normalised by face height) and the audio RMS envelope at the video rate.
  syncnet       LatentSync's StableSyncNet cosine between 16-frame lower-half face windows
                and their mel windows, at the true offset and at shifted offsets.
  av            stream durations and frame counts from ffprobe.
Also writes a contact sheet (8 frames, full frame + face crop) for visual review.

Usage:
  score_quality.py --video OUT.mp4 --audio reply-16k.wav --reference portrait.png \
      [--reference-video persona.mp4] --label NAME --out /jobs/ab/out/scores
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
APP = Path(os.environ.get("LATENTSYNC_APP", "/app/app"))
sys.path.insert(0, str(APP))

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--audio", required=True, help="16 kHz mono wav on the video's timeline")
parser.add_argument("--reference", required=True, help="portrait image of the person")
parser.add_argument("--reference-video", help="source footage of the person (persona clip)")
parser.add_argument("--label", required=True)
parser.add_argument("--out", default="/jobs/ab/out/scores")
parser.add_argument("--max-frames", type=int, default=2000)
parser.add_argument("--skip-syncnet", action="store_true")
args = parser.parse_args()
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,nb_frames,duration:format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def read_frames(path, limit):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


frames, fps = read_frames(args.video, args.max_frames)
if not frames:
    raise SystemExit(f"no frames decoded from {args.video}")
height, width = frames[0].shape[:2]

# ---------------------------------------------------------------- faces
from insightface.app import FaceAnalysis  # noqa: E402

root = os.environ.get("INSIGHTFACE_ROOT", "/models/insightface")
analyzer = FaceAnalysis(name="buffalo_l", root=root,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        allowed_modules=["detection", "landmark_2d_106", "recognition"])
analyzer.prepare(ctx_id=0, det_size=(640, 640))


def largest_face(image):
    faces = analyzer.get(image)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def embed_image(image):
    face = largest_face(image)
    return None if face is None else face.normed_embedding


reference_image = cv2.imread(args.reference)
reference_embedding = embed_image(reference_image)
if reference_embedding is None:
    raise SystemExit("no face in the reference portrait")
source_embedding = None
if args.reference_video:
    src_frames, _ = read_frames(args.reference_video, 400)
    embeddings = [e for e in (embed_image(f) for f in src_frames[::40]) if e is not None]
    if embeddings:
        source_embedding = np.mean(embeddings, axis=0)
        source_embedding /= np.linalg.norm(source_embedding)

boxes, identity_portrait, identity_source, crops, detected = [], [], [], [], 0
for frame in frames:
    face = largest_face(frame)
    if face is None:
        boxes.append(None); crops.append(None); continue
    detected += 1
    x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(width, x2), min(height, y2)
    boxes.append((x1, y1, x2, y2))
    identity_portrait.append(float(np.dot(face.normed_embedding, reference_embedding)))
    if source_embedding is not None:
        identity_source.append(float(np.dot(face.normed_embedding, source_embedding)))
    crop = frame[y1:y2, x1:x2]
    crops.append(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA) if crop.size else None)

valid = [c for c in crops if c is not None]
gray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in valid]
sharp_face = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in gray]
sharp_mouth = [float(cv2.Laplacian(g[160:240, 64:192], cv2.CV_64F).var()) for g in gray]
diffs = [float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) for a, b in zip(gray[:-1], gray[1:])]
mouth_mean = np.array([float(g[160:240, 64:192].mean()) for g in gray])
if len(mouth_mean) > 5:
    kernel = np.ones(5) / 5
    smooth = np.convolve(mouth_mean, kernel, mode="same")
    flicker = float(np.std((mouth_mean - smooth)[2:-2]))
else:
    flicker = None

# ------------------------------------------------------- mouth openness (MediaPipe)
openness = np.full(len(frames), np.nan)
try:
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as mesh:
        for i, frame in enumerate(frames):
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                continue
            pts = np.array([(v.x * width, v.y * height) for v in result.multi_face_landmarks[0].landmark])
            face_h = np.linalg.norm(pts[10] - pts[152])
            if face_h > 0:
                openness[i] = np.linalg.norm(pts[13] - pts[14]) / face_h
    mediapipe_ok = True
except Exception as exc:  # pragma: no cover - environment dependent
    mediapipe_ok = False
    openness_error = f"{type(exc).__name__}: {exc}"

# ------------------------------------------------------------- audio envelope
import soundfile as sf  # noqa: E402

wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
if wav.ndim > 1:
    wav = wav.mean(axis=1)
hop = sr / fps
envelope = np.array([np.sqrt(np.mean(wav[int(i * hop): int((i + 1) * hop)] ** 2) + 1e-12)
                     for i in range(min(len(frames), int(len(wav) / hop)))])
sync_proxy = None
if mediapipe_ok and np.isfinite(openness).sum() > 20:
    n = min(len(envelope), len(openness))
    o = np.nan_to_num(openness[:n], nan=np.nanmean(openness[:n]))
    e = envelope[:n]
    o = (o - o.mean()) / (o.std() + 1e-9)
    e = (e - e.mean()) / (e.std() + 1e-9)
    best = None
    for lag in range(-8, 9):            # positive lag: mouth lags audio by `lag` frames
        if lag >= 0:
            a, b = o[lag:], e[: n - lag]
        else:
            a, b = o[: n + lag], e[-lag:]
        r = float(np.mean(a * b))
        if best is None or r > best[0]:
            best = (r, lag)
    sync_proxy = {"peak_correlation": round(best[0], 3), "lag_frames": best[1],
                  "lag_ms": round(best[1] * 1000 / fps, 1), "frames_with_landmarks": int(np.isfinite(openness).sum())}

# ------------------------------------------------------------------ SyncNet
syncnet = None
if not args.skip_syncnet:
    try:
        import torch
        from omegaconf import OmegaConf
        from latentsync.models.stable_syncnet import StableSyncNet
        from latentsync.utils.image_processor import ImageProcessor

        os.chdir(str(APP))  # latentsync.utils.audio loads configs/audio.yaml and ImageProcessor its mask relative to the app root
        from latentsync.utils import audio as ls_audio

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = OmegaConf.load(str(HERE / "configs" / "syncnet_16_pixel_attn.yaml"))
        model = StableSyncNet(OmegaConf.to_container(config.model)).to(device)
        ckpt = torch.load(os.environ.get("STABLE_SYNCNET", "/models/latentsync/stable_syncnet.pt"),
                          map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        processor = ImageProcessor(256, device=str(device))
        aligned = []
        for frame in frames:
            try:
                face, _, _ = processor.affine_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                aligned.append(torch.as_tensor(face))   # (3, 256, 256) uint8
            except Exception:
                aligned.append(None)
        mel = ls_audio.melspectrogram(wav.astype(np.float32) if sr == 16000 else
                                      ls_audio.load_wav(args.audio, 16000)).T  # (T, 80), 80 Hz
        mel_t = torch.from_numpy(mel.astype(np.float32))
        steps = 16
        windows = [s for s in range(0, len(aligned) - steps + 1, steps)
                   if all(a is not None for a in aligned[s:s + steps])
                   and int(80 * (s + steps) / 25) + 52 <= len(mel_t)]

        def visual(s):
            clip = torch.stack(aligned[s:s + steps]).float() / 255.0      # (16, 3, 256, 256)
            clip = (clip - 0.5) / 0.5
            lower = clip[:, :, 128:, :]                                   # lower half (16, 3, 128, 256)
            return lower.reshape(1, -1, 128, 256)

        def audio_window(start_frame):
            start = int(80.0 * start_frame / 25)
            chunk = mel_t[start:start + 52]
            if len(chunk) < 52:
                chunk = torch.cat([chunk, chunk[-1:].repeat(52 - len(chunk), 1)])
            return chunk.T.unsqueeze(0).unsqueeze(0)                      # (1, 1, 80, 52)

        offsets = list(range(-15, 16))
        scores = {o: [] for o in offsets}
        with torch.no_grad():
            for s in windows:
                v = model.visual_encoder(visual(s).to(device)).flatten(1)
                v = torch.nn.functional.normalize(v, dim=1)
                for o in offsets:
                    start_frame = s + o
                    if start_frame < 0 or int(80 * (start_frame + steps) / 25) + 52 > len(mel_t):
                        continue
                    a = model.audio_encoder(audio_window(start_frame).to(device)).flatten(1)
                    a = torch.nn.functional.normalize(a, dim=1)
                    scores[o].append(float((v * a).sum()))
        means = {o: float(np.mean(v)) for o, v in scores.items() if v}
        if means:
            best_offset = max(means, key=means.get)
            others = [means[o] for o in means if abs(o) >= 3]
            syncnet = {"windows": len(windows), "cosine_at_zero": round(means.get(0, float("nan")), 4),
                       "best_offset_frames": best_offset, "cosine_at_best": round(means[best_offset], 4),
                       "margin_over_shifted": round(means.get(0, 0) - float(np.median(others)), 4) if others else None,
                       "per_offset": {str(o): round(v, 4) for o, v in sorted(means.items())},
                       "state_dict_missing": len(missing), "state_dict_unexpected": len(unexpected)}
    except Exception as exc:  # report, do not fail the run
        syncnet = {"error": f"{type(exc).__name__}: {exc}"}

# ------------------------------------------------------------ contact sheet
picks = np.linspace(0, len(frames) - 1, 8).astype(int)
tiles = []
for i in picks:
    full = cv2.resize(frames[i], (int(256 * width / height), 256)) if height >= width else cv2.resize(frames[i], (256, int(256 * height / width)))
    canvas = np.zeros((256, 256 + 256, 3), np.uint8)
    fh, fw = full.shape[:2]
    canvas[:fh, :fw] = full[:256, :256]
    if crops[i] is not None:
        canvas[:, 256:] = crops[i]
    cv2.putText(canvas, f"{i / fps:5.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(canvas)
sheet = np.concatenate([np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)], axis=0)
sheet_path = out_dir / f"{args.label}.sheet.png"
cv2.imwrite(str(sheet_path), sheet)
# Mouth strip: every 4th frame over the first 4 s, mouth region only, to show chunk seams.
strip = [c[150:250, 48:208] for c in crops[: int(4 * fps): 4] if c is not None]
if strip:
    cv2.imwrite(str(out_dir / f"{args.label}.mouths.png"), np.concatenate(strip, axis=1))


def stats(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(arr.mean()), 4), "min": round(float(arr.min()), 4),
            "p05": round(float(np.percentile(arr, 5)), 4), "p95": round(float(np.percentile(arr, 95)), 4),
            "std": round(float(arr.std()), 4)}


info = probe(args.video)
report = {
    "label": args.label, "video": args.video, "audio": args.audio, "reference": args.reference,
    "frames": len(frames), "fps": fps, "width": width, "height": height,
    "faces_detected": detected, "face_box_mean_px": [round(float(np.mean([b[2] - b[0] for b in boxes if b])), 1),
                                                    round(float(np.mean([b[3] - b[1] for b in boxes if b])), 1)] if detected else None,
    "identity_vs_portrait": stats(identity_portrait),
    "identity_vs_source_video": stats(identity_source) if identity_source else None,
    "identity_drift_first_to_last_quarter": (round(float(np.mean(identity_portrait[-len(identity_portrait) // 4:]) -
                                                        np.mean(identity_portrait[: len(identity_portrait) // 4])), 4)
                                             if len(identity_portrait) >= 8 else None),
    "sharpness_face_laplacian_var": stats(sharp_face),
    "sharpness_mouth_laplacian_var": stats(sharp_mouth),
    "temporal_face_mean_abs_diff": stats(diffs),
    "mouth_flicker_std": round(flicker, 4) if flicker is not None else None,
    "mouth_openness": stats([float(v) for v in openness if np.isfinite(v)]) if mediapipe_ok else {"error": openness_error},
    "sync_proxy": sync_proxy,
    "syncnet": syncnet,
    "av": info,
    "contact_sheet": str(sheet_path),
}
(out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=1))
np.save(out_dir / f"{args.label}.openness.npy", openness)
print(json.dumps({k: report[k] for k in ("label", "frames", "faces_detected", "identity_vs_portrait",
                                          "sharpness_mouth_laplacian_var", "temporal_face_mean_abs_diff",
                                          "mouth_flicker_std", "sync_proxy", "syncnet")}, indent=1))
