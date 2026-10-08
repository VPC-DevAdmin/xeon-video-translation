"""Single-GPU streaming smoke: native SDPA, fixed seed, padded tail, NVENC artifact."""

import argparse, hashlib, json, time, wave, subprocess
from pathlib import Path
from collections import deque
import numpy as np
import torch
import flash_head.src.pipeline.flash_head_pipeline as implementation
from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

parser = argparse.ArgumentParser()
parser.add_argument("--compile", action="store_true")
parser.add_argument("--model", choices=["lite", "pro"], default="lite")
parser.add_argument("--image", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--model-root", type=Path, default=Path("/experiment-models"))
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this experiment")
args.output.mkdir(parents=True, exist_ok=True)
implementation.COMPILE_MODEL = args.compile
implementation.COMPILE_VAE = args.compile
name = args.model + ("-compiled" if args.compile else "-eager")
out = args.output
torch.cuda.reset_peak_memory_stats()
load = time.perf_counter()
pipeline = get_pipeline(
    1,
    str(args.model_root / "SoulX-FlashHead-1_3B"),
    args.model,
    str(args.model_root / "wav2vec2-base-960h"),
)
get_base_data(pipeline, args.image, 42, False)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - load
params = get_infer_params()
sr = params["sample_rate"]
fps = params["tgt_fps"]
motion = params["motion_frames_num"]
count = params["frame_num"]
slice_samples = (count - motion) * sr // fps
with wave.open(args.audio) as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, sr)
    audio = (
        np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32)
        / 32768
    )
frames_expected = int(np.ceil(len(audio) * fps / sr))
audio = np.pad(audio, (0, (-len(audio)) % slice_samples))
cached = deque(
    np.zeros(sr * params["cached_audio_duration"], np.float32),
    maxlen=sr * params["cached_audio_duration"],
)
end = params["cached_audio_duration"] * fps
timings = []
frames = []
for index, part in enumerate(audio.reshape(-1, slice_samples)):
    torch.cuda.synchronize()
    start = time.perf_counter()
    cached.extend(part)
    embedding = get_audio_embedding(
        pipeline, np.asarray(cached, dtype=np.float32), end - count, end
    )
    video = run_pipeline(pipeline, embedding)[motion:].to(torch.uint8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    timings.append(
        {
            "chunk": index,
            "frames": len(video),
            "seconds": elapsed,
            "fps": len(video) / elapsed,
        }
    )
    frames.append(video.cpu().numpy())
    print(json.dumps(timings[-1]), flush=True)
video = np.concatenate(frames)[:frames_expected]
h, w = video.shape[1:3]
proc = subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        args.audio,
        "-vf",
        "drawtext=text=AI-generated:x=8:y=h-28:fontsize=18:fontcolor=white",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "18",
        "-c:a",
        "aac",
        "-shortest",
        str(out / f"{name}.mp4"),
    ],
    input=video.tobytes(),
    capture_output=True,
)
if proc.returncode:
    raise RuntimeError(proc.stderr.decode()[-1000:])
report = {
    "source_revision": "9bc03de06bb0de82cd6bc477804512ae06144bf2",
    "model_revisions": json.loads(
        (Path("/experiment") / "model-revisions.json").read_text()
    ),
    "input_sha256": {
        str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in [args.image, args.audio]
    },
    "seed": 42,
    "implementation": "FlashHead " + args.model,
    "device": torch.cuda.get_device_name(),
    "capability": torch.cuda.get_device_capability(),
    "attention": "torch SDPA (no FlashAttention or Sage installed)",
    "compile": args.compile,
    "load_and_prepare_seconds": load_seconds,
    "chunks": timings,
    "frames": len(video),
    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
    "quality_status": "unreviewed",
    "scope": "model chunk generation; not WebRTC or full voice-assistant latency",
}
(out / f"{name}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
