"""Design A for the video assistant reply: FlashHead streaming from a portrait.

Drives the FlashHead pipeline chunk by chunk the way a live session would:
audio arrives on a timeline, each 24-frame chunk is generated as soon as its
audio exists, and every chunk is timestamped so the playout margin can be
computed offline (same report shape as the LatentSync `latentsync_chunks`
event). Writes the video with the native-rate master audio and a JSON report.

Run inside the FlashHead lab container from the source root:
  PYTHONPATH=. python /path/stream_render.py --model pro --compile \
      --image /jobs/ab/inputs/portrait.png --audio /jobs/ab/inputs/reply14-16k.wav \
      --playback-audio /jobs/ab/inputs/reply14-master24k.wav --label pro-compiled-14s \
      --output /jobs/ab/out
"""

import argparse, hashlib, json, subprocess, time, wave
from collections import deque
from pathlib import Path

import numpy as np
import torch

from audio_contract import inspect_audio_pair
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
parser.add_argument("--model", choices=["lite", "pro"], default="pro")
parser.add_argument("--image", required=True)
parser.add_argument("--audio", required=True, help="16 kHz mono PCM16 conditioning copy")
parser.add_argument("--playback-audio", help="native-rate mono PCM16 master, same timeline")
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--model-root", type=Path, default=Path("/experiment-models"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--warm-chunks", type=int, default=2,
                    help="chunks of silence generated before the timed run (compile/allocator warm-up)")
args = parser.parse_args()

contract = inspect_audio_pair(args.audio, args.playback_audio)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
args.output.mkdir(parents=True, exist_ok=True)
implementation.COMPILE_MODEL = args.compile
implementation.COMPILE_VAE = args.compile

torch.cuda.reset_peak_memory_stats()
load_started = time.perf_counter()
pipeline = get_pipeline(1, str(args.model_root / "SoulX-FlashHead-1_3B"), args.model,
                        str(args.model_root / "wav2vec2-base-960h"))
get_base_data(pipeline, args.image, args.seed, False)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_started

params = get_infer_params()
sr, fps = params["sample_rate"], params["tgt_fps"]
motion, count = params["motion_frames_num"], params["frame_num"]
new_frames = count - motion                      # frames produced per chunk
slice_samples = new_frames * sr // fps           # audio consumed per chunk
cached_len = sr * params["cached_audio_duration"]
end = params["cached_audio_duration"] * fps


def make_cache():
    return deque(np.zeros(cached_len, np.float32), maxlen=cached_len)


def generate_chunk(cache, part):
    cache.extend(part)
    embedding = get_audio_embedding(pipeline, np.asarray(cache, dtype=np.float32), end - count, end)
    return run_pipeline(pipeline, embedding)[motion:].to(torch.uint8)


# Warm-up on silence: compiled graphs and allocator pools, like a mode load would.
warm_started = time.perf_counter()
warm_cache = make_cache()
for _ in range(args.warm_chunks):
    generate_chunk(warm_cache, np.zeros(slice_samples, np.float32))
torch.cuda.synchronize()
warm_seconds = time.perf_counter() - warm_started
# A fresh session starts from the portrait again.
get_base_data(pipeline, args.image, args.seed, False)
torch.cuda.synchronize()

with wave.open(args.audio) as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, sr)
    audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
frames_expected = int(np.ceil(len(audio) * fps / sr))
audio = np.pad(audio, (0, (-len(audio)) % slice_samples))

cache = make_cache()
chunks, frames = [], []
t0 = time.perf_counter()
t0_wall = time.time()
for index, part in enumerate(audio.reshape(-1, slice_samples)):
    submit = time.perf_counter() - t0
    video = generate_chunk(cache, part)
    torch.cuda.synchronize()
    done = time.perf_counter() - t0
    frames.append(video.cpu().numpy())
    restored = time.perf_counter() - t0          # frames are on the host, ready to encode
    chunks.append({"chunk": index, "frames": int(len(video)), "submit": round(submit, 3),
                   "result": round(done, 3), "restored": round(restored, 3),
                   "seconds": round(done - submit, 3), "fps": round(len(video) / (done - submit), 1)})
    print(json.dumps(chunks[-1]), flush=True)

video = np.concatenate(frames)[:frames_expected]
if len(video) != frames_expected:
    raise RuntimeError(f"missing frames: {len(video)} != {frames_expected}")
h, w = video.shape[1:3]
out_path = args.output / f"{args.label}.mp4"
proc = subprocess.run(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
     "-i", "pipe:0", "-i", contract["playback_path"], "-map", "0:v:0", "-map", "1:a:0",
     "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_path)],
    input=video.tobytes(), capture_output=True)
if proc.returncode:
    raise RuntimeError(proc.stderr.decode()[-1000:])

per_chunk = new_frames / fps
report = {
    "event": "flashhead_chunks", "design": f"flashhead-{args.model}" + ("-compiled" if args.compile else "-eager"),
    "label": args.label, "t0_wall": round(t0_wall, 3), "num_frames": new_frames, "fps": fps,
    "frames": int(len(video)), "chunks": chunks,
    "first_chunk_restored_seconds": chunks[0]["restored"],
    "last_chunk_restored_seconds": chunks[-1]["restored"],
    "min_head_start_seconds": round(max(c["restored"] - c["chunk"] * per_chunk for c in chunks), 3),
    "restored_fps_aggregate": round(len(video) / (chunks[-1]["restored"] - chunks[0]["submit"]), 2),
    "load_and_prepare_seconds": round(load_seconds, 2), "warmup_seconds": round(warm_seconds, 2),
    "warm_chunks": args.warm_chunks, "compile": args.compile, "seed": args.seed,
    "device": torch.cuda.get_device_name(), "attention": "torch SDPA",
    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
    "audio_contract": contract,
    "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in {args.image, args.audio, contract["playback_path"]}},
    "output": str(out_path),
}
(args.output / f"{args.label}.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k not in ("chunks", "input_sha256", "audio_contract")}), flush=True)
