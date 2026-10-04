#!/usr/bin/env python3
"""Hardware smoke probe: NVDEC -> DLPack -> CUDA resize, with bounded batches.

This is an isolated compatibility probe, not the production media adapter or an
end-to-end benchmark. It does not validate timestamps, color fidelity or encoding.
See NVIDIA's PyNvVideoCodec API guide; use a dedicated, explicitly selected GPU.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def probe(fixture, max_frames=120, batch_size=4):
    if max_frames < 1 or batch_size < 1 or batch_size > 32:
        raise ValueError('frames must be positive and batch size must be between 1 and 32')
    import torch
    if not torch.cuda.is_available():
        return {'status': 'blocked', 'reason': 'CUDA unavailable', 'quality_status': 'unreviewed'}
    import PyNvVideoCodec as nvc
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f'Expected target sm_120 hardware, found {capability}')
    with fixture.open('rb') as source:
        fixture_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.monotonic()
    decoder = nvc.SimpleDecoder(str(fixture), gpu_id=0, use_device_memory=True,
                               output_color_type=nvc.OutputColorType.RGBP)
    total = min(len(decoder), max_frames)
    count = 0
    shapes = set()
    checksum = torch.zeros((), device='cuda:0')
    with torch.inference_mode():
        while count < total:
            frames = decoder.get_batch_frames(min(batch_size, total-count))
            if not frames:
                raise RuntimeError('Decoder ended before the expected frame count')
            tensors = [torch.from_dlpack(frame) for frame in frames]
            for tensor in tensors:
                if tensor.device.type != 'cuda' or tensor.device.index != 0:
                    raise RuntimeError('DLPack frame is not resident on the selected CUDA device')
                if tensor.ndim != 3 or tensor.shape[0] != 3:
                    raise RuntimeError(f'Expected planar RGB CHW, found {tensor.shape}')
                shapes.add(tuple(tensor.shape))
            # stack and float allocate on device; DLPack imports share decode memory.
            batch = torch.stack(tensors).float().div_(255)
            resized = torch.nn.functional.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
            checksum += resized.mean()
            # Conservative lifetime boundary before releasing decoder surfaces.
            torch.cuda.synchronize(0)
            count += len(frames)
            del resized, batch, tensors, frames, tensor
    if not count:
        raise RuntimeError('No frames decoded')
    elapsed = time.monotonic() - started
    if not torch.isfinite(checksum).item():
        raise RuntimeError('Non-finite GPU output')
    return {'status': 'completed', 'quality_status': 'unreviewed',
            'scope': 'decode plus CUDA resize compatibility only',
            'fixture_sha256': fixture_hash, 'frames': count, 'shapes': sorted(shapes),
            'wall_seconds': elapsed, 'smoke_frames_per_second': count/elapsed,
            'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'memory_scope': 'PyTorch allocator only; excludes NVDEC and external allocations',
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'pynvvideocodec': importlib.metadata.version('PyNvVideoCodec'),
            'device': torch.cuda.get_device_name(0), 'compute_capability': list(capability),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = probe(args.fixture, args.frames, args.batch_size)
    except ImportError as exc:
        result = {'status': 'blocked', 'reason': str(exc), 'quality_status': 'unreviewed'}
    except Exception as exc:  # noqa: BLE001 -- serialize native library failures at the CLI boundary
        result = {'status': 'failed', 'reason': str(exc), 'quality_status': 'unreviewed'}
    output = args.output
    if output is None and os.environ.get('EXPERIMENT_OUTPUT_DIR'):
        output = Path(os.environ['EXPERIMENT_OUTPUT_DIR']) / 'gpu-frames.json'
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
