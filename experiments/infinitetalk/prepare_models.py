"""Download the exact weights used in this trial, never floating model heads."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/quality-models'))
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name('model-revisions.json').read_text())
args.root.mkdir(parents=True, exist_ok=True)
resolved = []
for model in manifest:
    destination = args.root / Path(model['local_dir']).name
    patterns = ['single/infinitetalk.safetensors', 'README.md', 'LICENSE'] if model['repo'] == 'MeiGen-AI/InfiniteTalk' else None
    snapshot_download(
        model['repo'], revision=model['revision'], local_dir=destination,
        allow_patterns=patterns, ignore_patterns=['*.msgpack', '*.h5', '*.ot'], max_workers=8,
    )
    resolved.append({**model, 'local_dir': str(destination)})
(args.root / 'model-revisions.json').write_text(json.dumps(resolved, indent=2))
