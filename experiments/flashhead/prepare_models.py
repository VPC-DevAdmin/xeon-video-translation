import json, pathlib, subprocess
from huggingface_hub import snapshot_download

root = pathlib.Path("/experiment-models")
records = []
for repo, folder, patterns in [
    (
        "Soul-AILab/SoulX-FlashHead-1_3B",
        "SoulX-FlashHead-1_3B",
        ["Model_Lite/*", "VAE_LTX/*", "Model_Pro/*", "VAE_Wan/*", "README.md"],
    ),
    (
        "facebook/wav2vec2-base-960h",
        "wav2vec2-base-960h",
        ["*.json", "*.safetensors", "pytorch_model.bin"],
    ),
]:
    revision = {
        "Soul-AILab/SoulX-FlashHead-1_3B": "59119b6c681230c3eeee157e224ae1941746711e",
        "facebook/wav2vec2-base-960h": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    }[repo]
    print("Downloading pinned model", repo, revision, flush=True)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=root / folder,
        allow_patterns=patterns,
        max_workers=3,
    )
    records.append({"model": repo, "revision": revision, "path": str(root / folder)})
(root / "model-revisions.json").write_text(json.dumps(records, indent=2))
print("Model downloads complete", flush=True)
