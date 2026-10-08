import importlib.util
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
import torch


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "app/musetalk" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


blend = load("tensor_blending")
reference = load("blending")


@pytest.mark.parametrize("box", [(24, 18, 80, 86), (0, 0, 30, 34), (67, 60, 96, 96)])
@pytest.mark.parametrize("feather", [0.04, 0.1])
def test_compositing_matches_reference_geometry_and_mask(box, feather):
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    face = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[64:450, 80:420] = 255
    prepared = blend.prepare(
        torch.from_numpy(image).permute(2, 0, 1),
        box,
        torch.from_numpy(mask),
        feather_ratio=feather,
    )
    result = blend.composite(prepared, torch.from_numpy(face).permute(2, 0, 1)[None])[
        0
    ].numpy()
    x, y, x1, y1 = box
    expected = reference.composite_np(
        image, cv2.resize(face, (x1 - x, y1 - y)), box, mask, feather_ratio=feather
    )
    error = np.abs(result.astype(float) - expected.astype(float))
    assert error.max() <= 3
    assert error.mean() < 0.15
    np.testing.assert_array_equal(image[:y, :], result[:y, :])


def test_batch_and_disclosure():
    image = torch.zeros((3, 64, 64), dtype=torch.uint8)
    badge = torch.zeros((1, 1, 64, 64))
    badge[:, :, -5:, :10] = 1
    p = blend.prepare(
        image, (10, 10, 40, 50), torch.ones((32, 32)) * 255, disclosure=badge
    )
    result = blend.composite(p, torch.ones((3, 3, 24, 24)) * 100)
    assert result.shape == (3, 64, 64, 3)
    assert (result[:, -5:, :10] == 255).all()
    assert torch.equal(result[0], result[2])


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="inside"):
        blend.prepare(torch.zeros((3, 32, 32)), (-1, 0, 30, 30), torch.zeros((32, 32)))


def test_vae_tensor_decode_preserves_bgr_and_portable_wrapper():
    # Exercise the actual methods without downloading a diffusion model.
    import ast
    from types import SimpleNamespace, MethodType

    source = Path(__file__).parents[1] / "app/musetalk/models/vae.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VAE"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"decode_latents_tensor", "decode_latents"}
    ]
    namespace = {"torch": torch, "np": np}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    vae = SimpleNamespace(
        scaling_factor=0.5,
        vae=SimpleNamespace(
            dtype=torch.float16, decode=lambda x: SimpleNamespace(sample=x)
        ),
    )
    vae.decode_latents_tensor = MethodType(namespace["decode_latents_tensor"], vae)
    vae.decode_latents = MethodType(namespace["decode_latents"], vae)
    latent = torch.tensor([0.5, -0.5, -0.5]).reshape(1, 3, 1, 1)
    result = vae.decode_latents_tensor(latent)
    assert result.shape == (1, 3, 1, 1) and result.dtype == torch.uint8
    assert result.flatten().tolist() == [0, 0, 255]
    assert vae.decode_latents(latent).tolist() == [[[[0, 0, 255]]]]
