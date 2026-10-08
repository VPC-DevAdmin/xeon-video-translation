from insightface.app import FaceAnalysis
import numpy as np
import torch

INSIGHTFACE_DETECT_SIZE = 512


def _trimmed_pack_root(root: str, pack: str, keep: tuple) -> str:
    """Return a root whose models/<pack>/ holds only `keep` (hard links or
    copies of the originals). Falls back to `root` if anything is missing."""
    import os
    import shutil

    source = os.path.join(root, "models", pack)
    if not all(os.path.exists(os.path.join(source, name)) for name in keep):
        return root
    trimmed_root = os.path.join(root, "trimmed")
    target = os.path.join(trimmed_root, "models", pack)
    try:
        os.makedirs(target, exist_ok=True)
        for name in keep:
            dst = os.path.join(target, name)
            src = os.path.join(source, name)
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                continue
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        for name in os.listdir(target):
            if name not in keep:
                os.remove(os.path.join(target, name))
    except OSError:
        return root
    return trimmed_root


class FaceDetector:
    def __init__(self, device="cuda"):
        # CPU patch: route CPU device names through onnxruntime's CPU
        # execution provider and InsightFace's ctx_id=-1 convention.
        # Upstream assumed CUDA end-to-end; on our host neither is true.
        is_cuda = str(device).startswith("cuda")
        from gpu_runtime import ort_cuda_provider
        providers = [ort_cuda_provider()] if is_cuda else ["CPUExecutionProvider"]
        # Upstream used a relative "checkpoints/auxiliary", which inside the
        # container is ephemeral: every recreated container re-downloaded
        # the 280 MB buffalo_l pack. Root it in the shared models volume
        # (same location the MuseTalk service uses) unless overridden.
        import os
        root = os.environ.get(
            "INSIGHTFACE_ROOT",
            os.path.join(os.environ.get("MODEL_CACHE_DIR", "/models"), "insightface"),
        )
        # insightface parses every ONNX file in the pack before applying
        # allowed_modules, and with the pure-python protobuf this image is
        # pinned to (mediapipe 0.10.11 needs protobuf 3.x) that is 36 s for
        # the 340 MB buffalo_l pack. Only det_10g and 2d106det are used, so
        # expose a view of the pack containing just those two files.
        root = _trimmed_pack_root(root, "buffalo_l", ("det_10g.onnx", "2d106det.onnx"))
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root=root,
            providers=providers,
        )
        ctx_id = cuda_to_int(device) if is_cuda else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
        if is_cuda:
            from gpu_runtime import require_ort_cuda
            require_ort_cuda(self.app)

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0
        # Confidence of the face returned by this call (0.0 when none); the
        # occlusion gate (latentsync_driver.face_parse) reads it.
        self.last_score = 0.0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > 1.5 or w / h < 0.2:
                    continue
                if face.det_score < threshold:
                    continue
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face

        if get_face_store is None:
            return None, None
        else:
            face = get_face_store
            self.last_score = float(face.det_score)
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            # Debug dump: annotated source frame with the final
            # derived bbox + all 106 landmarks drawn on top. No-op
            # unless LATENTSYNC_DEBUG_DUMP=1. See utils/_debug.py for
            # the rationale — this is the reviewer-suggested diagnostic
            # for "does detection work but landmarks are garbage?"
            try:
                from . import _debug
                _debug.dump_annotated_frame(
                    "01_detection_and_landmarks",
                    frame,
                    bbox=(x1, y1, x2, y2),
                    landmarks=lmk,
                )
            except Exception:
                pass

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
