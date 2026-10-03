"""Per-session VAD. Silero ONNX has no PyTorch dependency in ingest."""

import os
from pathlib import Path
import numpy as np


class Detector:
    def __init__(self):
        self.threshold = float(os.getenv("AVATAR_VAD_THRESHOLD", ".5"))
        self.kind = os.getenv("AVATAR_VAD_BACKEND", "energy")
        self.pending = np.empty(0, dtype=np.float32)
        self.probability = 0.0
        if self.kind == "silero":
            import onnxruntime as ort

            path = Path(os.environ["SILERO_VAD_MODEL"])
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros((1, 64), dtype=np.float32)
        elif self.kind != "energy":
            raise ValueError("unknown VAD backend")

    def speech(self, samples):
        if self.kind == "energy":
            return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) > float(
                os.getenv("AVATAR_VAD_RMS", "450")
            )
        self.pending = np.concatenate(
            [self.pending, samples.astype(np.float32) / 32768]
        )
        while len(self.pending) >= 512:
            frame = self.pending[:512].reshape(1, -1)
            self.pending = self.pending[512:]
            combined = np.concatenate([self.context, frame], axis=1)
            out, self.state = self.session.run(
                None,
                {
                    "input": combined,
                    "state": self.state,
                    "sr": np.array(16000, dtype=np.int64),
                },
            )
            self.context = combined[:, -64:]
            self.probability = float(out.reshape(-1)[0])
        return self.probability >= self.threshold
