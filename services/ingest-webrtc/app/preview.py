"""Bounded provisional ASR; a slow recognizer cannot block media ingestion."""

import asyncio
import os
import uuid
import wave
from collections import deque
import httpx
import numpy as np


class TranscriptPreview:
    def __init__(self, directory, backend, language, notify, owner="local"):
        self.directory = directory
        self.backend = backend
        self.language = language
        self.notify = notify
        self.owner = owner
        self.samples = deque()
        self.size = 0
        self.revision = 0
        self.task = None

    def push(self, samples):
        samples = samples.copy()
        self.samples.append(samples)
        self.size += len(samples)
        self.revision += 1
        while self.size > 16000 * 8 and self.samples:
            self.size -= len(self.samples.popleft())

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def run(self):
        last = 0
        while True:
            await asyncio.sleep(2)
            if self.revision == last or self.size < 16000:
                continue
            last = self.revision
            path = self.directory / f"preview-{uuid.uuid4().hex}.wav"
            with wave.open(str(path), "wb") as out:
                out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                out.writeframes(np.concatenate(self.samples).astype(np.int16).tobytes())
            try:
                async with httpx.AsyncClient(
                    timeout=30,
                    headers={
                        "x-internal-key": os.getenv("INTERNAL_API_KEY", ""),
                        "x-owner-id": self.owner,
                    },
                ) as client:
                    result = await client.post(
                        self.backend + "/speech/transcribe",
                        json={"audio_path": str(path), "language": self.language},
                    )
                    if result.is_success:
                        self.notify(result.json()["text"])
            except asyncio.CancelledError:
                # ASR may still hold the file; session orphan cleanup owns it.
                raise
            except httpx.HTTPError:
                pass
            else:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
