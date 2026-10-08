import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.pipeline.tts import _get_xtts

out = Path('/jobs/quality-v3')
out.mkdir(exist_ok=True)
text = "Good morning! I'm glad you're here. I've been working on something pretty useful. Imagine sharing a video in another language, while still sounding like yourself. That's what we're trying to get right."
tts = _get_xtts()
model = tts.synthesizer.tts_model
condition, speaker = model.get_conditioning_latents(audio_path=['/jobs/quality-v2/reference.wav'])
records = []
for name, temperature, top_p in [('xtts-conversational-065', 0.65, 0.85), ('xtts-conversational-085', 0.85, 0.90)]:
    torch.manual_seed(42)
    np.random.seed(42)
    start = time.perf_counter()
    result = model.inference(text, 'en', condition, speaker, temperature=temperature,
                             top_p=top_p, repetition_penalty=10., speed=1., enable_text_splitting=False)
    audio = np.asarray(result['wav'])
    sf.write(out / f'{name}.wav', audio, 24000, subtype='PCM_16')
    record = {'name': name, 'text': text, 'seed': 42, 'temperature': temperature,
              'top_p': top_p, 'repetition_penalty': 10., 'speed': 1.,
              'sample_rate': 24000, 'duration': len(audio) / 24000,
              'seconds': time.perf_counter() - start, 'split_sentences': False}
    records.append(record)
    print(json.dumps(record), flush=True)
(out/'voice-trials.json').write_text(json.dumps(records, indent=2))
(out/'friendly-script.txt').write_text(text)
