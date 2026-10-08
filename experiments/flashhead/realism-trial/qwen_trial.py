import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

out = Path('/experiment/quality-v3')
out.mkdir(exist_ok=True)
text = "Good morning! I'm glad you're here. I've been working on something pretty useful. Imagine sharing a video in another language, while still sounding like yourself. That's what we're trying to get right."
torch.manual_seed(42)
np.random.seed(42)
model = Qwen3TTSModel.from_pretrained('/experiment-models/Qwen3-TTS-1.7B-Base', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
start = time.perf_counter()
wavs, sr = model.generate_voice_clone(text=text, language='English', ref_audio='/jobs/quality-v2/reference.wav', ref_text='Good morning.', non_streaming_mode=True, max_new_tokens=1024)
sf.write(out/'qwen-conversational.wav', wavs[0], sr, subtype='PCM_16')
report = {'model': 'Qwen3-TTS-12Hz-1.7B-Base', 'revision': 'fd4b254389122332181a7c3db7f27e918eec64e3', 'text':text, 'sample_rate':sr, 'duration':len(wavs[0])/sr, 'generation_seconds':time.perf_counter()-start, 'seed':42, 'non_streaming_mode':True, 'attention':'sdpa', 'dtype':'bfloat16', 'device':'cuda:0', 'style_instruction_supported':False}
(out/'qwen-conversational.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report),flush=True)
