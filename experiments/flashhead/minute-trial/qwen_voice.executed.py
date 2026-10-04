from pathlib import Path
import time,json
from huggingface_hub import HfApi,snapshot_download
import torch,numpy as np,soundfile as sf
from qwen_tts import Qwen3TTSModel
out=Path('/experiment/qwen-voice');out.mkdir(exist_ok=True)
repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base';revision=HfApi().model_info(repo).sha
model_path=snapshot_download(repo,revision=revision,local_dir='/experiment-models/Qwen3-TTS-1.7B-Base')
print('Downloaded '+revision,flush=True)
torch.manual_seed(42);np.random.seed(42);start=time.perf_counter()
model=Qwen3TTSModel.from_pretrained(model_path,device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
prepared=time.perf_counter()
text=Path('/jobs/quality-v2/narration.txt').read_text().strip()
wavs,sr=model.generate_voice_clone(text=text,language='English',ref_audio='/jobs/quality-v2/reference.wav',ref_text='Good morning.',max_new_tokens=2048)
sf.write(out/'english-qwen.wav',wavs[0],sr)
report={'model':repo,'revision':revision,'device':'cuda:0','dtype':'bfloat16','attention':'SDPA','duration':len(wavs[0])/sr,'sample_rate':sr,'load_seconds':prepared-start,'generate_seconds':time.perf_counter()-prepared,'reference_seconds':1.77,'script':text}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
