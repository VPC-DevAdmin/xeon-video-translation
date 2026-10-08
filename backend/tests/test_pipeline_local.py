"""Real extraction, persistence, stage dispatch and mux; model calls are fixtures."""
import json
import shutil
from types import SimpleNamespace
import pytest
from app.config import settings
from app import storage
from app.pipeline import orchestrator as o, windowed

@pytest.mark.asyncio
async def test_complete_pipeline_and_checkpoint_reuse(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'job_artifacts_dir',tmp_path)
    monkeypatch.setattr(settings,'enable_watermark',True)
    monkeypatch.setattr(settings,'video_encoder','libx264')
    o._shutting_down = False
    o._jobs.clear();o._queues.clear();o._tasks.clear();o._lanes.clear();o._speech_lock=None
    state=o.JobState('e'*32,target_language='es',lipsync_backend='none',mode='dub')
    o.register_job(state);root=storage.job_dir(state.job_id);video=root/'input.mp4'
    windowed.run_ffmpeg(['-f','lavfi','-i','color=c=blue:size=160x120:rate=25:duration=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-c:a','aac','-shortest',video])
    calls=[]
    def transcribe(audio,out,**kwargs):
        calls.append('asr');data=dict(language='en',language_probability=1,text='Hello',duration=1,first_speech_seconds=0,segments=[dict(start=0,end=1,text='Hello',words=[])])
        out.write_text(json.dumps(data));return SimpleNamespace(to_dict=lambda:data)
    def translate(output_path,**kwargs):
        calls.append('translate');data=dict(source_language='en',target_language='es',backend='fixture',text='Hola',segments=[dict(start=0,end=1,text='Hola')])
        output_path.write_text(json.dumps(data));return SimpleNamespace(to_dict=lambda:data)
    def tts(output_path,reference_audio,**kwargs):
        calls.append('tts');shutil.copyfile(reference_audio,output_path);return SimpleNamespace(to_dict=lambda:dict(backend='fixture',language='es'))
    monkeypatch.setattr(o.transcribe,'transcribe',transcribe);monkeypatch.setattr(o.translate,'translate',translate);monkeypatch.setattr(o.tts,'synthesize',tts)
    await o.run_pipeline(state,video)
    assert state.status=='completed',state.error
    assert (root/'final.mp4').is_file() and abs(windowed.duration(root/'final.mp4')-1)<.1
    assert calls==['asr','translate','tts']
    state.status='queued';o.register_job(state)
    await o.run_pipeline(state,video)
    assert state.status=='completed' and calls==['asr','translate','tts']
    # A corrupted upstream checkpoint must invalidate downstream cached stages.
    (root/'audio.wav').write_bytes(b'corrupt');state.status='queued';o.register_job(state)
    await o.run_pipeline(state,video)
    assert state.status=='completed' and calls==['asr','translate','tts']*2
    assert not o._jobs and not o._tasks
