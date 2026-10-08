"""Prepare a new animation track; never retime existing translated footage."""
import argparse
import array
import json
from pathlib import Path
import re
import subprocess
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, type=Path)
parser.add_argument('--alignment', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with wave.open(str(args.input)) as stream:
    assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
    rate = stream.getframerate()
    samples = array.array('h', stream.readframes(stream.getnframes()))
duration = len(samples) / rate
words = [w for s in json.loads(args.alignment.read_text())['segments'] for w in s['words']]
if not words:
    raise ValueError('Word alignment is required before removing pauses')
proc = subprocess.run(['ffmpeg','-hide_banner','-i',str(args.input),'-af','silencedetect=noise=-40dB:d=1.0','-f','null','-'], capture_output=True, text=True, check=True)
silences = [(float(end)-float(length), float(end)) for end,length in re.findall(r'silence_end: ([\d.]+) \| silence_duration: ([\d.]+)',proc.stderr)]
cuts=[]
for start,end in silences:
    # Keep 200 ms on each side and exclude speech plus 80 ms guards.
    a,b=start+.2,end-.2
    if a <= words[0]['start'] or b >= words[-1]['end'] or b <= a:
        continue
    if any(w['start']-.08 < b and w['end']+.08 > a for w in words):
        continue
    cuts.append((round(a*rate),round(b*rate)))
kept = array.array('h')
cursor=0
fade=max(1,round(.005*rate))
for a,b in cuts:
    part=samples[cursor:a]
    for i in range(min(fade,len(part))): part[-1-i]=round(part[-1-i]*i/fade)
    kept.extend(part)
    cursor=b
    for i in range(min(fade,len(samples)-cursor)): samples[cursor+i]=round(samples[cursor+i]*i/fade)
kept.extend(samples[cursor:])
raw=args.output/'edited-native.wav'
with wave.open(str(raw),'wb') as stream:
    stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate);stream.writeframes(kept.tobytes())
# Static gain only: no time stretching, nonlinear loudness or pitch modification.
peak=max(abs(x) for x in kept)/32768
gain=min(1.,10**(-2/20)/max(peak,1e-9))
master=args.output/'master.wav'
subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-af',f'volume={gain:.10f}','-ar',str(rate),'-ac','1','-c:a','pcm_s16le',str(master)],check=True)
subprocess.run(['ffmpeg','-v','error','-y','-i',str(master),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(args.output/'conditioning-16k.wav')],check=True)
report={'input':str(args.input),'source_duration':duration,'master_sample_rate':rate,'duration':len(kept)/rate,'cuts':[{'start':a/rate,'end':b/rate,'removed_seconds':(b-a)/rate} for a,b in cuts],'gain':gain,'speed_change':False,'scope':'new video creation only; all cuts lie in measured silence outside recognized words plus guards'}
(args.output/'preparation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
