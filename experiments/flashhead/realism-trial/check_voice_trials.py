import json
import re
from pathlib import Path
from app.pipeline.transcribe import transcribe

out = Path('/jobs/quality-v3')
expected = (out/'friendly-script.txt').read_text()
words = lambda text: re.findall(r"\w+", text.casefold())
records=[]
for path in out.glob('*conversational*.wav'):
    result = transcribe(path, out / (path.stem + '.asr.json'), language='en')
    a,b=words(expected),words(result.text)
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    record={'file':path.name,'expected':expected,'recognized':result.text,'word_errors':row[-1],'reference_words':len(a)}
    records.append(record)
    print(json.dumps(record),flush=True)
(out/'voice-content-check.json').write_text(json.dumps(records,indent=2))
