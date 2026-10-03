"use client";
import { useEffect, useRef, useState } from "react";
import { API_BASE_URL as API, artifactUrl, JobRecord } from "../lib/api";

type Segment={start:number;end:number;text:string;speaker?:string};
type Document={segments:Segment[];review_issues?:{segment:number;issues:string[]}[]};
type Job=JobRecord & {options?:Record<string,any>;parent_job_id?:string;queue_position?:number};
async function request(path:string,init?:RequestInit) {
  const response=await fetch(`${API}${path}`,init);
  if(!response.ok) throw new Error(await response.text());
  return response.json();
}
export default function Studio() {
  const [jobs,setJobs]=useState<Job[]>([]),[job,setJob]=useState<Job|null>(null);
  const [translation,setTranslation]=useState<Document|null>(null),[transcript,setTranscript]=useState<Document|null>(null);
  const [options,setOptions]=useState<Record<string,any>>({}),[glossary,setGlossary]=useState("{}");
  const [voices,setVoices]=useState<string[]>([]),[source,setSource]=useState("");
  const [error,setError]=useState(""),[busy,setBusy]=useState(false),[token,setToken]=useState("");
  const [preview,setPreview]=useState("");const previewRef=useRef("");
  const terminal=!!job&&["completed","failed","cancelled"].includes(job.status);
  async function refresh() { try { setJobs((await request("/jobs?limit=100")).jobs); } catch(e) {setError(String(e));} }
  useEffect(()=>{void refresh();const id=setInterval(refresh,3000);return()=>{clearInterval(id);if(previewRef.current)URL.revokeObjectURL(previewRef.current);};},[]);
  useEffect(()=>{if(job){const next=jobs.find(j=>j.job_id===job.job_id);if(next)setJob(next);}},[jobs]);
  async function select(id:string) {
    setError("");setBusy(true);
    try {
      const next=await request(`/jobs/${id}`);setJob(next);setOptions(next.options||{});setGlossary(JSON.stringify(next.options?.glossary||{},null,2));
      async function doc(name:string) {const r=await fetch(artifactUrl(id,name));return r.ok?r.json():null;}
      const [t,s,a]=await Promise.all([doc("translation.json"),doc("transcript.json"),request(`/jobs/${id}/artifacts`)]);
      setTranslation(t);setTranscript(s);setSource(a.artifacts.find((f:any)=>f.name.startsWith("input."))?.name||"");
    }catch(e){setError(String(e));}finally{setBusy(false);}
  }
  async function action(path:string,method="POST",body?:unknown) {
    setBusy(true);setError("");
    try{const result=await request(path,{method,headers:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body)});await refresh();if(result.job_id)await select(result.job_id);if(method==="DELETE")setJob(null);}
    catch(e){setError(String(e));}finally{setBusy(false);}
  }
  async function revise(kind:"translation"|"transcript"|"options") {
    if(!job)return;
    let terms;try{terms=JSON.parse(glossary);}catch{setError("Glossary must be a JSON object mapping source terms to translated terms.");return;}
    const body:any={options:{...options,glossary:terms}};
    if(kind==="translation")body.translation=translation?.segments.map(s=>s.text);
    if(kind==="transcript"){body.transcript=transcript?.segments.map(s=>s.text);body.speakers=transcript?.segments.map(s=>s.speaker||"speaker_1");}
    await action(`/jobs/${job.job_id}/revise`,"POST",body);
  }
  async function voicePreview(){
    setBusy(true);setError("");try{
      const response=await fetch(`${API}/speech/preview`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({voice:options.voice,language:job?.target_language||"en"})});
      if(!response.ok)throw new Error(await response.text());
      if(previewRef.current)URL.revokeObjectURL(previewRef.current);previewRef.current=URL.createObjectURL(await response.blob());setPreview(previewRef.current);
    }catch(e){setError(String(e));}finally{setBusy(false);}
  }
  return <main className="mx-auto max-w-6xl p-6 space-y-6">
    <header><h1 className="text-3xl font-semibold">Translation studio</h1><nav className="flex gap-4 text-accent mt-3"><a href="/">Upload</a><a href="/live">Webcam</a><a href="/avatar">Voice avatar</a></nav></header>
    <details><summary>Account access</summary><div className="flex gap-2 mt-3"><input aria-label="Access token" type="password" value={token} onChange={e=>setToken(e.target.value)} className="bg-ink-800 p-2"/><button onClick={async()=>{const r=await fetch('/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});if(!r.ok)setError('Sign-in failed');else{setToken('');setError('');await refresh();}}}>Sign in</button><button onClick={async()=>{await fetch('/auth',{method:'DELETE'});setJobs([]);setJob(null);}}>Sign out</button></div></details>
    {error&&<p role="alert" className="text-red-300">{error}</p>}
    <section><h2 className="text-xl mb-2">Jobs</h2><div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr><th>Source</th><th>Language</th><th>Status</th><th>Queue</th></tr></thead><tbody>{jobs.map(j=><tr key={j.job_id} className="border-t border-ink-700"><td><button className="py-2 underline" onClick={()=>select(j.job_id)}>{j.input_filename||j.job_id.slice(0,8)}</button></td><td>{j.target_language}</td><td>{j.status}</td><td>{j.queue_position||"—"}</td></tr>)}</tbody></table></div>{!jobs.length&&<p>No jobs yet. Upload or record a video to begin.</p>}</section>
    {job&&<>
      <section className="space-y-3"><h2 className="text-xl">{job.input_filename} · {job.status}</h2>{job.error&&<p className="text-red-300">{job.error}</p>}
        <div className="flex flex-wrap gap-4">
          {!terminal&&<button disabled={busy} onClick={()=>action(`/jobs/${job.job_id}/cancel`)}>Cancel</button>}
          {terminal&&<><button disabled={busy} onClick={()=>action(`/jobs/${job.job_id}/revise`,"POST",{})}>Retry from checkpoint</button><button disabled={busy} onClick={()=>action(`/jobs/${job.job_id}`,"DELETE")}>Delete job and files</button><a href={`${API}/jobs/${job.job_id}/bundle`}>Download bundle</a></>}
          {translation&&<><a href={`${API}/jobs/${job.job_id}/subtitles/srt`}>SRT subtitles</a><a href={`${API}/jobs/${job.job_id}/subtitles/vtt`}>WebVTT subtitles</a></>}
        </div>
        <div className="grid md:grid-cols-2 gap-4">{source&&<div><h3>Original</h3><video controls className="w-full" src={artifactUrl(job.job_id,source)}/></div>}{job.status==="completed"&&<div><h3>Translated</h3><video controls className="w-full" src={artifactUrl(job.job_id,"final.mp4")}/><a href={artifactUrl(job.job_id,"translated_audio.wav")}>Download translated audio</a></div>}</div>
      </section>
      <section className="space-y-3"><h2 className="text-xl">Voice and quality options</h2>
        <button disabled={busy} onClick={async()=>{setBusy(true);try{setVoices((await request('/speech/voices')).voices);}catch(e){setError(String(e));}finally{setBusy(false);}}}>Load available voices</button>
        <select aria-label="Voice" className="bg-ink-800 p-2 ml-3" value={options.voice||""} onChange={e=>setOptions({...options,voice:e.target.value||null})}><option value="">Use source voice</option>{voices.map(v=><option key={v}>{v}</option>)}</select>
        <button disabled={!options.voice||busy} onClick={voicePreview}>Preview voice</button>{preview&&<audio controls autoPlay src={preview}/>}
        <div className="grid sm:grid-cols-2 gap-2">{[["windowed_lipsync","Render in bounded windows"],["rewrite_overruns","Try shorter translations when speech overruns"],["alignment","Align words to audio"],["diarization","Detect speakers"],["background_audio","Preserve background audio"]].map(([key,label])=><label key={key}><input type="checkbox" checked={!!options[key]} onChange={e=>setOptions({...options,[key]:e.target.checked})}/> {label}</label>)}</div>
        <p className="text-sm text-ink-400">Alignment, speaker detection and background separation require the optional audio-quality service. Duration rewriting uses Ollama.</p>
        {[...new Set(transcript?.segments.map(s=>s.speaker||"speaker_1")||[])].map(speaker=><label key={speaker} className="block">Voice for {speaker} <select aria-label={`Voice for ${speaker}`} className="bg-ink-800 p-2" value={options.speaker_voices?.[speaker]||""} onChange={e=>{const mapping={...options.speaker_voices};if(e.target.value)mapping[speaker]=e.target.value;else delete mapping[speaker];setOptions({...options,speaker_voices:mapping});}}><option value="">Use default voice</option>{voices.map(v=><option key={v}>{v}</option>)}</select></label>)}
        <label className="block">Background volume <input aria-label="Background volume" type="range" min="0" max="1" step="0.05" value={options.background_gain??.35} onChange={e=>setOptions({...options,background_gain:Number(e.target.value)})}/></label>
        <label className="block">Terminology glossary<textarea aria-label="Glossary" className="block w-full bg-ink-800 p-2 font-mono" value={glossary} onChange={e=>setGlossary(e.target.value)}/></label>
        <button disabled={!terminal||busy} onClick={()=>revise('options')}>Apply options in a new version</button>
      </section>
      {[["Transcript",transcript,setTranscript,"transcript"],["Translation",translation,setTranslation,"translation"]].map(([label,document,setDocument,kind]:any)=>document&&<section key={kind} className="space-y-2"><h2 className="text-xl">{label}</h2>{document.review_issues?.map((issue:any)=><p key={issue.segment} className="text-amber-300">Segment {issue.segment+1}: {issue.issues.join('; ')}</p>)}{document.segments.map((s:Segment,i:number)=><div key={i} className="grid grid-cols-[100px_1fr] gap-3"><span>{s.start.toFixed(2)}–{s.end.toFixed(2)}</span><div><textarea aria-label={`${label} segment ${i+1}`} className="w-full bg-ink-800 p-2" value={s.text} onChange={e=>setDocument({...document,segments:document.segments.map((x:Segment,k:number)=>k===i?{...x,text:e.target.value}:x)})}/>{kind==='transcript'&&<input aria-label={`Speaker ${i+1}`} className="bg-ink-800 p-1" value={s.speaker||'speaker_1'} onChange={e=>setDocument({...document,segments:document.segments.map((x:Segment,k:number)=>k===i?{...x,speaker:e.target.value}:x)})}/>}</div></div>)}<button disabled={!terminal||busy} onClick={()=>revise(kind)}>Save {kind} and regenerate changed work</button></section>)}
    </>}
  </main>;
}
