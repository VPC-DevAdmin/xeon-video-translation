"use client";
import { useEffect, useRef, useState } from "react";

type Checks = { voice?: Record<string, unknown> & { problems?: string[]; warnings?: string[]; ok?: boolean }; portrait?: Record<string, unknown> & { problems?: string[]; warnings?: string[]; ok?: boolean }; idle?: unknown };
type Script = { language: string; text: string; rules: { portrait: string; idle_seconds: number; voice_min_seconds: number; voice_target_seconds: number; voice_max_seconds: number }; consent: { version: string; text: string }; stock_voices?: string[] };
type Props = { language: string; onDone: (persona: { id: string; name: string }) => void };

const STEPS = ["consent", "portrait", "idle", "voice", "review"] as const;
type Step = typeof STEPS[number];
const PRIMARY = "aurora-wizard-primary";
const SECONDARY = "aurora-wizard-secondary";

export default function PersonaWizard({ language, onDone }: Props) {
  const [step, setStep] = useState<Step>("consent");
  const [script, setScript] = useState<Script | null>(null);
  const [name, setName] = useState("");
  const [consent, setConsent] = useState(false);
  const [portrait, setPortrait] = useState<Blob | null>(null);
  const [portraitUrl, setPortraitUrl] = useState("");
  const [idle, setIdle] = useState<Blob | null>(null);
  const [idleLeft, setIdleLeft] = useState<number | null>(null);
  const [voice, setVoice] = useState<Blob | null>(null);
  const [voiceSeconds, setVoiceSeconds] = useState(0);
  const [voiceMode, setVoiceMode] = useState<"record" | "stock">("record");
  const [stockVoice, setStockVoice] = useState("");
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [checks, setChecks] = useState<Checks | null>(null);
  const [persona, setPersona] = useState<{ id: string; name: string } | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [retaking, setRetaking] = useState(false);          // came from review: go straight back
  const [previewUrl, setPreviewUrl] = useState("");
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const timer = useRef<number | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const urls = useRef<string[]>([]);               // object URLs to revoke

  // Object URLs for previews are revoked when replaced and when the wizard closes.
  function objectUrl(blob: Blob) { const url = URL.createObjectURL(blob); urls.current.push(url); return url; }
  function revokeUrls() { urls.current.forEach(u => URL.revokeObjectURL(u)); urls.current = []; }

  useEffect(() => {
    fetch(`/api/personas/script?language=${encodeURIComponent(language)}`).then(async r => { if (r.ok) setScript(await r.json()); else setError(await r.text()); }).catch(e => setError(String(e)));
    return () => { stopStream(); revokeUrls(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language]);

  function stopStream() {
    recorder.current?.state === "recording" && recorder.current.stop();
    stream.current?.getTracks().forEach(t => t.stop()); stream.current = null;
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    void audioContext.current?.close().catch(() => undefined); audioContext.current = null;
  }

  // The portrait and idle steps need the camera; the voice step needs only the microphone.
  async function openMedia(kind: "camera" | "microphone") {
    stopStream();
    const media = await navigator.mediaDevices.getUserMedia(kind === "camera"
      ? { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false }
      : { video: false, audio: { echoCancellation: true, noiseSuppression: true } });
    stream.current = media;
    if (kind === "microphone") {
      const context = new AudioContext(); audioContext.current = context;
      const source = context.createMediaStreamSource(media);
      const node = context.createAnalyser(); node.fftSize = 1024; source.connect(node);
      const data = new Uint8Array(node.fftSize);
      timer.current = window.setInterval(() => { node.getByteTimeDomainData(data); let sum = 0; for (const v of data) { const d = (v - 128) / 128; sum += d * d; } setLevel(Math.sqrt(sum / data.length)); }, 100);
    }
  }

  // The preview <video> is rendered per step, so the camera is opened (and the
  // stream re-attached) after the step has mounted, not before.
  useEffect(() => {
    let cancelled = false;
    async function setup() {
      setCameraReady(false);
      if (step === "portrait" || step === "idle") await openMedia("camera");
      else if (step === "voice" && voiceMode === "record") await openMedia("microphone");
      else if (step === "voice") { stopStream(); setCameraReady(true); return; }
      else { stopStream(); return; }
      if (cancelled) return;
      if (video.current && stream.current) { video.current.srcObject = stream.current; await video.current.play().catch(() => undefined); }
      setCameraReady(true);
    }
    setup().catch(e => setError((step === "voice" ? "Microphone" : "Camera") + " access failed: " + String(e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, voiceMode]);

  function go(next: Step) { setError(""); setStep(next); }

  function capturePortrait() {
    const el = video.current; if (!el) { setError("Camera preview is not ready yet."); return; }
    if (!el.videoWidth) { setError("Camera has not delivered a frame yet; wait a moment and try again."); return; }
    const side = Math.min(el.videoWidth, el.videoHeight);
    const canvas = document.createElement("canvas"); canvas.width = 768; canvas.height = 768;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    ctx.drawImage(el, (el.videoWidth - side) / 2, (el.videoHeight - side) / 2, side, side, 0, 0, 768, 768);
    canvas.toBlob(blob => { if (blob) { setPortrait(blob); setPortraitUrl(objectUrl(blob)); } }, "image/png");
  }

  function recordIdle() {
    if (!stream.current || !script) return;
    const chunks: Blob[] = [];
    const rec = new MediaRecorder(new MediaStream(stream.current.getVideoTracks()), { mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm" });
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = () => { setIdle(new Blob(chunks, { type: "video/webm" })); setIdleLeft(null); };
    recorder.current = rec; rec.start(250);
    const seconds = script.rules.idle_seconds; setIdleLeft(seconds);
    let left = seconds;
    const tick = window.setInterval(() => { left -= 1; setIdleLeft(left); if (left <= 0) { window.clearInterval(tick); rec.state === "recording" && rec.stop(); } }, 1000);
  }

  function startVoice() {
    if (!stream.current || !script) return;
    const chunks: Blob[] = [];
    const rec = new MediaRecorder(new MediaStream(stream.current.getAudioTracks()), { mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm" });
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = () => { setVoice(new Blob(chunks, { type: "audio/webm" })); setRecording(false); };
    recorder.current = rec; rec.start(250); setRecording(true); setVoiceSeconds(0);
    const started = Date.now();
    const tick = window.setInterval(() => {
      const s = (Date.now() - started) / 1000; setVoiceSeconds(s);
      if (s >= script.rules.voice_max_seconds) { window.clearInterval(tick); rec.state === "recording" && rec.stop(); }
      if (rec.state !== "recording") window.clearInterval(tick);
    }, 200);
  }
  function stopVoice() { recorder.current?.state === "recording" && recorder.current.stop(); }

  async function submit() {
    const usingStock = voiceMode === "stock" && !!stockVoice;
    if (!portrait || !script || (!voice && !usingStock)) return;
    setBusy(usingStock ? "Checking the portrait…" : "Checking the portrait and the recording, building the voice…"); setError(""); setChecks(null);
    try {
      const form = new FormData();
      form.append("name", name || "New person"); form.append("language", language); form.append("consent", "yes"); form.append("script", script.text);
      form.append("portrait", portrait, "portrait.png");
      if (usingStock) form.append("stock_voice", stockVoice); else if (voice) form.append("voice", voice, "voice.webm");
      if (idle) form.append("idle", idle, "idle.webm");
      const response = await fetch("/api/personas", { method: "POST", body: form });
      const body = await response.json().catch(() => null);
      if (response.status === 422 && body?.detail?.checks) { setChecks(body.detail.checks); setError("Some checks failed. Retake the parts marked below."); return; }
      if (!response.ok) throw new Error(body?.detail ? JSON.stringify(body.detail) : await response.text());
      setChecks(body.checks); setPersona({ id: body.persona.id, name: body.persona.name });
      setBusy("Rendering a voice preview…");
      const preview = await fetch(`/api/personas/${body.persona.id}/preview`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      if (preview.ok) setPreviewUrl(objectUrl(await preview.blob()));
    } catch (e) { setError(String(e)); } finally { setBusy(""); }
  }

  const problems = (c?: { problems?: string[]; warnings?: string[] }) => <>
    {c?.problems?.length ? <ul className="list-disc ml-5 text-red-600">{c.problems.map((p, i) => <li key={i}>{p}</li>)}</ul> : <p className="text-green-700">OK</p>}
    {c?.warnings?.length ? <ul className="list-disc ml-5 text-amber-600">{c.warnings.map((p, i) => <li key={i}>{p}</li>)}</ul> : null}
  </>;

  return <section className="aurora-wizard">
    <ol className="aurora-wizard-steps" aria-label="Setup progress">{STEPS.map((s, i) => <li key={s} className={s === step ? "is-current" : ""} title={s}>{i + 1}</li>)}</ol>
    {error && <p role="alert" className="text-red-600">{error}</p>}

    {step === "consent" && script && <div className="space-y-3">
      <label className="block">Name <input className="border px-2 py-1 ml-2" value={name} onChange={e => setName(e.target.value)} placeholder="How should we call this persona?" /></label>
      <p className="text-sm">{script.consent.text}</p>
      <label className="flex gap-2 items-start"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} /> <span>I agree (consent version {script.consent.version})</span></label>
      <p className="text-sm text-ink-400">You will take a portrait and choose a voice. An {script.rules.idle_seconds}-second idle clip is optional.</p>
      <button className={PRIMARY} disabled={!consent} onClick={() => go("portrait")}>Start</button>
    </div>}

    {step === "portrait" && script && <div className="space-y-3">
      <p>{script.rules.portrait}</p>
      <div className="relative inline-block w-full max-w-sm">
        <video ref={video} autoPlay playsInline muted className="rounded bg-black w-full aspect-square object-cover" />
        <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center justify-center"><div className="border-2 border-white/80 rounded-[50%] w-[44%] h-[58%] -translate-y-[6%]" /></div>
      </div>
      <p className="text-sm text-ink-400">Fill the oval with your face. Light should fall on your face, not come from behind you. A sharp phone photo (front camera, good light) gives the renderer much more detail than a webcam frame.</p>
      <div className="flex gap-3 items-center flex-wrap">
        <button className={PRIMARY} onClick={capturePortrait} disabled={!cameraReady}>{portrait ? "Capture again" : "Capture portrait"}</button>
        <label className={SECONDARY + " cursor-pointer"}>Upload a photo instead
          <input type="file" accept="image/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) { setPortrait(f); setPortraitUrl(objectUrl(f)); } }} />
        </label>
        {portraitUrl && <img src={portraitUrl} alt="captured portrait" className="w-24 h-24 rounded object-cover" />}
        <button className={SECONDARY} disabled={!portrait} onClick={() => go(retaking ? "review" : "idle")}>{retaking ? "Back to review" : "Next"}</button>
      </div>
    </div>}

    {step === "idle" && script && <div className="space-y-3">
      <p>Now sit as you would while listening: look at the camera, breathe and blink normally, don&apos;t talk. We record {script.rules.idle_seconds} seconds. This clip is optional; it feeds the footage-based renderer later.</p>
      <video ref={video} autoPlay playsInline muted className="rounded bg-black w-full max-w-md aspect-video" />
      <div className="flex gap-3 items-center">
        <button className={PRIMARY} disabled={idleLeft !== null || !cameraReady} onClick={recordIdle}>{idleLeft !== null ? `Recording… ${idleLeft}` : idle ? "Record again" : "Record idle clip"}</button>
        {idle && <span className="text-green-700">clip captured</span>}
        <button className={SECONDARY} onClick={() => go("voice")}>{idle ? "Next" : "Skip"}</button>
      </div>
    </div>}

    {step === "voice" && script && <div className="space-y-3">
      <div className="flex gap-4 text-sm">
        <label className="flex items-center gap-1"><input type="radio" checked={voiceMode === "record"} onChange={() => setVoiceMode("record")} /> Record my voice</label>
        <label className="flex items-center gap-1"><input type="radio" checked={voiceMode === "stock"} onChange={() => setVoiceMode("stock")} /> Use a stock voice</label>
      </div>
      {voiceMode === "record" ? <>
        <p>Read this aloud at a natural pace (at least {script.rules.voice_min_seconds} seconds):</p>
        <blockquote className="border-l-4 pl-3 text-lg leading-relaxed">{script.text}</blockquote>
        <div className="flex gap-3 items-center">
          {!recording ? <button className={PRIMARY} onClick={startVoice} disabled={!cameraReady}>{voice ? "Record again" : "Start recording"}</button> : <button className={PRIMARY} onClick={stopVoice}>Stop ({voiceSeconds.toFixed(0)} s)</button>}
          <meter min={0} max={0.5} value={level} className="w-40" aria-label="microphone level" />
          {voice && !recording && <span className="text-green-700">{voiceSeconds.toFixed(0)} s recorded</span>}
          <button className={SECONDARY} disabled={!voice || recording} onClick={() => go("review")}>{retaking ? "Back to review" : "Next"}</button>
        </div>
      </> : <>
        <p>Pick one of the built-in voices. You can hear it in the preview on the next step.</p>
        <div className="flex gap-3 items-center">
          <select className="border px-2 py-1" value={stockVoice} onChange={e => setStockVoice(e.target.value)}>
            <option value="">Choose a voice…</option>
            {(script.stock_voices || []).map(v => <option key={v} value={v}>{v}</option>)}
          </select>
          <button className={SECONDARY} disabled={!stockVoice} onClick={() => go("review")}>{retaking ? "Back to review" : "Next"}</button>
        </div>
      </>}
    </div>}

    {step === "review" && <div className="space-y-3">
      <div className="flex gap-4 items-start">{portraitUrl && <img src={portraitUrl} alt="portrait" className="w-32 h-32 rounded object-cover" />}
        <ul className="text-sm"><li>Portrait: {portrait ? "captured" : "missing"}</li><li>Idle clip: {idle ? "captured" : "skipped"}</li><li>Voice: {voiceMode === "stock" ? (stockVoice ? `stock voice “${stockVoice}”` : "no stock voice chosen") : voice ? `${voiceSeconds.toFixed(0)} s recorded` : "missing"}</li></ul></div>
      {!persona && <button className={PRIMARY} disabled={!!busy || !portrait || (voiceMode === "stock" ? !stockVoice : !voice)} onClick={submit}>{busy || (checks ? "Check again and build" : "Check and build this persona")}</button>}
      {checks && <div className="grid gap-3 sm:grid-cols-2 text-sm">
        <div><h3 className="font-semibold">Portrait</h3>{problems(checks.portrait)}</div>
        <div><h3 className="font-semibold">Voice</h3>{problems(checks.voice)}
          {checks.voice && "speaker" in checks.voice && <p className="text-ink-400">stock voice {String(checks.voice.speaker)}</p>}
          {checks.voice && "script_match" in checks.voice && <p className="text-ink-400">words matched: {Math.round(Number(checks.voice.script_match) * 100)}% · {String(checks.voice.duration_seconds)} s · {String(checks.voice.level_dbfs)} dBFS</p>}</div>
      </div>}
      {checks && !persona && <div className="flex gap-3"><button className={SECONDARY} onClick={() => { setRetaking(true); go("portrait"); }}>Retake portrait</button><button className={SECONDARY} onClick={() => { setRetaking(true); go("voice"); }}>Change voice</button></div>}
      {persona && <div className="space-y-2">
        <p className="text-green-700">Persona “{persona.name}” is ready.</p>
        {previewUrl ? <audio controls src={previewUrl} /> : <p className="text-sm text-ink-400">{busy || "No preview available."}</p>}
        <button className={PRIMARY} onClick={() => { stopStream(); revokeUrls(); onDone(persona); }}>Use this persona</button>
      </div>}
    </div>}
  </section>;
}
