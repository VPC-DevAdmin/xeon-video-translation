"use client";
import { useEffect, useRef, useState } from "react";
import PersonaWizard from "../components/PersonaWizard";

const BASE = process.env.NEXT_PUBLIC_INGEST_BASE_URL || "/ingest";
type Persona = { id: string; name: string; language: string };
type Phase = "idle" | "preparing" | "connecting" | "live";

export default function AssistantPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [state, setState] = useState("");                 // listening / thinking / speaking …
  const [detail, setDetail] = useState("");
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personaId, setPersonaId] = useState("");
  const [language, setLanguage] = useState("en");
  const [wizard, setWizard] = useState(false);
  const [uploadMode, setUploadMode] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [messages, setMessages] = useState<{ who: string; text: string }[]>([]);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [showSelf, setShowSelf] = useState(true);
  const stage = useRef<HTMLVideoElement>(null);
  const selfView = useRef<HTMLVideoElement>(null);
  const peer = useRef<RTCPeerConnection | null>(null);
  const media = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const channel = useRef<RTCDataChannel | null>(null);
  const replyAt = useRef<number | null>(null);

  async function loadPersonas() {
    try {
      const r = await fetch("/api/personas");
      if (!r.ok) return;
      const list: Persona[] = (await r.json()).personas;
      setPersonas(list);
      setPersonaId(current => current || list[0]?.id || "");
      if (list[0]?.language) setLanguage(current => current || list[0].language);
    } catch { /* keep the empty list */ }
  }
  useEffect(() => { void loadPersonas(); }, []);
  useEffect(() => () => end(), []);
  useEffect(() => {
    const timer = setInterval(() => {
      if (replyAt.current === null) return;
      const left = (replyAt.current - Date.now()) / 1000;
      if (left <= 0) { replyAt.current = null; setCountdown(null); } else setCountdown(left);
    }, 100);
    return () => clearInterval(timer);
  }, []);

  function end() {
    if (session.current) void fetch(`${BASE}/assistant/sessions/${session.current}`, { method: "DELETE", keepalive: true });
    session.current = null;
    peer.current?.close(); peer.current = null;
    media.current?.getTracks().forEach(t => t.stop()); media.current = null;
    replyAt.current = null; setCountdown(null);
    setPhase("idle"); setState(""); setDetail("");
  }

  async function start() {
    setError(""); setMessages([]);
    const persona = personas.find(p => p.id === personaId);
    setPhase("preparing"); setState(`Preparing ${persona?.name ?? "the assistant"}…`);
    try {
      // Microphone (and camera for your own preview; the camera is not sent anywhere).
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: { width: 320, height: 240, facingMode: "user" } });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      }
      media.current = stream;
      if (selfView.current && stream.getVideoTracks().length) { selfView.current.srcObject = new MediaStream(stream.getVideoTracks()); void selfView.current.play().catch(() => undefined); }

      const form = new FormData(); form.append("language", language);
      if (personaId) form.append("persona_id", personaId); else if (image) form.append("image", image); else throw new Error("Set up a person first, or upload a portrait.");
      const created = await fetch(`${BASE}/assistant/sessions`, { method: "POST", body: form });
      if (!created.ok) throw new Error(await created.text());
      session.current = (await created.json()).session_id;

      setPhase("connecting"); setState("Connecting…");
      const configResponse = await fetch(`${BASE}/config`);
      if (!configResponse.ok) throw new Error("Unable to load WebRTC configuration");
      const config = await configResponse.json();
      const pc = peer.current = new RTCPeerConnection({ iceServers: config.iceServers });
      const incoming = new MediaStream();
      pc.ontrack = event => { incoming.addTrack(event.track); if (stage.current) stage.current.srcObject = incoming; };
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "connected") { setPhase("live"); setState("Listening"); setDetail("Say something."); }
        if (pc.connectionState === "failed") { setError("Connection failed; check network or TURN settings."); end(); }
      };
      const events = channel.current = pc.createDataChannel("events");
      events.onmessage = e => {
        const data = JSON.parse(e.data);
        switch (data.type) {
          case "transcript": setMessages(old => [...old.slice(-19), { who: "You", text: data.text }]); return;
          case "reply": setMessages(old => [...old.slice(-19), { who: persona?.name ?? "Assistant", text: data.text }]); return;
          case "thinking": setState("Thinking…"); setDetail(""); return;
          case "acknowledging": if (!data.pending) setDetail("acknowledging"); return;
          case "reply_scheduled": replyAt.current = Date.now() + data.start_in * 1000; setState("Thinking…"); setDetail(`answer in ${Math.max(0, data.start_in).toFixed(0)} s`); return;
          case "speaking": replyAt.current = null; setCountdown(null); setState("Speaking"); setDetail("Speak to interrupt."); return;
          case "listening": setState("Listening"); setDetail(data.stalls !== undefined ? (data.stalls ? `${data.stalls} stalled frames` : "") : ""); replyAt.current = null; setCountdown(null); return;
          case "ready": setDetail(`idle loop ${data.idle_seconds} s ready`); return;
          case "error": setError(data.message); return;
          default: return;
        }
      };
      stream.getAudioTracks().forEach(track => pc.addTrack(track, stream));
      pc.addTransceiver("video", { direction: "recvonly" });
      await pc.setLocalDescription(await pc.createOffer());
      // Gather for a bounded time: unreachable TURN entries would otherwise hold
      // the offer for tens of seconds. Whatever candidates exist by then go out.
      await new Promise<void>(resolve => {
        const done = () => { pc.removeEventListener("icegatheringstatechange", check); resolve(); };
        const timeout = setTimeout(done, 4000);
        const check = () => { if (pc.iceGatheringState === "complete") { clearTimeout(timeout); done(); } };
        pc.addEventListener("icegatheringstatechange", check); check();
      });
      const response = await fetch(`${BASE}/assistant/sessions/${session.current}/offer`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(pc.localDescription),
      });
      if (!response.ok) throw new Error(await response.text());
      await pc.setRemoteDescription(await response.json());
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => { pc.removeEventListener("connectionstatechange", check); reject(new Error("Media connection timed out; check TURN settings")); }, 20000);
        const check = () => {
          if (["connected", "failed", "closed"].includes(pc.connectionState)) {
            clearTimeout(timeout); pc.removeEventListener("connectionstatechange", check);
            if (pc.connectionState === "connected") resolve(); else reject(new Error("Media connection failed"));
          }
        };
        pc.addEventListener("connectionstatechange", check); check();
      });
      await stage.current?.play().catch(() => setError("Press play on the video to enable sound."));
    } catch (e) { setError(String(e)); end(); }
  }

  const live = phase !== "idle";
  const persona = personas.find(p => p.id === personaId);

  return <main className="mx-auto max-w-3xl p-6 space-y-4">
    <h1 className="text-2xl font-semibold">Video assistant</h1>

    <div className="relative mx-auto w-full max-w-[512px]">
      <video ref={stage} autoPlay playsInline className="w-full aspect-square rounded bg-black" />
      {!live && <div className="absolute inset-0 flex items-center justify-center text-ink-400">{persona ? `${persona.name} will appear here` : "Set up a person to begin"}</div>}
      {live && <div className="absolute left-3 top-3 rounded bg-black/60 px-3 py-1 text-sm text-white" aria-live="polite">
        {state}{countdown !== null ? ` · ${countdown.toFixed(0)} s` : ""}{detail && countdown === null ? ` · ${detail}` : ""}
      </div>}
      {live && showSelf && <video ref={selfView} autoPlay playsInline muted className="absolute bottom-3 right-3 w-28 rounded border border-white/40 bg-black" />}
    </div>

    <div className="flex flex-wrap items-center gap-3">
      {!live
        ? <button className="rounded bg-ink-900 px-5 py-2 text-white disabled:opacity-40" disabled={!personaId && !image} onClick={start}>Start chat</button>
        : <>
          <button className="rounded px-5 py-2 border" onClick={() => channel.current?.readyState === "open" && channel.current.send("interrupt")}>Interrupt</button>
          <button className="rounded px-5 py-2 border" onClick={end}>End chat</button>
          <label className="text-sm flex items-center gap-1"><input type="checkbox" checked={showSelf} onChange={e => setShowSelf(e.target.checked)} /> show my camera</label>
        </>}
    </div>

    {!live && <div className="flex flex-wrap items-center gap-3 text-sm">
      {personas.length > 0 && <label>Who answers:
        <select className="ml-2 border px-2 py-1" value={personaId} onChange={e => { setPersonaId(e.target.value); const p = personas.find(x => x.id === e.target.value); if (p) setLanguage(p.language); }}>
          {personas.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          {uploadMode && <option value="">(uploaded portrait)</option>}
        </select></label>}
      <button className="underline" onClick={() => setWizard(true)}>Use this person…</button>
      <label>Language:
        <select className="ml-2 border px-2 py-1" value={language} onChange={e => setLanguage(e.target.value)}>
          {[["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"], ["it", "Italian"], ["pt", "Portuguese"], ["ja", "Japanese"], ["zh", "Chinese"]].map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
      <button className="underline text-ink-400" onClick={() => { setUploadMode(v => !v); if (!uploadMode) setPersonaId(""); }}>{uploadMode ? "use a saved person" : "or upload a portrait"}</button>
      {uploadMode && <input type="file" accept="image/png,image/jpeg,image/webp" onChange={e => setImage(e.target.files?.[0] || null)} />}
    </div>}

    {wizard && <PersonaWizard language={language} onCancel={() => setWizard(false)} onDone={p => { setWizard(false); setUploadMode(false); setPersonaId(p.id); void loadPersonas(); }} />}
    {error && <p role="alert" className="text-red-600">{error}</p>}
    <div aria-live="polite" className="space-y-1">{messages.map((m, i) => <p key={i}><span className="font-semibold">{m.who}:</span> {m.text}</p>)}</div>
    <p className="text-xs text-ink-400">Start chat turns on your microphone (and camera for your own preview only). The assistant answers after a short pause while its reply video renders.</p>
  </main>;
}
