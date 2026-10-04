"use client";
import { useEffect, useRef, useState } from "react";
import PersonaWizard from "../components/PersonaWizard";

const BASE = process.env.NEXT_PUBLIC_INGEST_BASE_URL || "/ingest";

type Schedule = { start_in: number; head_start: number; required: number; render_ratio: number; reply_seconds: number };

export default function AssistantPage() {
  const [image, setImage] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [voices, setVoices] = useState<string[]>([]);
  const [language, setLanguage] = useState("en");
  const [status, setStatus] = useState("idle");
  const [messages, setMessages] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [prepare, setPrepare] = useState<Record<string, unknown> | null>(null);
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [stats, setStats] = useState<{ stalls: number; frames_sent: number } | null>(null);
  const [personas, setPersonas] = useState<{ id: string; name: string }[]>([]);
  const [personaId, setPersonaId] = useState("");
  const [wizard, setWizard] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const peer = useRef<RTCPeerConnection | null>(null);
  const microphone = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const channel = useRef<RTCDataChannel | null>(null);
  const scheduledAt = useRef<number | null>(null);

  function stop() {
    if (session.current) void fetch(`${BASE}/assistant/sessions/${session.current}`, { method: "DELETE", keepalive: true });
    session.current = null;
    peer.current?.close(); peer.current = null;
    microphone.current?.getTracks().forEach(t => t.stop()); microphone.current = null;
    setStatus("idle"); setSchedule(null); setCountdown(null);
  }
  useEffect(() => () => stop(), []);
  async function loadPersonas() {
    try { const r = await fetch("/api/personas"); if (r.ok) setPersonas((await r.json()).personas); } catch { /* list stays empty */ }
  }
  useEffect(() => { void loadPersonas(); }, []);
  useEffect(() => {
    const timer = setInterval(() => {
      if (scheduledAt.current === null) return;
      const left = (scheduledAt.current - Date.now()) / 1000;
      setCountdown(left > 0 ? left : null);
      if (left <= 0) scheduledAt.current = null;
    }, 100);
    return () => clearInterval(timer);
  }, []);

  async function start() {
    if (!image && !personaId) return;
    setError(""); setStatus("preparing the assistant (portrait, idle motion, acknowledgement)"); setMessages([]); setStats(null);
    try {
      const form = new FormData(); form.append("language", language);
      if (personaId) form.append("persona_id", personaId); else { if (image) form.append("image", image); if (voice) form.append("voice", voice); }
      const created = await fetch(`${BASE}/assistant/sessions`, { method: "POST", body: form });
      if (!created.ok) throw new Error(await created.text());
      const body = await created.json();
      session.current = body.session_id; setPrepare(body.prepare);
      setStatus("connecting");
      const configResponse = await fetch(`${BASE}/config`);
      if (!configResponse.ok) throw new Error("Unable to load WebRTC configuration");
      const config = await configResponse.json();
      const pc = peer.current = new RTCPeerConnection({ iceServers: config.iceServers });
      const incoming = new MediaStream();
      pc.ontrack = event => { incoming.addTrack(event.track); if (video.current) video.current.srcObject = incoming; };
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "connected") setStatus("listening");
        if (pc.connectionState === "failed") { setError("Connection failed; check TURN settings"); stop(); }
      };
      const events = channel.current = pc.createDataChannel("events");
      events.onmessage = e => {
        const data = JSON.parse(e.data);
        switch (data.type) {
          case "transcript": setMessages(old => [...old.slice(-19), `You: ${data.text}`]); return;
          case "reply": setMessages(old => [...old.slice(-19), `Assistant: ${data.text}`]); return;
          case "reply_scheduled": setSchedule(data); scheduledAt.current = Date.now() + data.start_in * 1000; setStatus("reply rendering, acknowledgement and idle playing"); return;
          case "acknowledging": setStatus("acknowledging"); return;
          case "speaking": setStatus("speaking"); return;
          case "listening": setStatus("listening"); if (data.frames_sent !== undefined) setStats({ stalls: data.stalls, frames_sent: data.frames_sent }); scheduledAt.current = null; setCountdown(null); return;
          case "error": setError(data.message); return;
          default: setStatus(data.type);
        }
      };
      const mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      microphone.current = mic;
      mic.getTracks().forEach(track => pc.addTrack(track, mic));
      pc.addTransceiver("video", { direction: "recvonly" });
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => { pc.removeEventListener("icegatheringstatechange", check); reject(new Error("ICE gathering timed out")); }, 15000);
        const check = () => { if (pc.iceGatheringState === "complete") { clearTimeout(timeout); pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
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
      await video.current?.play().catch(() => setError("Press play to enable assistant audio."));
    } catch (e) { setError(String(e)); stop(); }
  }

  return <main className="mx-auto max-w-3xl p-6 space-y-4">
    <h1 className="text-2xl font-semibold">Video assistant</h1>
    <p>Choose a person, then speak. The assistant acknowledges at once, thinks, and its reply video starts after a short head start. Speak again or press Interrupt to cut it off.</p>
    <div className="flex flex-wrap gap-3 items-center">
      <select aria-label="Persona" value={personaId} disabled={status !== "idle"} onChange={e => setPersonaId(e.target.value)}>
        <option value="">Portrait upload + bundled voice</option>
        {personas.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
      </select>
      <button disabled={status !== "idle"} onClick={() => setWizard(true)}>Use this person…</button>
    </div>
    {wizard && <PersonaWizard language={language} onCancel={() => setWizard(false)} onDone={p => { setWizard(false); setPersonaId(p.id); void loadPersonas(); }} />}
    {!personaId && <input type="file" accept="image/png,image/jpeg,image/webp" disabled={status !== "idle"} onChange={e => setImage(e.target.files?.[0] || null)} />}
    <select value={language} disabled={status !== "idle"} onChange={e => setLanguage(e.target.value)}>
      {[["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"], ["it", "Italian"], ["pt", "Portuguese"], ["ja", "Japanese"], ["zh", "Chinese"]].map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select>
    <div>
      <button disabled={status !== "idle"} onClick={async () => { try { const r = await fetch("/api/speech/voices"); if (!r.ok) throw new Error(await r.text()); setVoices((await r.json()).voices); } catch (e) { setError(String(e)); } }}>Load voices</button>
      <select aria-label="Assistant voice" value={voice} onChange={e => setVoice(e.target.value)} disabled={status !== "idle"}><option value="">Default voice</option>{voices.map(v => <option key={v}>{v}</option>)}</select>
    </div>
    <video ref={video} autoPlay playsInline controls className="w-full rounded bg-black aspect-square max-h-[512px]" />
    <p aria-live="polite">{status}{countdown !== null && ` · reply starts in ${countdown.toFixed(1)} s`}</p>
    {schedule && <p className="text-sm text-ink-400">Head start {schedule.head_start.toFixed(1)} s (needed {schedule.required.toFixed(1)} s at render rate {schedule.render_ratio.toFixed(2)}× for a {schedule.reply_seconds.toFixed(1)} s reply)</p>}
    {stats && <p className="text-sm text-ink-400">Last reply: {stats.stalls} stalled frames of {stats.frames_sent} sent</p>}
    {prepare && <p className="text-sm text-ink-400">Prepared in {String((prepare as { total_seconds?: number }).total_seconds)} s</p>}
    <div className="flex gap-4">
      <button disabled={(!image && !personaId) || status !== "idle"} onClick={start}>Start conversation</button>
      <button disabled={status === "idle"} onClick={() => channel.current?.readyState === "open" && channel.current.send("interrupt")}>Interrupt</button>
      <button disabled={status === "idle"} onClick={stop}>End conversation</button>
    </div>
    {error && <p role="alert" className="text-red-600">{error}</p>}
    <div aria-live="polite">{messages.map((text, index) => <p key={index}>{text}</p>)}</div>
    <a href="/avatar" className="underline">Older avatar page</a> · <a href="/live" className="underline">Video translation</a>
  </main>;
}
