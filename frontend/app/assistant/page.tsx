"use client";

import { useEffect, useRef, useState } from "react";
import PersonaWizard from "../components/PersonaWizard";
import AssistantIcon from "./AssistantIcon";
import "./assistant.css";

const BASE = process.env.NEXT_PUBLIC_INGEST_BASE_URL || "/ingest";
const LANGUAGES = [["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"], ["it", "Italian"], ["pt", "Portuguese"], ["ja", "Japanese"], ["zh", "Chinese"]] as const;
type Persona = { id: string; name: string; language: string };
type Phase = "idle" | "preparing" | "connecting" | "live";
type Panel = "settings" | "wizard" | null;
type Reception = { fps: number; lost: number; pli: number; nack: number; jitter: number; codec: string; dropped: number };

export default function AssistantPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [state, setState] = useState("");
  const [detail, setDetail] = useState("");
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personaId, setPersonaId] = useState("");
  const [language, setLanguage] = useState("en");
  const [panel, setPanel] = useState<Panel>(null);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const [uploadMode, setUploadMode] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [messages, setMessages] = useState<{ who: string; text: string }[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [cameraError, setCameraError] = useState("");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [showSelf, setShowSelf] = useState(true);
  const [micOn, setMicOn] = useState(true);
  const [rx, setRx] = useState<Reception | null>(null);
  const rxPrev = useRef<{ frames: number; t: number } | null>(null);
  const stage = useRef<HTMLVideoElement>(null);
  const selfView = useRef<HTMLVideoElement>(null);
  const peer = useRef<RTCPeerConnection | null>(null);
  const media = useRef<MediaStream | null>(null);
  const camera = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const channel = useRef<RTCDataChannel | null>(null);
  const replyAt = useRef<number | null>(null);
  const active = phase !== "idle";
  const connected = phase === "live";
  const persona = personas.find(p => p.id === personaId);

  async function loadPersonas(preferredId?: string) {
    try {
      const response = await fetch("/api/personas");
      if (!response.ok) return;
      const list: Persona[] = (await response.json()).personas;
      setPersonas(list);
      if (preferredId) {
        const selected = list.find(p => p.id === preferredId);
        if (selected) { setPersonaId(selected.id); setLanguage(selected.language); }
      } else if (list[0]) {
        setPersonaId(current => current || list[0].id);
        setLanguage(list[0].language);
      }
    } catch { /* The empty state remains usable for a new face or portrait upload. */ }
  }

  useEffect(() => { void loadPersonas(); }, []);
  useEffect(() => () => end(), []);
  useEffect(() => {
    if (!image) { setImagePreview(""); return; }
    const url = URL.createObjectURL(image);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);
  useEffect(() => {
    if (!panel) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const drawer = document.querySelector<HTMLElement>(".aurora-drawer");
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    drawer?.querySelector<HTMLElement>("button")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setPanel(null); return; }
      if (event.key !== "Tab" || !drawer) return;
      const controls = Array.from(drawer.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])"))
        .filter(element => element.getClientRects().length > 0);
      if (!controls.length) return;
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = oldOverflow; previous?.focus(); };
  }, [panel]);
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (replyAt.current === null) return;
      const remaining = (replyAt.current - Date.now()) / 1000;
      if (remaining <= 0) { replyAt.current = null; setCountdown(null); }
      else setCountdown(remaining);
    }, 100);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    const timer = window.setInterval(async () => {
      const pc = peer.current;
      if (!pc || pc.connectionState !== "connected") return;
      try {
        const report = await pc.getStats();
        const codecs: Record<string, string> = {};
        report.forEach(item => { if (item.type === "codec") codecs[item.id] = item.mimeType; });
        report.forEach(item => {
          if (item.type !== "inbound-rtp" || item.kind !== "video") return;
          const now = Date.now();
          const previous = rxPrev.current;
          const fps = previous ? ((item.framesDecoded - previous.frames) * 1000) / Math.max(1, now - previous.t) : 0;
          rxPrev.current = { frames: item.framesDecoded, t: now };
          setRx({ fps: Math.round(fps * 10) / 10, lost: item.packetsLost ?? 0, pli: item.pliCount ?? 0, nack: item.nackCount ?? 0, jitter: Math.round((item.jitter ?? 0) * 1000), codec: codecs[item.codecId] || "", dropped: item.framesDropped ?? 0 });
        });
      } catch { /* Stats are optional and must never interrupt a call. */ }
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    let cancelled = false;
    if (!connected || !showSelf) {
      camera.current?.getTracks().forEach(track => track.stop());
      camera.current = null;
      if (selfView.current) selfView.current.srcObject = null;
      return;
    }
    void navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240, facingMode: "user" }, audio: false })
      .then(stream => {
        if (cancelled) { stream.getTracks().forEach(track => track.stop()); return; }
        camera.current = stream;
        if (selfView.current) { selfView.current.srcObject = stream; void selfView.current.play().catch(() => undefined); }
        setCameraError("");
      })
      .catch(() => { if (!cancelled) { setCameraError("Camera preview unavailable"); setShowSelf(false); } });
    return () => { cancelled = true; camera.current?.getTracks().forEach(track => track.stop()); camera.current = null; };
  }, [connected, showSelf]);

  function end() {
    const id = session.current;
    session.current = null;
    if (id) void fetch(`${BASE}/assistant/sessions/${id}`, { method: "DELETE", keepalive: true });
    peer.current?.close(); peer.current = null;
    channel.current = null;
    media.current?.getTracks().forEach(track => track.stop()); media.current = null;
    camera.current?.getTracks().forEach(track => track.stop()); camera.current = null;
    if (stage.current) stage.current.srcObject = null;
    if (selfView.current) selfView.current.srcObject = null;
    replyAt.current = null;
    rxPrev.current = null;
    setCountdown(null); setRx(null); setPhase("idle"); setState(""); setDetail(""); setMicOn(true);
  }

  async function start() {
    if (!personaId && !image) { setPanel("settings"); return; }
    setError(""); setMessages([]); setNotes([]); setCameraError("");
    setPhase("preparing"); setState(`Preparing ${persona?.name ?? "the assistant"}…`);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
      media.current = stream;
      const form = new FormData(); form.append("language", language);
      if (personaId) form.append("persona_id", personaId);
      else if (image) form.append("image", image);
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
      events.onmessage = event => {
        let data: Record<string, any>;
        try { data = JSON.parse(event.data); } catch { return; }
        switch (data.type) {
          case "transcript": setMessages(old => [...old.slice(-19), { who: "You", text: String(data.text) }]); return;
          case "reply": setMessages(old => [...old.slice(-19), { who: persona?.name ?? "Assistant", text: String(data.text) }]); return;
          case "thinking": setState("Thinking…"); setDetail(""); setNotes([]); return;
          case "acknowledging": setDetail(data.pending ? "Acknowledgement rendering" : "Acknowledging"); return;
          case "filler_ready": if (data.kind === "opener" && data.count === 1) setDetail("Acknowledgement ready"); return;
          case "working": setState("Looking it up…"); return;
          case "filler": if (data.kind !== "opener") setDetail(`“${data.text}”`); return;
          case "speech_check": setNotes(old => [...old.slice(-2), `“${data.text}”: ${data.fallback ? "stock voice used after cloned takes failed" : `take ${data.takes}, ${Math.round(data.match * 100)}% words matched`}`]); return;
          case "reply_scheduled": replyAt.current = Date.now() + data.start_in * 1000; setState("Thinking…"); setDetail("Preparing a reply"); return;
          case "speaking": replyAt.current = null; setCountdown(null); setState("Speaking"); setDetail("Speak to interrupt."); return;
          case "listening": setState("Listening"); setDetail(""); replyAt.current = null; setCountdown(null); return;
          case "ready": setDetail("Ready"); return;
          case "error": setError(String(data.message)); return;
          default: return;
        }
      };
      stream.getAudioTracks().forEach(track => pc.addTrack(track, stream));
      pc.addTransceiver("video", { direction: "recvonly" });
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise<void>(resolve => {
        const done = () => { pc.removeEventListener("icegatheringstatechange", check); resolve(); };
        const timeout = window.setTimeout(done, 4000);
        const check = () => { if (pc.iceGatheringState === "complete") { window.clearTimeout(timeout); done(); } };
        pc.addEventListener("icegatheringstatechange", check); check();
      });
      const response = await fetch(`${BASE}/assistant/sessions/${session.current}/offer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(pc.localDescription) });
      if (!response.ok) throw new Error(await response.text());
      await pc.setRemoteDescription(await response.json());
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => { pc.removeEventListener("connectionstatechange", check); reject(new Error("Media connection timed out; check TURN settings")); }, 20000);
        const check = () => {
          if (["connected", "failed", "closed"].includes(pc.connectionState)) {
            window.clearTimeout(timeout); pc.removeEventListener("connectionstatechange", check);
            if (pc.connectionState === "connected") resolve(); else reject(new Error("Media connection failed"));
          }
        };
        pc.addEventListener("connectionstatechange", check); check();
      });
      await stage.current?.play().catch(() => setError("Press play on the video to enable sound."));
    } catch (cause) { setError(String(cause)); end(); }
  }

  function toggleMic() {
    const next = !micOn;
    media.current?.getAudioTracks().forEach(track => { track.enabled = next; });
    setMicOn(next);
  }

  const portrait = uploadMode && imagePreview ? imagePreview : personaId ? `/api/personas/${encodeURIComponent(personaId)}/portrait` : "";
  const displayName = persona?.name ?? (uploadMode && image ? image.name.replace(/\.[^.]+$/, "") : "Video assistant");
  const status = active ? state || "Preparing…" : personaId || image ? "Your assistant is ready" : "Create a face to begin";
  const subtitle = active ? (countdown !== null ? `Reply in ${Math.ceil(countdown)} s` : detail) : personaId || image ? "Press Start to begin talking" : "Use the person icon to set up your assistant";

  return <main className="aurora-assistant">
    <header className="aurora-header">
      <div className="aurora-brand"><span className="aurora-brand-mark">D</span><strong>Dell Technologies</strong><span className="aurora-brand-rule" /><b>intel</b><small>video assistant</small></div>
      <div className="aurora-tools">
        <span className="aurora-ready"><i />{phase === "live" ? "Connected" : active ? "Connecting" : "Ready"}</span>
        <button className={`aurora-icon-button ${transcriptOpen ? "is-selected" : ""}`} onClick={() => setTranscriptOpen(value => !value)} aria-label={transcriptOpen ? "Hide transcript" : "Show transcript"} aria-pressed={transcriptOpen} title="Transcript"><AssistantIcon name="transcript" /></button>
        <button className="aurora-icon-button" onClick={() => setPanel("wizard")} aria-label="Create a new face" title="Create a new face"><AssistantIcon name="person" /></button>
        <button className="aurora-icon-button" onClick={() => setPanel("settings")} aria-label="Open settings" title="Settings"><AssistantIcon name="settings" /></button>
      </div>
    </header>

    <div className={`aurora-workspace ${transcriptOpen ? "has-transcript" : ""}`}>
      <section className="aurora-call" aria-label="Video assistant">
        <div className="aurora-stage">
          {portrait && <img className="aurora-portrait" src={portrait} alt={displayName} />}
          <video ref={stage} autoPlay playsInline className={`aurora-video ${connected ? "is-active" : ""}`} aria-label="Assistant video" />
          <div className="aurora-stage-sheen" />
          <div className="aurora-stage-top"><span className="aurora-persona-pill"><i /><span><strong>{displayName}</strong><small>{phase === "live" ? "In conversation" : active ? "Connecting" : "Ready to talk"}</small></span></span>{active && <span className="aurora-hd">LIVE</span>}</div>
          <div className="aurora-stage-bottom"><div className="aurora-status"><span className="aurora-orb">⌁</span><span><strong aria-live="polite">{status}</strong><small>{subtitle}</small></span></div>{active && showSelf && <video ref={selfView} autoPlay playsInline muted className="aurora-self-view" aria-label="Your camera preview" />}</div>
        </div>
        <div className="aurora-dock">
          <button className={`aurora-round-button ${!micOn ? "is-off" : ""}`} disabled={!active} onClick={toggleMic} aria-label={micOn ? "Mute microphone" : "Unmute microphone"} title={micOn ? "Mute microphone" : "Unmute microphone"}><AssistantIcon name="microphone" /></button>
          <button className={`aurora-call-button ${active ? "is-ending" : ""}`} onClick={() => { if (active) end(); else void start(); }} disabled={!active && !personaId && !image}><span className="aurora-call-dot" />{active ? "End conversation" : "Start conversation"}</button>
          <button className="aurora-round-button" disabled={phase !== "live"} onClick={() => { if (channel.current?.readyState === "open") channel.current.send("interrupt"); }} aria-label="Interrupt assistant" title="Interrupt assistant"><AssistantIcon name="interrupt" /></button>
          <button className={`aurora-round-button ${!showSelf ? "is-off" : ""}`} disabled={!active} onClick={() => setShowSelf(value => !value)} aria-label={showSelf ? "Hide camera preview" : "Show camera preview"} title={showSelf ? "Hide camera preview" : "Show camera preview"}><AssistantIcon name="camera" /></button>
        </div>
        {error && <div className="aurora-error" role="alert">{error}</div>}
        {cameraError && <div className="aurora-camera-note" role="status">{cameraError}</div>}
        {!active && !personaId && !image && <button className="aurora-setup-link" onClick={() => setPanel("wizard")}>Create a new face</button>}
      </section>

      {transcriptOpen && <aside className="aurora-transcript" aria-label="Conversation transcript"><div className="aurora-transcript-head"><div><small>CONVERSATION</small><h2>Transcript</h2></div><button className="aurora-icon-button" onClick={() => setTranscriptOpen(false)} aria-label="Hide transcript"><AssistantIcon name="close" /></button></div><div className="aurora-messages" aria-live="polite">{messages.length ? messages.map((message, index) => <div key={index} className={`aurora-message ${message.who === "You" ? "is-user" : "is-assistant"}`}><small>{message.who}</small><p>{message.text}</p></div>) : <p className="aurora-empty-transcript">Your conversation will appear here.</p>}</div><div className="aurora-transcript-foot">{active ? "Live transcript" : "Start a conversation to see a transcript"}</div></aside>}
    </div>

    {panel && <><button className="aurora-backdrop" onClick={() => setPanel(null)} aria-label="Close panel" /><aside className="aurora-drawer" role="dialog" aria-modal="true" aria-label={panel === "settings" ? "Assistant settings" : "New face wizard"}>
      {panel === "settings" ? <><div className="aurora-drawer-head"><div><small>PREFERENCES</small><h2>Settings</h2></div><button className="aurora-icon-button" onClick={() => setPanel(null)} aria-label="Close settings"><AssistantIcon name="close" /></button></div>
        <div className="aurora-setting"><label htmlFor="aurora-persona">Assistant</label><select id="aurora-persona" value={uploadMode ? "upload" : personaId} disabled={active} onChange={event => { if (event.target.value === "upload") { setUploadMode(true); setPersonaId(""); } else { setUploadMode(false); setPersonaId(event.target.value); const choice = personas.find(p => p.id === event.target.value); if (choice) setLanguage(choice.language); } }}><option value="" disabled>Choose a person</option>{personas.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}<option value="upload">Upload a portrait</option></select></div>
        {uploadMode && <div className="aurora-setting"><label htmlFor="aurora-upload">Portrait image</label><input id="aurora-upload" type="file" accept="image/png,image/jpeg,image/webp" disabled={active} onChange={event => setImage(event.target.files?.[0] || null)} /></div>}
        <div className="aurora-setting"><label htmlFor="aurora-language">Conversation language</label><select id="aurora-language" value={language} disabled={active} onChange={event => setLanguage(event.target.value)}>{LANGUAGES.map(([code, name]) => <option key={code} value={code}>{name}</option>)}</select></div>
        <label className="aurora-setting aurora-setting-toggle"><span>Show my camera preview</span><input type="checkbox" checked={showSelf} onChange={event => setShowSelf(event.target.checked)} /></label>
        <button className="aurora-new-face-link" onClick={() => setPanel("wizard")}><AssistantIcon name="person" /> Create a new face</button>
        <details className="aurora-diagnostics"><summary>Connection details</summary>{rx ? <p>Video {rx.codec.replace("video/", "")} · {rx.fps} fps decoded · {rx.lost} lost packets · {rx.pli} picture-loss requests · {rx.nack} retransmits · {rx.jitter} ms jitter · {rx.dropped} dropped frames</p> : <p>No active video connection.</p>}{notes.length > 0 && <ul>{notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}</details>
        <p className="aurora-privacy">Your camera is only used for your local preview. The assistant receives your microphone.</p>
      </> : <><div className="aurora-drawer-head"><div><small>NEW FACE</small><h2>Create an assistant</h2></div><button className="aurora-icon-button" onClick={() => setPanel(null)} aria-label="Close face wizard"><AssistantIcon name="close" /></button></div><PersonaWizard language={language} onDone={created => { setPanel(null); setUploadMode(false); setImage(null); void loadPersonas(created.id); }} /></>}
    </aside></>}
  </main>;
}
