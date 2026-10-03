"use client";
import { useEffect, useRef, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_INGEST_BASE_URL || "/ingest";
export default function AvatarPage() {
  const [image, setImage] = useState<File | null>(null);
  const [voice,setVoice]=useState("");
  const [voices,setVoices]=useState<string[]>([]);
  const [language, setLanguage] = useState("en");
  const [status, setStatus] = useState("idle");
  const [messages, setMessages] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [partial,setPartial]=useState("");
  const [latency,setLatency]=useState<number|null>(null);
  const acknowledgements=useRef<{token:string;end:number}[]>([]);
  const video = useRef<HTMLVideoElement>(null);
  const peer = useRef<RTCPeerConnection | null>(null);
  const microphone = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const channel = useRef<RTCDataChannel | null>(null);

  function stop() {
    if (session.current) void fetch(`${BASE}/avatar/sessions/${session.current}`, { method: "DELETE", keepalive: true });
    session.current = null;
    acknowledgements.current=[];
    peer.current?.close(); peer.current = null;
    microphone.current?.getTracks().forEach(t => t.stop()); microphone.current = null;
    setStatus("idle");
  }
  useEffect(() => () => stop(), []);

  async function start() {
    if (!image) return;
    setError(""); setStatus("connecting"); setMessages([]);
    try {
      const form = new FormData(); form.append("image", image); form.append("language", language); if(voice)form.append("voice",voice);
      const created = await fetch(`${BASE}/avatar/sessions`, { method: "POST", body: form });
      if (!created.ok) throw new Error(await created.text());
      session.current = (await created.json()).session_id;
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
        if (data.type === "playout") { acknowledgements.current.push({token:data.token,end:data.end}); return; }
        if (data.type === "partial_transcript") { setPartial(data.text); return; }
        if (data.type === "latency") { setLatency(data.seconds); return; }
        if (data.type === "degraded") { setError(data.message); return; }
        if (data.type === "listening") acknowledgements.current=[];
        if (data.type === "error") setError(data.message);
        else if (data.type === "transcript" || data.type === "reply")
          setMessages(old => [...old.slice(-19), `${data.type === "transcript" ? "You" : "Assistant"}: ${data.text}`]);
        else setStatus(data.type);
      };
      const mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      microphone.current = mic;
      mic.getTracks().forEach(track => pc.addTrack(track, mic));
      pc.addTransceiver("video", { direction: "recvonly" });
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => { pc.removeEventListener("icegatheringstatechange", check); reject(new Error("ICE gathering timed out")); },15000);
        const check = () => { if (pc.iceGatheringState === "complete") { clearTimeout(timeout); pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
        pc.addEventListener("icegatheringstatechange", check); check();
      });
      const response = await fetch(`${BASE}/avatar/sessions/${session.current}/offer`, {
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
      await video.current?.play().catch(() => setError("Press play to enable avatar audio."));
    } catch (e) { setError(String(e)); stop(); }
  }
  return <main className="mx-auto max-w-3xl p-6 space-y-4">
    <h1 className="text-2xl font-semibold">Live voice avatar</h1>
    <a href="/studio" className="underline">Jobs and editor</a>
    {partial&&<p className="text-ink-400">Hearing: {partial}</p>}
    {latency!==null&&<p className="text-sm">Last response started in {latency.toFixed(2)} seconds</p>}
    <p>Choose a portrait, then speak to the assistant. Speak again or press Interrupt to stop its reply.</p>
    <input type="file" accept="image/png,image/jpeg,image/webp" disabled={status !== "idle"} onChange={e => setImage(e.target.files?.[0] || null)} />
    <select value={language} disabled={status !== "idle"} onChange={e => setLanguage(e.target.value)}>
      {[["en","English"],["es","Spanish"],["fr","French"],["de","German"],["ja","Japanese"],["zh","Chinese"]].map(([value,label]) => <option key={value} value={value}>{label}</option>)}
    </select>
    <div><button disabled={status!=="idle"} onClick={async()=>{try{const r=await fetch('/api/speech/voices');if(!r.ok)throw new Error(await r.text());setVoices((await r.json()).voices);}catch(e){setError(String(e));}}}>Load voices</button><select aria-label="Avatar voice" value={voice} onChange={e=>setVoice(e.target.value)} disabled={status!=="idle"}><option value="">Default voice</option>{voices.map(v=><option key={v}>{v}</option>)}</select></div>
    <video onTimeUpdate={()=>{const clock=video.current?.currentTime??0;acknowledgements.current=acknowledgements.current.filter(item=>{if(clock>=item.end){if(channel.current?.readyState==='open')channel.current.send(JSON.stringify({type:'played',token:item.token}));return false;}return true;});}} ref={video} autoPlay playsInline controls className="w-full rounded bg-black aspect-video" />
    <p aria-live="polite">{status}</p>
    <div className="flex gap-4">
      <button disabled={!image || status !== "idle"} onClick={start}>Start conversation</button>
      <button disabled={status === "idle"} onClick={() => channel.current?.readyState === "open" && channel.current.send("interrupt")}>Interrupt</button>
      <button disabled={status === "idle"} onClick={stop}>End conversation</button>
    </div>
    {error && <p role="alert" className="text-red-600">{error}</p>}
    <div aria-live="polite">{messages.map((text,index) => <p key={index}>{text}</p>)}</div>
    <a href="/live" className="underline">Video translation</a>
  </main>;
}
