"use client";

/**
 * /live — webcam capture over WebRTC (GPU track).
 *
 * Streams the webcam to the ingest-webrtc service, which records it and,
 * on Stop, submits a backend job with the chosen mode's parameters. The
 * returned job then goes through the same PipelineView / ResultPlayer as
 * an uploaded file. See docs/gpu/README.md.
 */

import { useEffect, useRef, useState } from "react";
import { PipelineView } from "../components/PipelineView";
import { ResultPlayer } from "../components/ResultPlayer";
import { LanguagePicker } from "../components/LanguagePicker";
import { getJob, openJobEventStream, type JobRecord } from "../lib/api";

const INGEST_BASE_URL =
  process.env.NEXT_PUBLIC_INGEST_BASE_URL || "/ingest";

type Mode = "fast" | "quality" | "dub";

const MODE_LABELS: Record<Mode, string> = {
  fast: "Real-time (minutes, some quality traded for speed)",
  quality: "Batch (best quality, takes as long as it takes)",
  dub: "Fastest (translated audio, original video)",
};

export default function LivePage() {
  const [target, setTarget] = useState("es");
  const [captions,setCaptions]=useState(false);
  const [caption,setCaption]=useState("");
  const [elapsed,setElapsed]=useState(0);
  const [limit,setLimit]=useState(120);
  const [connection,setConnection]=useState("");
  const [mode, setMode] = useState<Mode>("fast");
  const [phase, setPhase] = useState<"idle" | "connecting" | "recording" | "submitting">("idle");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      esRef.current?.close();
      if (sessionRef.current) void fetch(`${INGEST_BASE_URL}/sessions/${sessionRef.current}`, { method: "DELETE", keepalive: true });
      pcRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(()=>{
    if(phase!=="recording")return;
    const timer=setInterval(async()=>{
      const id=sessionRef.current;if(!id)return;
      try{const r=await fetch(`${INGEST_BASE_URL}/sessions/${id}`);if(r.ok){const data=await r.json();setElapsed(data.seconds??0);setCaption(data.caption||"");if(data.error)setError(data.error);}}catch{}
      const pc=pcRef.current;if(pc){const stats=await pc.getStats();stats.forEach(report=>{if(report.type==="candidate-pair"&&report.state==="succeeded")setConnection(`Connected · round trip ${Math.round((report.currentRoundTripTime||0)*1000)} ms`);});}
    },1000);return()=>clearInterval(timer);
  },[phase]);

  async function start() {
    setError(null);
    esRef.current?.close();
    setJob(null);
    setPhase("connecting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: mode === "quality" ? 1280 : 640 }, height: { ideal: mode === "quality" ? 720 : 360 }, frameRate: { ideal: 25, max: 25 } },
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 48000 },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      const configResponse = await fetch(`${INGEST_BASE_URL}/config`);
      if (!configResponse.ok) throw new Error("Unable to load recording configuration");
      const config = await configResponse.json();
      setLimit(config.maxSeconds);setElapsed(0);setCaption("");
      const pc = new RTCPeerConnection({ iceServers: config.iceServers });
      pcRef.current = pc;
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGathering(pc);

      const sessionId = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
      sessionRef.current = sessionId;
      const resp = await fetch(`${INGEST_BASE_URL}/sessions/${sessionId}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription!.sdp,
          type: pc.localDescription!.type,
          target_language: target,
          mode,
          live_captions:captions,
        }),
      });
      if (!resp.ok) throw new Error(`ingest offer failed: ${resp.status} ${await resp.text()}`);
      const answer = await resp.json();
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      await waitForConnection(pc);
      const started = Date.now();
      while (true) {
        const stats = await pc.getStats();
        let sent = false;
        stats.forEach(report => { if (report.type === "outbound-rtp" && report.kind === "video" && report.packetsSent > 0) sent = true; });
        if (sent) break;
        if (Date.now() - started > 5000) throw new Error("Camera connected but no video packets were sent");
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      setPhase("recording");
    } catch (e: any) {
      setError(e?.message ?? String(e));
      if (sessionRef.current) void fetch(`${INGEST_BASE_URL}/sessions/${sessionRef.current}`, { method: "DELETE" });
      teardown();
      setPhase("idle");
    }
  }

  async function stop() {
    const sessionId = sessionRef.current;
    if (!sessionId) return;
    setPhase("submitting");
    try {
      const resp = await fetch(`${INGEST_BASE_URL}/sessions/${sessionId}/stop`, { method: "POST" });
      if (!resp.ok) throw new Error(`ingest stop failed: ${resp.status} ${await resp.text()}`);
      const { job_id } = await resp.json();
      const initial = await getJob(job_id);
      setJob(initial);
      esRef.current?.close();
      esRef.current = openJobEventStream(job_id, async (eventName, data) => {
        if (eventName === "stage_progress") {
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  stages: prev.stages.map((s) =>
                    s.name === data.stage ? { ...s, progress: data.percent } : s
                  ),
                }
              : prev
          );
          return;
        }
        try {
          setJob(await getJob(job_id));
        } catch {
          /* tolerate transient errors */
        }
      });
      teardown();
      setPhase("idle");
    } catch (e: any) {
      setError(e?.message ?? String(e));
      teardown(false);
      setPhase("recording");
    } finally {
      // A failed submission retains the session so Stop can be retried.
      if (sessionRef.current) setPhase("recording");
    }
  }

  function teardown(clearSession = true) {
    pcRef.current?.close();
    pcRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (clearSession) sessionRef.current = null;
  }

  const busy = phase !== "idle";

  return (
    <main className="mx-auto max-w-3xl p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Live capture</h1>
        <p className="text-sm text-gray-500">
          Webcam → WebRTC → GPU pipeline. Stop recording to submit the clip.
          <a href="/avatar" className="ml-3 underline">Live voice avatar</a>
        </p>
      </header>
      <a href="/studio" className="underline">Jobs and editor</a>
      <label className="block"><input type="checkbox" checked={captions} disabled={busy} onChange={e=>setCaptions(e.target.checked)}/> Provisional captions during capture</label>
      {phase==="recording"&&<p>{Math.floor(elapsed)} / {limit} seconds · {connection}</p>}
      {caption&&<p aria-live="polite">{caption}</p>}

      <video ref={videoRef} autoPlay muted playsInline className="w-full rounded bg-black aspect-video" />

      <div className="grid gap-4 sm:grid-cols-2">
        <LanguagePicker value={target} onChange={setTarget} />
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">Mode</span>
          <select
            className="rounded border p-2"
            value={mode}
            disabled={busy}
            onChange={(e) => setMode(e.target.value as Mode)}
          >
            {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex gap-3">
        <button
          className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
          onClick={start}
          disabled={busy}
        >
          {phase === "connecting" ? "Connecting…" : "Start recording"}
        </button>
        <button
          className="rounded bg-red-600 px-4 py-2 text-white disabled:opacity-50"
          onClick={stop}
          disabled={phase !== "recording"}
        >
          {phase === "submitting" ? "Submitting…" : "Stop & translate"}
        </button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {job && (
        <>
          <PipelineView job={job} />
          {job.status === "completed" && <ResultPlayer job={job} />}
        </>
      )}
    </main>
  );
}

function waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => { clearTimeout(timer); pc.removeEventListener("icegatheringstatechange", check); error ? reject(error) : resolve(); };
    const check = () => { if (pc.iceGatheringState === "complete") finish(); };
    const timer = setTimeout(() => finish(new Error("ICE gathering timed out; check TURN settings")), 15000);
    pc.addEventListener("icegatheringstatechange", check); check();
  });
}

function waitForConnection(pc: RTCPeerConnection): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error("Media connection timed out; check TURN settings")), 20000);
    const finish = (error?: Error) => {
      clearTimeout(timeout);
      pc.removeEventListener("connectionstatechange", check);
      error ? reject(error) : resolve();
    };
    const check = () => {
      if (pc.connectionState === "connected") finish();
      else if (["failed", "closed"].includes(pc.connectionState)) finish(new Error("Media connection failed"));
    };
    pc.addEventListener("connectionstatechange", check);
    check();
  });
}
