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
  process.env.NEXT_PUBLIC_INGEST_BASE_URL || "http://localhost:8091";

type Mode = "fast" | "quality";

const MODE_LABELS: Record<Mode, string> = {
  fast: "Real-time (minutes, some quality traded for speed)",
  quality: "Batch (best quality, takes as long as it takes)",
};

export default function LivePage() {
  const [target, setTarget] = useState("es");
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
      pcRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  async function start() {
    setError(null);
    setJob(null);
    setPhase("connecting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 48000 },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      const pc = new RTCPeerConnection();
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
        }),
      });
      if (!resp.ok) throw new Error(`ingest offer failed: ${resp.status} ${await resp.text()}`);
      const answer = await resp.json();
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      setPhase("recording");
    } catch (e: any) {
      setError(e?.message ?? String(e));
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
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      teardown();
      setPhase("idle");
    }
  }

  function teardown() {
    pcRef.current?.close();
    pcRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    sessionRef.current = null;
  }

  const busy = phase !== "idle";

  return (
    <main className="mx-auto max-w-3xl p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Live capture</h1>
        <p className="text-sm text-gray-500">
          Webcam → WebRTC → GPU pipeline. Stop recording to submit the clip.
        </p>
      </header>

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
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    // Don't hang forever on a flaky network; partial candidates still work on LAN.
    setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    }, 2000);
  });
}
