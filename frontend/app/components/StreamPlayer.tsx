"use client";

import { useEffect, useRef, useState } from "react";
import { artifactUrl } from "../lib/api";

/** Plays a streaming-translation job while it is still rendering.
 *
 *  The backend publishes each finished piece as an HLS segment and, with every
 *  `stream_segment` event, the head-start numbers: how much media is ready, how
 *  much remains, and the production rate. Playback from the start cannot stall
 *  once ready >= (1 - rate) * remaining + margin, so the player waits for the
 *  backend's `can_play` before starting unless the viewer presses play early. */
export type StreamProgress = {
  ready_seconds: number; total_seconds: number; rate: number; can_play: boolean; index: number; kind: string;
};

export function StreamPlayer({ jobId, progress, ended }: { jobId: string; progress: StreamProgress | null; ended: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const hls = useRef<any>(null);
  const [attached, setAttached] = useState(false);
  const [started, setStarted] = useState(false);
  const url = artifactUrl(jobId, "stream.m3u8");

  useEffect(() => {
    const element = video.current;
    if (!element || attached || !progress) return;
    let cancelled = false;
    (async () => {
      if (element.canPlayType("application/vnd.apple.mpegurl")) {
        element.src = url;
        setAttached(true);
        return;
      }
      const Hls = (await import("hls.js")).default;
      if (cancelled || !Hls.isSupported()) return;
      const instance = new Hls({ lowLatencyMode: false, maxBufferLength: 60, liveSyncDuration: 0, startPosition: 0 });
      instance.loadSource(url);
      instance.attachMedia(element);
      hls.current = instance;
      setAttached(true);
    })();
    return () => { cancelled = true; };
  }, [attached, progress, url]);

  useEffect(() => () => { hls.current?.destroy(); hls.current = null; }, []);

  useEffect(() => {
    // Start when the head-start rule says playback cannot stall (or when everything is in).
    if (!attached || started || !progress) return;
    if (progress.can_play || ended) {
      setStarted(true);
      void video.current?.play().catch(() => undefined);
    }
  }, [attached, started, progress, ended]);

  if (!progress) return null;
  const ready = progress.ready_seconds, total = progress.total_seconds;
  const remaining = Math.max(0, total - ready);
  const needed = Math.max(0, remaining * (1 - Math.min(progress.rate, 1)) + 2);
  return <section className="mt-8">
    <h2 className="text-sm uppercase tracking-wide text-ink-400 mb-3">Translated video, live</h2>
    <div className="border border-ink-600 rounded-lg bg-ink-800 p-4">
      <video ref={video} controls playsInline className="w-full rounded border border-ink-700 bg-black" />
      <div className="mt-3 h-2 rounded bg-ink-700 overflow-hidden" aria-label="rendered so far">
        <div className="h-full bg-emerald-600 transition-all" style={{ width: `${total ? Math.min(100, (ready / total) * 100) : 0}%` }} />
      </div>
      <p className="mt-2 text-sm text-ink-400">
        {ready.toFixed(1)} of {total.toFixed(1)} s ready · producing {progress.rate.toFixed(2)} s per second ·{" "}
        {started ? "playing" : ended ? "complete" : progress.can_play ? "can play without stalling" : `waiting for ${needed.toFixed(0)} s of buffer so playback never stalls`}
        {!started && !ended && <button className="ml-3 underline text-accent-soft" onClick={() => { setStarted(true); void video.current?.play().catch(() => undefined); }}>play now anyway</button>}
      </p>
    </div>
  </section>;
}
