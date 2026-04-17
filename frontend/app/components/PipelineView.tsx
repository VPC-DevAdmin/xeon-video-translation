"use client";

import { artifactUrl } from "../lib/api";
import type { JobRecord, StageName, StageStatus } from "../lib/api";

interface StageMeta {
  key: StageName;
  title: string;
  blurb: string;
  badge?: string;
}

const STAGES: StageMeta[] = [
  { key: "audio", title: "Audio extract", blurb: "ffmpeg → 16 kHz mono WAV" },
  { key: "transcribe", title: "Transcribe", blurb: "faster-whisper (CPU int8)" },
  { key: "translate", title: "Translate", blurb: "NLLB-200 distilled-600M" },
  { key: "tts", title: "Voice clone", blurb: "XTTS-v2 (CPU)" },
  { key: "lipsync", title: "Lip sync", blurb: "selectable backend" },
  { key: "mux", title: "Mux & watermark", blurb: "ffmpeg + overlay" },
];

const STATUS_LABEL: Record<StageStatus, string> = {
  pending: "Pending",
  running: "Running…",
  done: "Done",
  failed: "Failed",
  skipped: "Skipped",
};

const STATUS_COLOR: Record<StageStatus, string> = {
  pending: "bg-ink-600 text-ink-300",
  running: "bg-accent text-white animate-pulse",
  done: "bg-emerald-600 text-white",
  failed: "bg-red-600 text-white",
  skipped: "bg-ink-700 text-ink-400",
};

export function PipelineView({ job }: { job: JobRecord | null }) {
  const byName = new Map<StageName, JobRecord["stages"][number]>();
  if (job) {
    for (const s of job.stages) byName.set(s.name, s);
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {STAGES.map((meta) => {
        const stage = byName.get(meta.key);
        const status: StageStatus = stage?.status ?? "pending";
        return (
          <div
            key={meta.key}
            className="stage-card border border-ink-600 rounded-lg p-4 bg-ink-800"
          >
            <div className="flex items-center justify-between">
              <div className="font-medium text-ink-200">{meta.title}</div>
              <span
                className={`text-xs px-2 py-0.5 rounded ${STATUS_COLOR[status]}`}
              >
                {STATUS_LABEL[status]}
              </span>
            </div>
            <div className="text-xs text-ink-400 mt-1 flex gap-2 items-center">
              <span>{meta.blurb}</span>
              {meta.badge && (
                <span className="px-1.5 py-0.5 rounded bg-ink-700 text-ink-300 text-[10px] uppercase">
                  {meta.badge}
                </span>
              )}
            </div>

            <ProgressAndEta stage={stage} />

            {stage?.duration_ms != null && stage?.status === "done" && (
              <div className="text-xs text-ink-400 mt-2">
                {(stage.duration_ms / 1000).toFixed(2)}s
              </div>
            )}

            {stage?.error && (
              <pre className="text-xs text-red-300 mt-2 whitespace-pre-wrap">
                {stage.error}
              </pre>
            )}

            <StageOutput
              name={meta.key}
              output={stage?.output}
              jobId={job?.job_id}
            />
          </div>
        );
      })}
    </div>
  );
}

function ProgressAndEta({
  stage,
}: {
  stage: JobRecord["stages"][number] | undefined;
}) {
  if (!stage) return null;
  if (stage.status !== "running") return null;

  const pct =
    stage.progress != null
      ? Math.round(stage.progress * 100)
      : null;
  const eta =
    stage.eta_seconds != null
      ? `~${formatDuration(stage.eta_seconds)}`
      : null;

  return (
    <div className="mt-3">
      <div className="h-1.5 w-full rounded bg-ink-700 overflow-hidden">
        {pct != null ? (
          <div
            className="h-full bg-accent transition-all"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full bg-accent/60 animate-pulse w-1/3" />
        )}
      </div>
      <div className="text-[11px] text-ink-400 mt-1 flex justify-between">
        <span>{pct != null ? `${pct}%` : "working…"}</span>
        {eta && <span>{eta}</span>}
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.ceil(seconds - m * 60);
  return `${m}m ${s}s`;
}

function StageOutput({
  name,
  output,
  jobId,
}: {
  name: StageName;
  output: any;
  jobId: string | undefined;
}) {
  if (!output) return null;

  if (name === "audio") {
    return (
      <div className="text-xs text-ink-400 mt-2">
        duration: {output.duration_seconds?.toFixed?.(2)}s
      </div>
    );
  }

  if (name === "transcribe") {
    return (
      <div className="mt-2">
        <div className="text-[11px] text-ink-400 uppercase tracking-wide">
          {output.language} · {output.segment_count} segments
        </div>
        <div className="text-sm text-ink-200 mt-1 leading-snug max-h-40 overflow-auto">
          {output.text}
        </div>
      </div>
    );
  }

  if (name === "translate") {
    return (
      <div className="mt-2">
        <div className="text-[11px] text-ink-400 uppercase tracking-wide">
          {output.source_language} → {output.target_language} · {output.backend}
        </div>
        <div className="text-sm text-ink-200 mt-1 leading-snug max-h-40 overflow-auto">
          {output.text}
        </div>
      </div>
    );
  }

  if (name === "tts" && jobId && output.path) {
    return (
      <div className="mt-2">
        <div className="text-[11px] text-ink-400 uppercase tracking-wide">
          {output.backend} · voice cloned from original
        </div>
        <audio
          controls
          className="w-full mt-2"
          src={artifactUrl(jobId, output.path)}
        />
      </div>
    );
  }

  if (name === "lipsync") {
    return (
      <div className="text-[11px] text-ink-400 mt-2">
        backend: {output.backend}
        {output.passthrough && " · passthrough (no mouth retouch)"}
      </div>
    );
  }

  return null;
}
