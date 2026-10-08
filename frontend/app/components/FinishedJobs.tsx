"use client";

import { useEffect, useState } from "react";
import { artifactUrl, listJobs, API_BASE_URL, type JobRecord } from "../lib/api";
import { LANGUAGES } from "./LanguagePicker";

const MODE_LABEL: Record<string, string> = { stream: "Streaming", fast: "Fast", quality: "Quality", dub: "Audio only" };

function stem(name: string): string {
  return (name || "video").replace(/\.[^.]+$/, "");
}

function clock(seconds: number | null): string {
  if (!seconds && seconds !== 0) return "—";
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function when(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** Completed translations, newest first, with watch and download links. The
 *  page itself only tracks the job it started; this lists every finished one
 *  (including jobs started from the studio or the API). */
export function FinishedJobs({ refreshKey }: { refreshKey?: string }) {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const SHOWN = 10;
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const all = await listJobs(100);
        if (alive) {
          setJobs(all.filter(j => j.status === "completed"));
          setError("");
        }
      } catch (e) {
        if (alive) setError(String(e));
      }
    }
    void load();
    const id = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [refreshKey]);

  return (
    <section className="mb-8" aria-labelledby="finished-heading">
      <h2 id="finished-heading" className="text-sm uppercase tracking-wide text-ink-400 mb-3">
        Finished translations
      </h2>
      {error && <p className="text-sm text-red-300">Could not load the list: {error}</p>}
      {!error && !jobs.length && <p className="text-sm text-ink-400">Nothing finished yet.</p>}
      <ul className="divide-y divide-ink-700 border border-ink-700 rounded">
        {(showAll ? jobs : jobs.slice(0, SHOWN)).map(job => {
          const lang = LANGUAGES.find(l => l.code === job.target_language);
          const base = `${stem(job.input_filename)} · ${job.target_language} · ${job.mode || "job"}`;
          const isAudioOnly = job.mode === "dub";
          return (
            <li key={job.job_id} className="p-3">
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="font-medium text-ink-200 break-all">{job.input_filename || job.job_id.slice(0, 8)}</span>
                <span className="text-sm text-ink-400">
                  {lang ? `${lang.flag} ${lang.name}` : job.target_language} · {MODE_LABEL[job.mode || ""] || job.mode || "—"} ·{" "}
                  {clock(job.source_duration_seconds)} · finished {when(job.completed_at)}
                </span>
              </div>
              <div className="flex flex-wrap gap-4 mt-2 text-sm">
                {!isAudioOnly && (
                  <button className="text-accent-soft underline" onClick={() => setOpen(open === job.job_id ? null : job.job_id)}>
                    {open === job.job_id ? "Hide" : "Watch"}
                  </button>
                )}
                {!isAudioOnly && (
                  <a className="text-accent-soft underline" href={artifactUrl(job.job_id, "final.mp4")} download={`${base}.mp4`}>
                    Download video
                  </a>
                )}
                <a className="text-accent-soft underline" href={artifactUrl(job.job_id, "translated_audio.wav")} download={`${base}.wav`}>
                  Download audio
                </a>
                <a className="text-accent-soft underline" href={`${API_BASE_URL}/jobs/${job.job_id}/subtitles/srt`} download={`${base}.srt`}>
                  Subtitles (SRT)
                </a>
                <a className="text-ink-400 underline" href={`/studio`}>Open in studio</a>
              </div>
              {open === job.job_id && (
                <video controls preload="metadata" className="w-full mt-3 rounded bg-black" src={artifactUrl(job.job_id, "final.mp4")} />
              )}
            </li>
          );
        })}
      </ul>
      {jobs.length > SHOWN && (
        <button className="mt-2 text-sm text-accent-soft underline" onClick={() => setShowAll(!showAll)}>
          {showAll ? `Show the newest ${SHOWN}` : `Show all ${jobs.length}`}
        </button>
      )}
    </section>
  );
}
