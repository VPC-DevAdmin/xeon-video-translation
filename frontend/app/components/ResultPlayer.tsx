"use client";

import { artifactUrl } from "../lib/api";
import type { JobRecord } from "../lib/api";

export function ResultPlayer({ job }: { job: JobRecord | null }) {
  if (!job || job.status !== "completed") return null;
  const finalStage = job.stages.find((s) => s.name === "mux");
  if (!finalStage || finalStage.status !== "done") return null;
  const path = finalStage.output?.path as string | undefined;
  if (!path) return null;

  const url = artifactUrl(job.job_id, path);

  return (
    <section className="mt-8">
      <h2 className="text-sm uppercase tracking-wide text-ink-400 mb-3">
        Final video
      </h2>
      <div className="border border-ink-600 rounded-lg bg-ink-800 p-4">
        <video
          controls
          className="w-full rounded border border-ink-700 bg-black"
          src={url}
        />
        <div className="mt-3 flex items-center justify-between text-sm">
          <p className="text-ink-400">
            AI-generated content. Watermark is embedded. See <code>docs/ethics.md</code>.
          </p>
          <a
            href={url}
            download
            className="text-accent-soft hover:text-accent underline"
          >
            Download final.mp4
          </a>
        </div>
      </div>
    </section>
  );
}
