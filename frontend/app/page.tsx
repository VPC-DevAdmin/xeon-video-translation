"use client";

import { useEffect, useRef, useState } from "react";
import { Uploader } from "./components/Uploader";
import { LanguagePicker } from "./components/LanguagePicker";
import { PipelineView } from "./components/PipelineView";
import {
  createJob,
  getJob,
  openJobEventStream,
  type JobRecord,
} from "./lib/api";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("es");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  async function onTranslate() {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setJob(null);
    try {
      const created = await createJob(file, target);
      // Hydrate initial JobRecord, then subscribe to events.
      const initial = await getJob(created.job_id);
      setJob(initial);

      esRef.current?.close();
      esRef.current = openJobEventStream(created.job_id, async (eventName) => {
        // Re-fetch the canonical job record on every event. Simple and robust;
        // the payload we get from SSE is enough to drive UI but we don't want
        // to maintain two reducers.
        if (
          eventName === "stage_completed" ||
          eventName === "stage_started" ||
          eventName === "stage_skipped" ||
          eventName === "job_completed" ||
          eventName === "error"
        ) {
          try {
            const next = await getJob(created.job_id);
            setJob(next);
          } catch {
            /* tolerate transient errors */
          }
        }
      });
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = !!file && !submitting && job?.status !== "running";

  return (
    <main className="max-w-5xl mx-auto px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-ink-200">polyglot-demo</h1>
        <p className="text-ink-400 mt-1">
          Open-source video translation · CPU-only build · M1 + M2
        </p>
      </header>

      <section className="grid md:grid-cols-[1fr_240px] gap-6 items-start mb-6">
        <Uploader file={file} onFile={setFile} />
        <div className="flex flex-col gap-4">
          <LanguagePicker value={target} onChange={setTarget} />
          <button
            onClick={onTranslate}
            disabled={!canSubmit}
            className="bg-accent hover:bg-accent-soft disabled:bg-ink-600
                       disabled:text-ink-400 text-white rounded px-4 py-2
                       font-medium transition-colors"
          >
            {submitting ? "Uploading…" : "Translate"}
          </button>
          {job && (
            <div className="text-xs text-ink-400 break-all">
              job: {job.job_id.slice(0, 8)}… · {job.status}
            </div>
          )}
        </div>
      </section>

      {error && (
        <div className="mb-6 p-3 rounded border border-red-700 bg-red-950/40 text-red-200 text-sm">
          {error}
        </div>
      )}

      <section className="mb-8">
        <h2 className="text-sm uppercase tracking-wide text-ink-400 mb-3">
          Pipeline
        </h2>
        <PipelineView job={job} />
      </section>

      <footer className="border-t border-ink-700 pt-4 text-xs text-ink-400">
        Outputs are AI-generated and watermarked. See <code>docs/ethics.md</code>.
      </footer>
    </main>
  );
}
