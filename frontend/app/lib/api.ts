export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type StageName =
  | "audio"
  | "transcribe"
  | "translate"
  | "tts"
  | "lipsync"
  | "mux";

export type StageStatus =
  | "pending"
  | "running"
  | "done"
  | "failed"
  | "skipped";

export interface StageRecord {
  name: StageName;
  status: StageStatus;
  duration_ms: number | null;
  output: any;
  error: string | null;
}

export interface JobRecord {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  current_stage: StageName | null;
  target_language: string;
  source_language: string | null;
  input_filename: string;
  created_at: string;
  completed_at: string | null;
  error: string | null;
  stages: StageRecord[];
  result_url?: string;
}

export async function createJob(
  video: File,
  target_language: string,
  source_language?: string
): Promise<{ job_id: string; status: string; created_at: string }> {
  const fd = new FormData();
  fd.append("video", video);
  fd.append("target_language", target_language);
  if (source_language) fd.append("source_language", source_language);

  const res = await fetch(`${API_BASE_URL}/jobs`, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /jobs failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`GET /jobs/${jobId} failed (${res.status})`);
  return res.json();
}

export function openJobEventStream(
  jobId: string,
  onEvent: (event: string, data: any) => void,
  onError?: (e: Event) => void
): EventSource {
  const url = `${API_BASE_URL}/jobs/${jobId}/events`;
  const es = new EventSource(url);
  const handle = (name: string) => (ev: MessageEvent) => {
    let parsed: any = ev.data;
    try {
      parsed = JSON.parse(ev.data);
    } catch {
      /* keep as string */
    }
    onEvent(name, parsed);
  };
  // Listen for the named events the backend emits.
  for (const name of [
    "job_started",
    "stage_started",
    "stage_progress",
    "stage_completed",
    "stage_skipped",
    "job_completed",
    "error",
    "ping",
    "stream_end",
  ]) {
    es.addEventListener(name, handle(name) as EventListener);
  }
  if (onError) es.onerror = onError;
  return es;
}

export function artifactUrl(jobId: string, name: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/artifacts/${name}`;
}
