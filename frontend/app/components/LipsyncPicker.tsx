"use client";

import type { LipsyncBackend } from "../lib/api";

const OPTIONS: {
  code: LipsyncBackend;
  label: string;
  blurb: string;
}[] = [
  { code: "none", label: "None (dub over)", blurb: "Instant, no lipsync. Mouth stays in source language." },
  { code: "wav2lip", label: "Wav2Lip", blurb: "2020. ~30–60s for a 3s clip on Xeon. Mediocre quality." },
  { code: "musetalk", label: "MuseTalk", blurb: "Not yet wired up — see docs/lipsync.md." },
  { code: "latentsync", label: "LatentSync", blurb: "Not yet wired up. Effectively unusable on CPU." },
];

export function LipsyncPicker({
  value,
  onChange,
}: {
  value: LipsyncBackend;
  onChange: (backend: LipsyncBackend) => void;
}) {
  const current = OPTIONS.find((o) => o.code === value);
  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-ink-300">Lipsync backend</label>
      <select
        className="bg-ink-700 border border-ink-600 rounded px-3 py-2 text-ink-200
                   focus:outline-none focus:ring-2 focus:ring-accent"
        value={value}
        onChange={(e) => onChange(e.target.value as LipsyncBackend)}
      >
        {OPTIONS.map((o) => (
          <option key={o.code} value={o.code}>
            {o.label}
          </option>
        ))}
      </select>
      {current && (
        <p className="text-xs text-ink-400">{current.blurb}</p>
      )}
    </div>
  );
}
