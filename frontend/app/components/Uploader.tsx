"use client";

import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";

// Mirrors MAX_VIDEO_SIZE_MB / MAX_VIDEO_DURATION_SECONDS on the backend, which
// stays authoritative (it rejects oversize uploads and over-long clips).
export const MAX_UPLOAD_MB = 4096;
export const MAX_UPLOAD_MINUTES = 30;

export function Uploader({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
}) {
  const [rejected, setRejected] = useState<string | null>(null);
  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]) => {
      if (accepted[0]) {
        setRejected(null);
        onFile(accepted[0]);
      } else if (rejections[0]) {
        const r = rejections[0];
        const tooBig = r.errors.some(e => e.code === "file-too-large");
        setRejected(
          tooBig
            ? `${r.file.name} is ${(r.file.size / (1024 * 1024)).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB.`
            : `${r.file.name}: ${r.errors.map(e => e.message).join("; ")}`
        );
      }
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/mp4": [".mp4", ".m4v"],
      "video/quicktime": [".mov"],
      "video/webm": [".webm"],
      "video/x-matroska": [".mkv"],
    },
    multiple: false,
    maxSize: MAX_UPLOAD_MB * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
        transition-colors
        ${
          isDragActive
            ? "border-accent bg-ink-700"
            : "border-ink-600 hover:border-ink-500"
        }`}
    >
      <input {...getInputProps()} />
      {file ? (
        <div>
          <div className="font-medium text-ink-200">{file.name}</div>
          <div className="text-sm text-ink-400 mt-1">
            {(file.size / (1024 * 1024)).toFixed(1)} MB
          </div>
          <button
            className="mt-3 text-sm text-accent-soft underline"
            onClick={(e) => {
              e.stopPropagation();
              onFile(null);
            }}
          >
            Choose a different file
          </button>
        </div>
      ) : (
        <div className="text-ink-300">
          <p className="font-medium">Drop a video here, or click to browse</p>
          <p className="text-sm text-ink-400 mt-1">
            mp4 / mov / webm / mkv · up to {MAX_UPLOAD_MB / 1024} GB · ≤ {MAX_UPLOAD_MINUTES} min
          </p>
          {rejected && <p className="text-sm text-red-400 mt-2">{rejected}</p>}
        </div>
      )}
    </div>
  );
}
