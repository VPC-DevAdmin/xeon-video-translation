"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

export function Uploader({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
}) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFile(accepted[0]);
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
    maxSize: 100 * 1024 * 1024,
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
            mp4 / mov / webm / mkv · up to 100 MB · ≤ 60 s
          </p>
        </div>
      )}
    </div>
  );
}
