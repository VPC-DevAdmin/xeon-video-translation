"use client";

interface Lang {
  code: string;
  name: string;
  flag: string;
}

// Codes here must exist in backend/app/pipeline/translate.py NLLB_LANG_CODES.
// Indic languages at the end are the IndicF5-supported set (TTS backend
// #74 routes these to IndicF5 via tts_backend=auto).
export const LANGUAGES: Lang[] = [
  { code: "es", name: "Spanish", flag: "🇪🇸" },
  { code: "fr", name: "French", flag: "🇫🇷" },
  { code: "de", name: "German", flag: "🇩🇪" },
  { code: "ja", name: "Japanese", flag: "🇯🇵" },
  { code: "zh", name: "Mandarin", flag: "🇨🇳" },
  { code: "hi", name: "Hindi", flag: "🇮🇳" },
  { code: "pt", name: "Portuguese", flag: "🇵🇹" },
  { code: "it", name: "Italian", flag: "🇮🇹" },
  { code: "ko", name: "Korean", flag: "🇰🇷" },
  { code: "bn", name: "Bengali", flag: "🇧🇩" },
  { code: "ta", name: "Tamil", flag: "🇮🇳" },
  { code: "te", name: "Telugu", flag: "🇮🇳" },
  { code: "mr", name: "Marathi", flag: "🇮🇳" },
  { code: "gu", name: "Gujarati", flag: "🇮🇳" },
  { code: "kn", name: "Kannada", flag: "🇮🇳" },
  { code: "ml", name: "Malayalam", flag: "🇮🇳" },
  { code: "pa", name: "Punjabi", flag: "🇮🇳" },
];

export function LanguagePicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (code: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-ink-300">Translate to</label>
      <select
        className="bg-ink-700 border border-ink-600 rounded px-3 py-2 text-ink-200
                   focus:outline-none focus:ring-2 focus:ring-accent"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.flag} {l.name}
          </option>
        ))}
      </select>
    </div>
  );
}
