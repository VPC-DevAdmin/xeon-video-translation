"use client";
import { useEffect, useState } from "react";
import "./lookup-card.css";

/** The notes card the persona looks at while it "looks something up": the question,
 *  a checking state, then a few key phrases of the planned answer appearing one by
 *  one. It sits at the lower left of the stage, where the posed portrait's gaze points. */
export type LookupState = { question: string; notes: string[]; startedAt: number };

export function notesFrom(reply: string, max = 4): string[] {
  const parts = reply.split(/(?<=[.!?。！？])\s+/).map(p => p.trim()).filter(Boolean);
  return parts.slice(0, max).map(p => {
    const words = p.replace(/[.!?。！？]+$/, "").split(/\s+/);
    return words.length > 7 ? words.slice(0, 7).join(" ") + "…" : words.join(" ");
  });
}

export default function LookupCard({ state }: { state: LookupState | null }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    setShown(0);
    if (!state) return;
    const timer = window.setInterval(() => setShown(n => Math.min(n + 1, state.notes.length)), 800);
    return () => window.clearInterval(timer);
  }, [state]);
  if (!state) return null;
  return <aside className="lookup-card" aria-live="polite" aria-label="Notes">
    <div className="lookup-card-head"><span className="lookup-card-dot" />Looking up</div>
    {state.question && <p className="lookup-card-question">{state.question}</p>}
    <ul className="lookup-card-notes">
      {state.notes.slice(0, shown).map((note, i) => <li key={i}>{note}</li>)}
      {shown < state.notes.length || !state.notes.length ? <li className="lookup-card-pending">checking…</li> : null}
    </ul>
  </aside>;
}
