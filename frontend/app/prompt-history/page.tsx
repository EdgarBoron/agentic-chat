"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

type PromptHistoryEntry = {
  id: string;
  prompt_text: string;
  timestamp: string | null;
  note: string | null;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001";

function formatTimestamp(ts: string | null): string {
  if (!ts) return "unknown time";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export default function PromptHistoryPage() {
  const [entries, setEntries] = useState<PromptHistoryEntry[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  const load = useCallback(async () => {
    setStatus("loading");
    try {
      const res = await fetch(`${BACKEND_URL}/prompt-history`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: PromptHistoryEntry[] = await res.json();
      setEntries(data);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <main className="history-page">
      <div className="page-nav">
        <Link href="/">← Back to chat</Link>
        <button onClick={load} disabled={status === "loading"}>
          Refresh
        </button>
      </div>
      <h1>Prompt History</h1>

      {status === "loading" && <p className="history-status">Loading…</p>}
      {status === "error" && (
        <p className="history-status error">Failed to load prompt history.</p>
      )}
      {status === "ready" && entries.length === 0 && (
        <p className="history-status">No prompts saved yet.</p>
      )}

      <div className="history-list">
        {entries.map((entry) => (
          <div key={entry.id} className="history-item">
            <div className="history-item-meta">{formatTimestamp(entry.timestamp)}</div>
            <div className="history-item-text">{entry.prompt_text}</div>
            {entry.note && <div className="history-item-note">{entry.note}</div>}
          </div>
        ))}
      </div>
    </main>
  );
}
