"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

type PromptHistoryEntry = {
  id: string;
  prompt_text: string;
  timestamp: string | null;
  note: string | null;
  image_urls: string[];
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

  function handleDeleted(id: string) {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }

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
          <HistoryItem key={entry.id} entry={entry} onDeleted={handleDeleted} />
        ))}
      </div>
    </main>
  );
}

function HistoryItem({
  entry,
  onDeleted,
}: {
  entry: PromptHistoryEntry;
  onDeleted: (id: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(entry.prompt_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable — silently ignore.
    }
  }

  function handleDeleteClick() {
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      setTimeout(() => setConfirmingDelete(false), 3000);
      return;
    }
    setDeleting(true);
    fetch(`${BACKEND_URL}/prompt-history/${entry.id}`, { method: "DELETE" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        onDeleted(entry.id);
      })
      .catch(() => {
        setDeleting(false);
        setConfirmingDelete(false);
      });
  }

  return (
    <div className="history-item">
      <div className="history-item-top">
        <div className="history-item-meta">{formatTimestamp(entry.timestamp)}</div>
        <div className="history-item-actions">
          <button type="button" onClick={handleCopy}>
            {copied ? "Copied!" : "Copy"}
          </button>
          <button
            type="button"
            className={confirmingDelete ? "confirm-delete" : ""}
            onClick={handleDeleteClick}
            disabled={deleting}
          >
            {deleting ? "Deleting…" : confirmingDelete ? "Confirm delete?" : "Delete"}
          </button>
        </div>
      </div>
      <div className="history-item-text">{entry.prompt_text}</div>
      {entry.note && <div className="history-item-note">{entry.note}</div>}
      {entry.image_urls.length > 0 && (
        <div className="history-item-images">
          {entry.image_urls.map((url) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={url}
              src={`${BACKEND_URL}${url}`}
              alt="Generated"
              className="history-item-image"
            />
          ))}
        </div>
      )}
    </div>
  );
}
