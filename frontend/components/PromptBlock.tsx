"use client";

import { useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001";

type StoreState = "idle" | "noting" | "saving" | "saved" | "error";

export function PromptBlock({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const [storeState, setStoreState] = useState<StoreState>("idle");
  const [note, setNote] = useState("");

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — silently ignore.
    }
  }

  async function confirmStore() {
    setStoreState("saving");
    try {
      const res = await fetch(`${BACKEND_URL}/prompt-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_text: content.trim(), note: note.trim() }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStoreState("saved");
      setNote("");
      setTimeout(() => setStoreState("idle"), 1500);
    } catch {
      setStoreState("error");
      setTimeout(() => setStoreState("idle"), 1500);
    }
  }

  const storeLabel =
    storeState === "saving"
      ? "Storing…"
      : storeState === "saved"
        ? "Stored!"
        : storeState === "error"
          ? "Failed"
          : "Store";

  return (
    <div className="prompt-block">
      <div className="prompt-block-header">
        <span>Prompt</span>
        {storeState !== "noting" && (
          <div className="prompt-block-actions">
            <button type="button" onClick={handleCopy}>
              {copied ? "Copied!" : "Copy"}
            </button>
            <button
              type="button"
              onClick={() => setStoreState("noting")}
              disabled={storeState === "saving"}
            >
              {storeLabel}
            </button>
          </div>
        )}
      </div>
      {storeState === "noting" && (
        <div className="prompt-block-note">
          <input
            autoFocus
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Optional note (e.g. why this one's good)"
            onKeyDown={(e) => {
              if (e.key === "Enter") confirmStore();
              if (e.key === "Escape") setStoreState("idle");
            }}
          />
          <button type="button" onClick={confirmStore}>
            Save
          </button>
          <button type="button" onClick={() => setStoreState("idle")}>
            Skip note
          </button>
        </div>
      )}
      <pre>{content.trim()}</pre>
    </div>
  );
}
