"use client";

import { useEffect, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001";

type StoreState = "idle" | "noting" | "saving" | "saved" | "error";

type GenerateState =
  | "idle"
  | "confirming"
  | "stopping_vllm"
  | "generating"
  | "restarting_vllm"
  | "done"
  | "error";

type GenerateEvent =
  | { type: "phase"; phase: string; elapsed: number; index?: number; count?: number }
  | { type: "heartbeat"; phase: string; elapsed: number; index?: number; count?: number }
  | { type: "image"; image_data: string; index: number; seed: number | null; elapsed: number }
  | { type: "done"; elapsed: number }
  | { type: "error"; error: string };

const GENERATE_PHASE_LABELS: Record<string, string> = {
  stopping_vllm: "Pausing chat model…",
  generating: "Generating image…",
  restarting_vllm: "Resuming chat model…",
};

export function PromptBlock({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const [storeState, setStoreState] = useState<StoreState>("idle");
  const [note, setNote] = useState("");
  const [suggesting, setSuggesting] = useState(false);

  const [generateState, setGenerateState] = useState<GenerateState>("idle");
  const [generateElapsed, setGenerateElapsed] = useState(0);
  const [generateError, setGenerateError] = useState("");
  const [generateBatch, setGenerateBatch] = useState<{ index: number; count: number } | null>(
    null,
  );
  const [images, setImages] = useState<{ src: string; seed: number | null }[]>([]);

  const [promptText, setPromptText] = useState(content.trim());
  const [promptEdited, setPromptEdited] = useState(false);

  // `content` keeps growing token-by-token while the assistant message is
  // still streaming. Until the user actually edits the box, mirror that
  // live — otherwise the textarea's local state freezes at whatever content
  // existed on first mount and printing appears to just stop.
  useEffect(() => {
    if (!promptEdited) setPromptText(content.trim());
  }, [content, promptEdited]);

  const [width, setWidth] = useState(1088);
  const [height, setHeight] = useState(1600);
  const [steps, setSteps] = useState(10);
  const [guidance, setGuidance] = useState(0);
  const [seed, setSeed] = useState("");
  const [count, setCount] = useState(1);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(promptText.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — silently ignore.
    }
  }

  function openNoteInput() {
    setStoreState("noting");
    setSuggesting(true);
    fetch(`${BACKEND_URL}/prompt-history/suggest-note`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_text: promptText.trim() }),
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { note: string } | null) => {
        // Only fill in if the user hasn't already typed something while we waited.
        setNote((current) => (data && !current ? data.note : current));
      })
      .catch(() => {})
      .finally(() => setSuggesting(false));
  }

  async function confirmStore() {
    setStoreState("saving");
    try {
      const res = await fetch(`${BACKEND_URL}/prompt-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_text: promptText.trim(), note: note.trim() }),
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

  async function confirmGenerate() {
    setGenerateState("stopping_vllm");
    setGenerateElapsed(0);
    setGenerateError("");
    setGenerateBatch(null);
    setImages([]);
    try {
      const trimmedSeed = seed.trim();
      const res = await fetch(`${BACKEND_URL}/generate-image/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_text: promptText.trim(),
          width,
          height,
          steps,
          guidance,
          seed: trimmedSeed === "" ? null : Number(trimmedSeed),
          count,
        }),
      });
      if (!res.ok || !res.body) {
        throw new Error(res.status === 409 ? "A generation is already in progress" : `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const line of parts) {
          if (!line.startsWith("data: ")) continue;
          const evt = JSON.parse(line.slice(6)) as GenerateEvent;
          if (evt.type === "phase" || evt.type === "heartbeat") {
            setGenerateState(evt.phase as GenerateState);
            setGenerateElapsed(evt.elapsed);
            if (evt.index !== undefined && evt.count !== undefined) {
              setGenerateBatch({ index: evt.index, count: evt.count });
            }
          } else if (evt.type === "image") {
            setImages((prev) => [...prev, { src: evt.image_data, seed: evt.seed }]);
          } else if (evt.type === "done") {
            setGenerateState("done");
          } else if (evt.type === "error") {
            setGenerateError(evt.error);
            setGenerateState("error");
          }
        }
      }
    } catch (e) {
      setGenerateError(e instanceof Error ? e.message : "Generation failed");
      setGenerateState("error");
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

  const generateBusy =
    generateState === "stopping_vllm" ||
    generateState === "generating" ||
    generateState === "restarting_vllm";

  const generateLabel =
    generateState === "done" ? "Generated!" : generateBusy ? "Generating…" : "Generate";

  return (
    <div className="prompt-block">
      <div className="prompt-block-header">
        <span>Prompt</span>
        {storeState !== "noting" && generateState !== "confirming" && !generateBusy && (
          <div className="prompt-block-actions">
            <button type="button" onClick={handleCopy}>
              {copied ? "Copied!" : "Copy"}
            </button>
            <button type="button" onClick={openNoteInput} disabled={storeState === "saving"}>
              {storeLabel}
            </button>
            <button type="button" onClick={() => setGenerateState("confirming")}>
              {generateLabel}
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
            placeholder={suggesting ? "Suggesting a note…" : "Optional note"}
            onKeyDown={(e) => {
              if (e.key === "Enter") confirmStore();
              if (e.key === "Escape") setStoreState("idle");
            }}
          />
          <button type="button" onClick={confirmStore}>
            Save
          </button>
          <button type="button" onClick={() => setStoreState("idle")}>
            Cancel
          </button>
        </div>
      )}
      {generateState === "confirming" && (
        <div className="prompt-generate-settings">
          <span className="prompt-generate-confirm-text">
            This will pause chat for a few minutes while the image generates. Continue?
          </span>
          <div className="prompt-generate-fields">
            <label>
              Width
              <input
                type="number"
                min={64}
                step={64}
                value={width}
                onChange={(e) => setWidth(Number(e.target.value))}
              />
            </label>
            <label>
              Height
              <input
                type="number"
                min={64}
                step={64}
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
              />
            </label>
            <label>
              Steps
              <input
                type="number"
                min={1}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
              />
            </label>
            <label>
              Guidance
              <input
                type="number"
                min={0}
                step={0.1}
                value={guidance}
                onChange={(e) => setGuidance(Number(e.target.value))}
              />
            </label>
            <label>
              Seed
              <input
                type="text"
                inputMode="numeric"
                placeholder="Random"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </label>
            <label>
              Count
              <input
                type="number"
                min={1}
                max={50}
                value={count}
                onChange={(e) => setCount(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="prompt-block-note">
            <button type="button" onClick={confirmGenerate} disabled={promptText.trim() === ""}>
              Yes, generate
            </button>
            <button type="button" onClick={() => setGenerateState("idle")}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {generateBusy && (
        <div className="prompt-generate-progress">
          {GENERATE_PHASE_LABELS[generateState] ?? "Working…"}
          {generateBatch && generateBatch.count > 1
            ? ` (image ${generateBatch.index + 1} of ${generateBatch.count})`
            : ""}{" "}
          ({generateElapsed}s elapsed)
        </div>
      )}
      {generateState === "error" && (
        <div className="prompt-generate-error">
          {generateError}
          <button type="button" onClick={() => setGenerateState("idle")}>
            Dismiss
          </button>
        </div>
      )}
      <textarea
        className="prompt-block-text"
        value={promptText}
        onChange={(e) => {
          setPromptEdited(true);
          setPromptText(e.target.value);
        }}
        rows={Math.max(9, promptText.split("\n").length)}
      />
      {images.length > 0 && (
        <div className="prompt-generated-images">
          {images.map((img, i) => (
            <figure key={i} className="prompt-generated-image-item">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={img.src} alt={`Generated ${i + 1}`} className="prompt-generated-image" />
              {img.seed !== null && (
                <figcaption className="prompt-generated-image-seed">seed: {img.seed}</figcaption>
              )}
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
