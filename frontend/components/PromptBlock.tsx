"use client";

import { useState } from "react";

export function PromptBlock({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — silently ignore.
    }
  }

  return (
    <div className="prompt-block">
      <div className="prompt-block-header">
        <span>Prompt</span>
        <button type="button" onClick={handleCopy}>
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre>{content.trim()}</pre>
    </div>
  );
}
