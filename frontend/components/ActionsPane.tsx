import type { DynamicToolUIPart, UIMessage } from "ai";
import { isDynamicToolUIPart } from "ai";

const TOOL_LABELS: Record<string, string> = {
  web_search: "Web search",
  search_prompt_reference: "Reference library search",
  search_prompt_history: "Prompt history search",
  save_prompt_to_history: "Save prompt to history",
};

function formatClockTime(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour12: false });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function stringifyPreview(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function ActionsPane({ messages }: { messages: UIMessage[] }) {
  const actions = messages
    .flatMap((m) => m.parts)
    .filter(isDynamicToolUIPart)
    .map((part) => {
      const meta = (part.toolMetadata ?? {}) as {
        startedAt?: number;
        durationMs?: number | null;
      };
      return { part, startedAt: meta.startedAt, durationMs: meta.durationMs };
    })
    .sort((a, b) => (a.startedAt ?? 0) - (b.startedAt ?? 0));

  return (
    <aside className="actions-pane">
      <div className="actions-pane-header">Actions</div>
      <div className="actions-list">
        {actions.length === 0 && (
          <div className="actions-empty">No actions performed yet.</div>
        )}
        {actions.map(({ part, startedAt, durationMs }, i) => (
          <ActionItem key={part.toolCallId + i} part={part} startedAt={startedAt} durationMs={durationMs} />
        ))}
      </div>
    </aside>
  );
}

function ActionItem({
  part,
  startedAt,
  durationMs,
}: {
  part: DynamicToolUIPart;
  startedAt: number | undefined;
  durationMs: number | null | undefined;
}) {
  const label = TOOL_LABELS[part.toolName] ?? part.toolName;
  const running = part.state === "input-streaming" || part.state === "input-available";
  const failed = part.state === "output-error";

  return (
    <div className={`action-item ${running ? "running" : ""} ${failed ? "failed" : ""}`}>
      <div className="action-item-row">
        <span className="action-status">{running ? "⏳" : failed ? "✗" : "✓"}</span>
        <span className="action-name">{label}</span>
      </div>
      <div className="action-meta">
        {startedAt !== undefined && <span>{formatClockTime(startedAt)}</span>}
        {typeof durationMs === "number" && <span>· {formatDuration(durationMs)}</span>}
        {running && <span>· running…</span>}
      </div>
      {part.input !== undefined && (
        <details className="action-details">
          <summary>input</summary>
          <pre>{stringifyPreview(part.input)}</pre>
        </details>
      )}
      {part.state === "output-available" && (
        <details className="action-details">
          <summary>output</summary>
          <pre>{stringifyPreview(part.output)}</pre>
        </details>
      )}
      {failed && <div className="action-error">{part.errorText}</div>}
    </div>
  );
}
