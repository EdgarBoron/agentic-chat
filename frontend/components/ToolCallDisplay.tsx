import type { DynamicToolUIPart } from "ai";

const TOOL_LABELS: Record<string, string> = {
  web_search: "Searching the web",
  search_prompt_reference: "Checking reference library",
  search_prompt_history: "Checking past prompts",
  save_prompt_to_history: "Saving prompt to history",
};

function stringifyOutput(output: unknown): string {
  if (typeof output === "string") return output;
  try {
    return JSON.stringify(output, null, 2);
  } catch {
    return String(output);
  }
}

export function ToolCallDisplay({ part }: { part: DynamicToolUIPart }) {
  const label = TOOL_LABELS[part.toolName] ?? part.toolName;

  if (part.state === "input-streaming" || part.state === "input-available") {
    return <div className="tool-call">⏳ {label}…</div>;
  }

  if (part.state === "output-available") {
    return (
      <div className="tool-call">
        ✓ {label}
        <pre>{stringifyOutput(part.output).slice(0, 800)}</pre>
      </div>
    );
  }

  if (part.state === "output-error") {
    return <div className="tool-call">✗ {label} failed: {part.errorText}</div>;
  }

  return null;
}
