export type MessageSegment =
  | { type: "text"; content: string }
  | { type: "code"; content: string };

/**
 * Splits message text on ``` fences into alternating text/code segments.
 * An unclosed trailing fence (still streaming in) is still treated as code,
 * so the copy block renders progressively rather than showing raw
 * backticks mid-stream.
 */
export function parsePromptBlocks(text: string): MessageSegment[] {
  const segments: MessageSegment[] = [];
  const parts = text.split("```");

  parts.forEach((part, i) => {
    if (part === "") return;
    if (i % 2 === 1) {
      // Strip an optional leading language-tag line, e.g. ```text\n...
      const content = part.replace(/^[a-zA-Z0-9_-]*\n/, "");
      segments.push({ type: "code", content });
    } else {
      segments.push({ type: "text", content: part });
    }
  });

  return segments;
}
