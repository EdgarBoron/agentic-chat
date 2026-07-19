export type MessageSegment =
  | { type: "text"; content: string }
  | { type: "code"; lang: string; content: string };

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
      // Split an optional leading language-tag line, e.g. ```consistency\n...
      const match = part.match(/^([a-zA-Z0-9_-]*)\n([\s\S]*)$/);
      const lang = match ? match[1] : "";
      const content = match ? match[2] : part;
      segments.push({ type: "code", lang, content });
    } else {
      segments.push({ type: "text", content: part });
    }
  });

  return segments;
}
