import type { UIMessage } from "ai";
import { isTextUIPart } from "ai";
import { parsePromptBlocks } from "@/lib/parse-prompt-blocks";
import { PromptBlock } from "@/components/PromptBlock";

export function ChatMessage({ message }: { message: UIMessage }) {
  return (
    <>
      {message.parts.map((part, i) => {
        if (isTextUIPart(part)) {
          const segments = parsePromptBlocks(part.text);
          return (
            <div key={i} className={`message ${message.role}`}>
              {segments.map((seg, j) =>
                seg.type === "code" ? (
                  <PromptBlock key={j} content={seg.content} />
                ) : (
                  <span key={j} className="message-text">
                    {seg.content}
                  </span>
                )
              )}
            </div>
          );
        }
        return null;
      })}
    </>
  );
}
