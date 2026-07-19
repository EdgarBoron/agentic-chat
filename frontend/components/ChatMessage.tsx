import type { UIMessage } from "ai";
import { isTextUIPart } from "ai";
import { parsePromptBlocks } from "@/lib/parse-prompt-blocks";
import { PromptBlock } from "@/components/PromptBlock";
import { ConsistencyBlock } from "@/components/ConsistencyBlock";

export function ChatMessage({
  message,
  onSendText,
}: {
  message: UIMessage;
  onSendText: (text: string) => void;
}) {
  return (
    <>
      {message.parts.map((part, i) => {
        if (isTextUIPart(part)) {
          const segments = parsePromptBlocks(part.text);
          return (
            <div key={i} className={`message ${message.role}`}>
              {segments.map((seg, j) => {
                if (seg.type !== "code") {
                  return (
                    <span key={j} className="message-text">
                      {seg.content}
                    </span>
                  );
                }
                if (seg.lang === "consistency") {
                  return <ConsistencyBlock key={j} content={seg.content} onSubmit={onSendText} />;
                }
                return <PromptBlock key={j} content={seg.content} />;
              })}
            </div>
          );
        }
        return null;
      })}
    </>
  );
}
