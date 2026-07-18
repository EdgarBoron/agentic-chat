import type { UIMessage } from "ai";
import { isDynamicToolUIPart, isTextUIPart } from "ai";
import { ToolCallDisplay } from "./ToolCallDisplay";

export function ChatMessage({ message }: { message: UIMessage }) {
  return (
    <>
      {message.parts.map((part, i) => {
        if (isTextUIPart(part)) {
          return (
            <div key={i} className={`message ${message.role}`}>
              {part.text}
            </div>
          );
        }
        if (isDynamicToolUIPart(part)) {
          return <ToolCallDisplay key={i} part={part} />;
        }
        return null;
      })}
    </>
  );
}
