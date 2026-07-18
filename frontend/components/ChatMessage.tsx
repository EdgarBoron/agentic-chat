import type { UIMessage } from "ai";
import { isTextUIPart } from "ai";

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
        return null;
      })}
    </>
  );
}
