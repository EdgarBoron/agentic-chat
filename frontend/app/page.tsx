"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { BackendChatTransport } from "@/lib/backend-chat-transport";
import { ChatMessage } from "@/components/ChatMessage";

const transport = new BackendChatTransport(
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001"
);

export default function ChatPage() {
  const { messages, sendMessage, status, stop } = useChat({ transport });
  const [input, setInput] = useState("");
  const busy = status === "streaming" || status === "submitted";

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    sendMessage({ text: input });
    setInput("");
  }

  return (
    <main>
      <div className="messages">
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} />
        ))}
      </div>
      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe the image you want a prompt for..."
          disabled={busy}
        />
        {busy ? (
          <button type="button" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </main>
  );
}
