"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { BackendChatTransport } from "@/lib/backend-chat-transport";
import { getOrCreateThreadId } from "@/lib/thread-id";
import { ChatMessage } from "@/components/ChatMessage";
import { ActionsPane } from "@/components/ActionsPane";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001";
const transport = new BackendChatTransport(BACKEND_URL);

type HistoryMessage = { role: "user" | "assistant"; content: string };

export default function ChatPage() {
  const [ready, setReady] = useState(false);
  const [threadId, setThreadId] = useState("");
  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);

  useEffect(() => {
    const id = getOrCreateThreadId();
    setThreadId(id);

    fetch(`${BACKEND_URL}/chat/history/${id}`)
      .then((res) => (res.ok ? res.json() : []))
      .then((history: HistoryMessage[]) => {
        setInitialMessages(
          history.map((m, i) => ({
            id: `hist-${i}`,
            role: m.role,
            parts: [{ type: "text", text: m.content }],
          }))
        );
      })
      .catch(() => {})
      .finally(() => setReady(true));
  }, []);

  if (!ready) {
    return (
      <div className="layout">
        <main>
          <p className="history-status">Loading conversation…</p>
        </main>
      </div>
    );
  }

  return <Chat threadId={threadId} initialMessages={initialMessages} />;
}

function Chat({
  threadId,
  initialMessages,
}: {
  threadId: string;
  initialMessages: UIMessage[];
}) {
  const { messages, sendMessage, status, stop } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
  });
  const [input, setInput] = useState("");
  const busy = status === "streaming" || status === "submitted";

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    sendMessage({ text: input });
    setInput("");
  }

  return (
    <div className="layout">
      <ActionsPane messages={messages} />
      <main>
        <div className="page-nav">
          <Link href="/prompt-history">Prompt history →</Link>
        </div>
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
    </div>
  );
}
