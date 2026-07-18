"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { BackendChatTransport } from "@/lib/backend-chat-transport";
import { getOrCreateThreadId, resetThreadId } from "@/lib/thread-id";
import { matchCommands, findExactCommand, helpText } from "@/lib/commands";
import { ChatMessage } from "@/components/ChatMessage";
import { ActionsPane } from "@/components/ActionsPane";
import { CommandAutocomplete } from "@/components/CommandAutocomplete";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8001";
const transport = new BackendChatTransport(BACKEND_URL);

type HistoryMessage = { role: "user" | "assistant"; content: string };

function toUIMessages(history: HistoryMessage[]): UIMessage[] {
  return history.map((m, i) => ({
    id: `hist-${i}`,
    role: m.role,
    parts: [{ type: "text", text: m.content }],
  }));
}

export default function ChatPage() {
  const [ready, setReady] = useState(false);
  const [threadId, setThreadId] = useState("");
  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);

  useEffect(() => {
    const id = getOrCreateThreadId();
    setThreadId(id);

    fetch(`${BACKEND_URL}/chat/history/${id}`)
      .then((res) => (res.ok ? res.json() : []))
      .then((history: HistoryMessage[]) => setInitialMessages(toUIMessages(history)))
      .catch(() => {})
      .finally(() => setReady(true));
  }, []);

  function handleClear() {
    const newId = resetThreadId();
    setInitialMessages([]);
    setThreadId(newId);
  }

  if (!ready) {
    return (
      <div className="layout">
        <main>
          <p className="history-status">Loading conversation…</p>
        </main>
      </div>
    );
  }

  return (
    <Chat
      key={threadId}
      threadId={threadId}
      initialMessages={initialMessages}
      onClear={handleClear}
    />
  );
}

function Chat({
  threadId,
  initialMessages,
  onClear,
}: {
  threadId: string;
  initialMessages: UIMessage[];
  onClear: () => void;
}) {
  const { messages, sendMessage, setMessages, status, stop, error, clearError } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
  });
  const [input, setInput] = useState("");
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const busy = status === "streaming" || status === "submitted";
  const commandMatches = matchCommands(input);
  const showAutocomplete = commandMatches.length > 0;

  function runCommand(name: string) {
    if (name === "help") {
      setMessages((prev) => [
        ...prev,
        { id: `cmd-${Date.now()}-u`, role: "user", parts: [{ type: "text", text: "/help" }] },
        {
          id: `cmd-${Date.now()}-a`,
          role: "assistant",
          parts: [{ type: "text", text: helpText() }],
        },
      ]);
    } else if (name === "clear") {
      onClear();
    }
    setInput("");
    setHighlightedIndex(0);
  }

  function handleInputChange(value: string) {
    setInput(value);
    setHighlightedIndex(0);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!showAutocomplete) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((i) => (i + 1) % commandMatches.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((i) => (i - 1 + commandMatches.length) % commandMatches.length);
    } else if (e.key === "Escape") {
      setInput("");
    } else if (e.key === "Enter") {
      e.preventDefault();
      runCommand(commandMatches[highlightedIndex].name);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;

    const exact = findExactCommand(input);
    if (exact) {
      runCommand(exact.name);
      return;
    }

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
        {error && (
          <div className="error-banner">
            <span>{error.message}</span>
            <button type="button" onClick={() => clearError()}>
              Dismiss
            </button>
          </div>
        )}
        <form className="chat-input-row" onSubmit={handleSubmit}>
          <div className="chat-input-wrap">
            {showAutocomplete && (
              <CommandAutocomplete
                commands={commandMatches}
                highlightedIndex={highlightedIndex}
                onSelect={(cmd) => runCommand(cmd.name)}
              />
            )}
            <input
              value={input}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe the image you want a prompt for... (try /help)"
              disabled={busy}
            />
          </div>
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
