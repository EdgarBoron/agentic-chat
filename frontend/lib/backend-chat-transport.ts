import type { ChatTransport, UIMessage, UIMessageChunk } from "ai";

type BackendEvent =
  | { type: "token"; content: string }
  | { type: "tool_start"; tool: string; call_id: string; input?: unknown; started_at: number }
  | { type: "tool_end"; tool: string; call_id: string; output?: unknown; duration_ms: number | null }
  | { type: "error"; error: string };

/**
 * Translates the FastAPI backend's custom SSE event schema
 * (token / tool_start / tool_end / [DONE] / error) into UIMessageChunks
 * for useChat. The backend does its own LangGraph orchestration, so this
 * is not a Vercel-AI-SDK-native streamText server.
 */
export class BackendChatTransport implements ChatTransport<UIMessage> {
  constructor(private baseUrl: string) {}

  async sendMessages({
    chatId,
    messages,
    abortSignal,
  }: Parameters<ChatTransport<UIMessage>["sendMessages"]>[0]): Promise<
    ReadableStream<UIMessageChunk>
  > {
    const last = messages[messages.length - 1];
    const text = last.parts
      .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
      .map((p) => p.text)
      .join("");

    const res = await fetch(`${this.baseUrl}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: chatId }),
      signal: abortSignal,
    });
    if (!res.ok || !res.body) {
      throw new Error(`Backend chat request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let textStarted = false;
    // tool-output-available replaces toolMetadata wholesale rather than
    // merging it, so startedAt (set on tool-input-available) must be
    // carried forward manually into the tool-output-available chunk.
    const startedAtByCallId: Record<string, number> = {};

    return new ReadableStream<UIMessageChunk>({
      async start(controller) {
        controller.enqueue({ type: "start" });
        let buf = "";
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const events = buf.split("\n\n");
            buf = events.pop() ?? "";

            for (const line of events) {
              if (!line.startsWith("data: ")) continue;
              const payload = line.slice(6);
              if (payload === "[DONE]") continue;

              const evt = JSON.parse(payload) as BackendEvent;

              if (evt.type === "token") {
                if (!textStarted) {
                  controller.enqueue({ type: "text-start", id: "response" });
                  textStarted = true;
                }
                controller.enqueue({
                  type: "text-delta",
                  id: "response",
                  delta: evt.content,
                });
              } else if (evt.type === "tool_start") {
                startedAtByCallId[evt.call_id] = evt.started_at;
                controller.enqueue({
                  type: "tool-input-available",
                  toolCallId: evt.call_id,
                  toolName: evt.tool,
                  input: evt.input ?? {},
                  dynamic: true,
                  toolMetadata: { startedAt: evt.started_at },
                });
              } else if (evt.type === "tool_end") {
                const startedAt = startedAtByCallId[evt.call_id];
                controller.enqueue({
                  type: "tool-output-available",
                  toolCallId: evt.call_id,
                  output: evt.output,
                  dynamic: true,
                  toolMetadata: { startedAt, durationMs: evt.duration_ms },
                });
              } else if (evt.type === "error") {
                controller.enqueue({ type: "error", errorText: evt.error });
              }
            }
          }
          if (textStarted) {
            controller.enqueue({ type: "text-end", id: "response" });
          }
          controller.enqueue({ type: "finish" });
        } finally {
          controller.close();
        }
      },
    });
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    // No server-side resumable-stream support in the backend; a page
    // reload simply starts a fresh request on the next message.
    return null;
  }
}
