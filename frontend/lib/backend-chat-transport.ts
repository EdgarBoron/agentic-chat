import type { ChatTransport, UIMessage, UIMessageChunk } from "ai";

type BackendEvent =
  | { type: "token"; content: string }
  | { type: "tool_start"; tool: string; input?: unknown }
  | { type: "tool_end"; tool: string; output?: unknown }
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
    // Tracks in-flight tool calls per tool name so repeated calls to the
    // same tool within one turn get distinct toolCallIds.
    const pendingCallIds: Record<string, string[]> = {};
    let textStarted = false;

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
                const callId = `${evt.tool}-${crypto.randomUUID()}`;
                pendingCallIds[evt.tool] ??= [];
                pendingCallIds[evt.tool].push(callId);
                controller.enqueue({
                  type: "tool-input-available",
                  toolCallId: callId,
                  toolName: evt.tool,
                  input: evt.input ?? {},
                  dynamic: true,
                });
              } else if (evt.type === "tool_end") {
                const callId = pendingCallIds[evt.tool]?.shift() ?? evt.tool;
                controller.enqueue({
                  type: "tool-output-available",
                  toolCallId: callId,
                  output: evt.output,
                  dynamic: true,
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
