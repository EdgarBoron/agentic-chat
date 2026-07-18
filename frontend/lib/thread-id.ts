const STORAGE_KEY = "agentic-chat-thread-id";

export function getOrCreateThreadId(): string {
  const existing = localStorage.getItem(STORAGE_KEY);
  if (existing) return existing;
  return resetThreadId();
}

export function resetThreadId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(STORAGE_KEY, id);
  return id;
}
