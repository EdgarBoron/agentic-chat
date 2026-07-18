const STORAGE_KEY = "agentic-chat-target-mode";
const DEFAULT_MODE = "flux";

export function getStoredTargetMode(): string {
  return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_MODE;
}

export function storeTargetMode(mode: string): void {
  localStorage.setItem(STORAGE_KEY, mode);
}
