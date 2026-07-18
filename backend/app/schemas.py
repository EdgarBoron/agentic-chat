from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class PromptHistoryEntry(BaseModel):
    id: str
    prompt_text: str
    timestamp: str | None = None
    note: str | None = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
