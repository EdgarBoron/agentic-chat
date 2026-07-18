from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    target_mode: str = "flux"


class PromptHistoryEntry(BaseModel):
    id: str
    prompt_text: str
    timestamp: str | None = None
    note: str | None = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class SavePromptRequest(BaseModel):
    prompt_text: str
    note: str = ""


class SuggestNoteRequest(BaseModel):
    prompt_text: str
