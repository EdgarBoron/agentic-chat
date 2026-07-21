from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    target_mode: str = "flux"


class PromptHistoryEntry(BaseModel):
    id: str
    prompt_text: str
    timestamp: str | None = None
    note: str | None = None
    image_url: str | None = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class SavePromptRequest(BaseModel):
    prompt_text: str
    note: str = ""


class SuggestNoteRequest(BaseModel):
    prompt_text: str


class GenerateImageRequest(BaseModel):
    prompt_text: str
    width: int = 1088
    height: int = 1600
    steps: int = 10
    guidance: float = 0.0
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=50)
