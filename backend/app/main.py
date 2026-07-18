import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app.agent.graph import agent_handle
from app.config import settings
from app.memory.chroma_client import get_history_collection, save_prompt
from app.schemas import (
    ChatHistoryMessage,
    ChatRequest,
    PromptHistoryEntry,
    SavePromptRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent_handle.start(settings)
    yield
    await agent_handle.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/prompt-history", response_model=list[PromptHistoryEntry])
async def prompt_history():
    coll = get_history_collection(settings.chroma_persist_dir)
    res = coll.get()
    entries = [
        PromptHistoryEntry(
            id=doc_id,
            prompt_text=doc,
            timestamp=meta.get("timestamp"),
            note=meta.get("note") or None,
        )
        for doc_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    entries.sort(key=lambda e: e.timestamp or "", reverse=True)
    return entries


@app.post("/prompt-history")
async def save_prompt_history(req: SavePromptRequest):
    save_prompt(settings.chroma_persist_dir, req.prompt_text, req.note)
    return {"status": "saved"}


@app.get("/chat/history/{thread_id}", response_model=list[ChatHistoryMessage])
async def chat_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent_handle.graph.aget_state(config)
    messages = state.values.get("messages", [])

    history: list[ChatHistoryMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            history.append(ChatHistoryMessage(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage) and msg.content:
            # Skip tool-call-only AI turns (empty content); only text replies
            # are reconstructable as chat bubbles on the frontend.
            history.append(ChatHistoryMessage(role="assistant", content=str(msg.content)))
    return history


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # A tool-call round trip is ~2 graph steps; the system prompt caps the
    # agent at 3 tool calls, so 15 gives headroom while still failing fast
    # (rather than the LangGraph default of 25) if it starts looping.
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 15}

    async def gen():
        tool_start_times: dict[str, float] = {}
        try:
            async for event in agent_handle.graph.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

                elif kind == "on_tool_start":
                    call_id = event["run_id"]
                    tool_start_times[call_id] = time.monotonic()
                    payload = {
                        "type": "tool_start",
                        "tool": event["name"],
                        "call_id": call_id,
                        "input": event["data"].get("input"),
                        "started_at": time.time() * 1000,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif kind == "on_tool_end":
                    call_id = event["run_id"]
                    started = tool_start_times.pop(call_id, None)
                    duration_ms = (
                        round((time.monotonic() - started) * 1000)
                        if started is not None
                        else None
                    )
                    payload = {
                        "type": "tool_end",
                        "tool": event["name"],
                        "call_id": call_id,
                        "output": str(event["data"].get("output", ""))[:2000],
                        "duration_ms": duration_ms,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
        except GraphRecursionError:
            msg = (
                "I got stuck researching this and couldn't settle on a final "
                "prompt. Try rephrasing your request more specifically."
            )
            yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
        except Exception as e:  # noqa: BLE001 - surface any failure to the client stream
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
