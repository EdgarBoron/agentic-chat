import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.agent.graph import agent_handle
from app.config import settings
from app.schemas import ChatRequest


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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    async def gen():
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
                    payload = {
                        "type": "tool_start",
                        "tool": event["name"],
                        "input": event["data"].get("input"),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif kind == "on_tool_end":
                    payload = {
                        "type": "tool_end",
                        "tool": event["name"],
                        "output": str(event["data"].get("output", ""))[:2000],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
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
