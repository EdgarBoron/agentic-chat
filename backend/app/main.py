import asyncio
import json
import time
from contextlib import asynccontextmanager

import docker
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from app.agent.graph import agent_handle
from app.agent.prompts import NOTE_SUGGESTION_PROMPT, TARGET_MODES
from app.config import settings
from app.memory import image_store
from app.memory.chroma_client import get_history_collection, save_prompt
from app.memory.hashing import prompt_hash
from app.schemas import (
    ChatHistoryMessage,
    ChatRequest,
    GenerateImageRequest,
    PromptHistoryEntry,
    SavePromptRequest,
    SuggestNoteRequest,
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


@app.get("/target-modes")
async def target_modes():
    return TARGET_MODES


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
            # Resolved fresh on every read so linkage works regardless of
            # whether Generate or Store happened first for this prompt.
            image_url=(
                f"/generated-images/{doc_id}"
                if image_store.has_image(settings.images_db_path, doc_id)
                else None
            ),
        )
        for doc_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    entries.sort(key=lambda e: e.timestamp or "", reverse=True)
    return entries


@app.post("/prompt-history")
async def save_prompt_history(req: SavePromptRequest):
    save_prompt(settings.chroma_persist_dir, req.prompt_text, req.note)
    return {"status": "saved"}


@app.post("/prompt-history/suggest-note")
async def suggest_note(req: SuggestNoteRequest):
    response = await agent_handle.llm.ainvoke(
        [
            SystemMessage(content=NOTE_SUGGESTION_PROMPT),
            HumanMessage(content=req.prompt_text),
        ]
    )
    note = str(response.content).strip().strip('"').strip("'")
    return {"note": note}


@app.delete("/prompt-history/{entry_id}")
async def delete_prompt_history(entry_id: str):
    coll = get_history_collection(settings.chroma_persist_dir)
    # Delete by exact id only — never a `where` filter here, it's too easy
    # to accidentally match far more entries than intended.
    coll.delete(ids=[entry_id])
    return {"status": "deleted"}


@app.get("/generated-images/{image_hash}")
async def get_generated_image(image_hash: str):
    path = image_store.get_image_path(settings.images_db_path, image_hash)
    if path is None:
        raise HTTPException(status_code=404, detail="No image for this prompt")
    return FileResponse(path, media_type="image/png")


def _find_vllm_container(client: docker.DockerClient):
    containers = client.containers.list(
        all=True, filters={"label": "com.docker.compose.service=vllm"}
    )
    if not containers:
        raise RuntimeError("vllm container not found")
    return containers[0]


async def _wait_for_vllm_ready(timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{settings.vllm_base_url}/models")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    raise TimeoutError("vLLM did not become ready within the timeout")


_generate_lock = asyncio.Lock()


@app.post("/generate-image/stream")
async def generate_image_stream(req: GenerateImageRequest):
    if _generate_lock.locked():
        raise HTTPException(
            status_code=409, detail="A generation is already in progress"
        )

    async def gen():
        async with _generate_lock:
            start = time.monotonic()

            def elapsed() -> int:
                return round(time.monotonic() - start)

            def sse(event_type: str, **fields) -> str:
                return f"data: {json.dumps({'type': event_type, **fields})}\n\n"

            # Phase 1: stop vllm to free VRAM for generation. Only relevant
            # when the chat LLM is the local vLLM server — a remote
            # provider (e.g. OpenAI) doesn't hold any local GPU memory, so
            # there's nothing to pause around the generation call.
            container = None
            if settings.uses_local_vllm:
                yield sse("phase", phase="stopping_vllm", elapsed=elapsed())
                try:
                    client = docker.from_env()
                    container = await asyncio.to_thread(_find_vllm_container, client)
                    if container.status == "running":
                        await asyncio.to_thread(container.stop)
                except Exception as e:  # noqa: BLE001
                    # Nothing was actually stopped, so nothing needs restarting.
                    yield sse("error", error=f"Failed to stop vLLM: {e}")
                    return

            # Phase 2: generate, with periodic heartbeats since this can
            # take several minutes.
            #
            # Everything from here down runs inside try/finally, not just
            # try/except: if the client disconnects (e.g. tab closed, curl
            # killed), Starlette tears this generator down by throwing
            # GeneratorExit at whichever `yield` is currently suspended.
            # GeneratorExit isn't caught by `except Exception`, so without
            # the finally block vLLM's restart would be skipped entirely —
            # it'd be left stopped with no error surfaced, since the
            # `async with _generate_lock` still releases normally on any
            # exit path. A finally block's plain `await` calls still run
            # during GeneratorExit unwinding (only a further `yield` there
            # would be invalid), so `container.start()` is guaranteed here
            # regardless of whether anyone is still listening.
            generation_error: str | None = None
            restart_error: str | None = None
            image_bytes: bytes | None = None
            task: asyncio.Task | None = None
            try:
                yield sse("phase", phase="generating", elapsed=elapsed())
                try:
                    async with httpx.AsyncClient(timeout=None) as hc:
                        # imagegen's request schema uses `prompt` (matches
                        # the diffusers pipeline parameter name); our
                        # external API uses `prompt_text` (matches every
                        # other endpoint's convention) — translate here
                        # rather than renaming either side to match the
                        # other.
                        imagegen_payload = {
                            "prompt": req.prompt_text,
                            "width": req.width,
                            "height": req.height,
                            "steps": req.steps,
                            "guidance": req.guidance,
                            "seed": req.seed,
                        }
                        task = asyncio.ensure_future(
                            hc.post(
                                f"{settings.imagegen_url}/generate",
                                json=imagegen_payload,
                            )
                        )
                        while not task.done():
                            try:
                                await asyncio.wait_for(asyncio.shield(task), timeout=5)
                            except asyncio.TimeoutError:
                                yield sse(
                                    "heartbeat", phase="generating", elapsed=elapsed()
                                )
                        resp = task.result()
                        resp.raise_for_status()
                        image_bytes = resp.content
                except Exception as e:  # noqa: BLE001
                    generation_error = str(e)
                finally:
                    # If we're unwinding early (client disconnected), the
                    # imagegen call was `shield`ed from cancellation so it
                    # would otherwise keep running orphaned in the
                    # background — with the lock already released, a new
                    # request could pile another generation on top of it.
                    if task is not None and not task.done():
                        task.cancel()

                if image_bytes is not None:
                    try:
                        await asyncio.to_thread(
                            image_store.save_image,
                            settings.images_db_path,
                            settings.images_dir,
                            prompt_hash(req.prompt_text),
                            image_bytes,
                            req.width,
                            req.height,
                            req.steps,
                            req.guidance,
                            req.seed,
                        )
                    except Exception as e:  # noqa: BLE001
                        generation_error = generation_error or f"Failed to save image: {e}"

                # Phase 3: restart vllm — always, since we're the ones who
                # stopped it, regardless of whether generation succeeded.
                # Skipped entirely if phase 1 never stopped it.
                if container is not None:
                    yield sse("phase", phase="restarting_vllm", elapsed=elapsed())
            finally:
                if container is not None:
                    try:
                        await asyncio.to_thread(container.start)
                        await _wait_for_vllm_ready()
                    except Exception as e:  # noqa: BLE001
                        restart_error = str(e)

            if generation_error and restart_error:
                yield sse(
                    "error",
                    error=(
                        f"Generation failed ({generation_error}) and the chat "
                        f"model also failed to restart ({restart_error}) — "
                        "chat may be unavailable, check the server."
                    ),
                )
            elif generation_error:
                yield sse(
                    "error",
                    error=f"Generation failed: {generation_error}. Chat has been restored.",
                )
            elif restart_error:
                yield sse(
                    "error",
                    error=(
                        f"Image generated successfully, but the chat model failed "
                        f"to restart: {restart_error}. Chat may be unavailable."
                    ),
                )
            else:
                yield sse(
                    "done",
                    image_url=f"/generated-images/{prompt_hash(req.prompt_text)}",
                    elapsed=elapsed(),
                )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
                {
                    "messages": [HumanMessage(content=req.message)],
                    "target_mode": req.target_mode,
                },
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
