# Agentic Chat — Local Flux/SD3 Prompt-Crafting Assistant

A locally hosted agentic chat assistant that helps craft **Flux/SD3-style
natural-language txt2img prompts**. It uses a local LLM, live web search,
and a persistent knowledge base to research and produce prompt text — it
does not call any image-generation backend itself.

## Architecture

```
Browser
  │  (SSE)
  ▼
Next.js frontend (:3000)
  │  POST /chat/stream
  ▼
FastAPI + LangGraph backend (:8001)
  │                              │
  │  OpenAI-compatible API       │  tool calls
  ▼                              ▼
vLLM (:8000, GPU)          SearXNG (:8080)   ChromaDB (embedded,
serving Llama-3.1-8B       web search          in backend-data volume)
                                                 - prompt_reference
                                                 - prompt_history
```

Everything runs as four Docker Compose services: `vllm`, `searxng`,
`backend`, `frontend`. ChromaDB runs **embedded inside the backend
process** (not a separate container) with a persistent Docker volume.

### Request flow

1. User sends a message from the Next.js chat UI.
2. The backend runs a LangGraph ReAct agent (`create_react_agent`) against
   the local model served by vLLM.
3. The agent may call up to a few tools per turn:
   - `search_prompt_reference` — semantic search over a local library of
     prompt-engineering reference docs (`data/reference/`)
   - `web_search` — live web search via self-hosted SearXNG, for
     current/trending information
   - `search_prompt_history` — semantic search over previously crafted
     prompts, to reuse/adapt past results
   - `save_prompt_to_history` — persists the final prompt for future reuse
4. The backend streams tokens and tool-call events back to the browser
   over Server-Sent Events; the frontend renders live text plus a
   tool-call timeline.
5. Conversation state is checkpointed to SQLite per `thread_id`, so a
   conversation survives a backend restart.

## Frameworks & tools used

| Layer | Choice | Why |
|---|---|---|
| Model serving | [vLLM](https://github.com/vllm-project/vllm) (`vllm/vllm-openai` Docker image) | Serves the already-downloaded local HF Transformers snapshot directly (no re-download/conversion), OpenAI-compatible API, native tool-calling support |
| Model | `Meta-Llama-3.1-8B-Instruct` (local HF snapshot, bf16) | Already present on disk; strong tool-calling support via vLLM's `llama3_json` parser |
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`create_react_agent`) + [LangChain](https://github.com/langchain-ai/langchain) core | ReAct tool-calling loop with built-in SQLite checkpointing for conversation persistence |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Async SSE streaming endpoint (`/chat/stream`) |
| Web search | [SearXNG](https://github.com/searxng/searxng) (self-hosted) | Private metasearch, no API key or third-party dependency |
| Vector store / RAG | [ChromaDB](https://github.com/chroma-core/chroma), embedded persistent client | Two collections (`prompt_reference`, `prompt_history`); local CPU ONNX embeddings (`all-MiniLM-L6-v2`), no extra embedding server needed |
| Frontend | [Next.js](https://nextjs.org/) 16 + [Vercel AI SDK](https://sdk.vercel.ai/) (`ai`, `@ai-sdk/react`) | `useChat` hook driven by a custom `ChatTransport` that adapts the backend's SSE event schema into `UIMessageChunk`s |
| Orchestration | Docker Compose, with GPU passthrough (`--gpus`) for vLLM | Single-command local stack |

## Prerequisites

- Docker Desktop with the WSL2 backend and GPU passthrough enabled
  (verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`)
- An NVIDIA GPU with enough VRAM for the model (this project was built and
  tested against an RTX 3090, 24GB VRAM; bf16 Llama-3.1-8B weights use
  ~15GB, leaving headroom for a 16K-token KV cache)
- A local HF Transformers snapshot of `Meta-Llama-3.1-8B-Instruct` (or
  point `MODEL_SNAPSHOT_PATH` at a different local Llama-3.x snapshot)
- If running under WSL2, kernel ≥ 4.19.121 (check with `wsl -- uname -r`)
  for vLLM's pinned-memory support

## Starting the stack

1. Copy the env template and point it at your local model snapshot:

   ```
   cp .env.example .env
   # edit .env: set MODEL_SNAPSHOT_PATH to your local HF snapshot directory
   ```

2. Bring everything up:

   ```
   docker compose up -d
   ```

   First start will pull/build images and load the model into VRAM — the
   `vllm` service alone can take several minutes (weight loading + CUDA
   graph capture). Watch progress with:

   ```
   docker compose logs -f vllm
   ```

   It's ready once this returns a model list:

   ```
   curl http://localhost:8000/v1/models
   ```

3. (Optional) Seed the reference library. Drop `.md`/`.txt` files with
   prompting techniques/style guides into `data/reference/`, then run:

   ```
   docker compose run --rm backend python -m app.ingest.ingest_reference
   ```

   Re-run any time after adding/editing files — it's idempotent (upserts
   by content hash).

4. Open the chat UI:

   ```
   http://localhost:3000
   ```

## Services & ports

| Service | Port | Purpose |
|---|---|---|
| `frontend` | 3000 | Chat UI |
| `backend` | 8001 | FastAPI (`/chat/stream`, `/healthz`) |
| `vllm` | 8000 | OpenAI-compatible LLM API |
| `searxng` | 8080 | Web search JSON API |

## Useful commands

```
docker compose ps                          # service status
docker compose logs -f backend             # tail backend logs
docker compose restart backend             # restart backend (checkpoints persist)
docker compose down                        # stop everything (volumes kept)
```

## Notes / known limitations

- Single-user local tool: no auth, no multi-tenant isolation.
- The 8B model is capable but not perfect at agentic tool-use discipline;
  the system prompt (`backend/app/agent/prompts.py`) enforces a tool-call
  budget per turn to prevent runaway tool-loop behavior.
- `data/reference/` and the Chroma persistence volume are the two places
  the assistant's "knowledge" lives — back those up if you want to keep
  accumulated history/reference material.
