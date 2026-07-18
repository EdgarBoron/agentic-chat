# Local Agentic Flux/SD3 Prompt-Crafting Assistant — Implementation Plan

## Context

The repo currently contains only a research survey (`best-aproaches-to-create-an-agentic-ai-based-on-lo.md`) comparing options for building a locally-hosted agentic AI with a web chat UI. The goal is to turn that research into a real, running personal tool: a chat agent that helps craft **Flux/SD3-style natural-language txt2img prompts**, using web search for reference/trend info and a local knowledge base (static reference docs + history of past prompts) for grounding. The agent only produces prompt text — it does not call any image-generation backend.

Environment was verified before planning:
- Docker Desktop 28.4.0 (WSL2 backend) is already sufficient — `docker run --gpus all ... nvidia-smi` succeeded against the RTX 3090 (24GB VRAM). No extra host setup needed.
- A local HF Transformers snapshot of `Meta-Llama-3.1-8B-Instruct` already exists at `C:/Users/oball/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16` (bf16, ~16GB across 4 safetensors shards, valid config/tokenizer files, `chat_template` embedded). This will be served directly by vLLM — no re-download, no Ollama.

Decisions locked in with the user: Path C (full custom app), LangGraph backend, Next.js + Vercel AI SDK frontend, self-hosted SearXNG for web search, ChromaDB (embedded, CPU embeddings) for two collections — a static prompt-engineering reference library and an episodic history of past generated prompts. Single-user local tool: no auth, no Postgres/Redis, no nginx/monitoring, no MCP servers (plain Python tool functions are sufficient).

## Architecture

```
Next.js (chat UI, port 3000) → FastAPI/LangGraph backend (port 8001, SSE) → vLLM (port 8000, GPU)
                                              ↓                              
                                   SearXNG (port 8080)   Chroma (embedded, 2 collections in backend-data volume)
```

Docker Compose orchestrates 4 services: `vllm`, `searxng`, `backend`, `frontend`. Chroma runs embedded inside the backend process (persistent volume), not as a separate container.

## Repository layout

```
agentic-chat/
├── docker-compose.yml
├── .env / .env.example
├── data/reference/            # user drops style guides / cheatsheets here
├── searxng/settings.yml       # JSON format + limiter disabled
├── backend/
│   ├── Dockerfile, pyproject.toml
│   └── app/
│       ├── main.py            # FastAPI app, /chat/stream SSE, /healthz
│       ├── config.py          # pydantic-settings
│       ├── schemas.py
│       ├── agent/
│       │   ├── graph.py       # create_react_agent + AsyncSqliteSaver
│       │   ├── prompts.py     # Flux/SD3 system prompt
│       │   └── tools/         # web_search, reference_search, history_search, history_save
│       ├── memory/chroma_client.py   # PersistentClient, 2 collections, CPU embedding fn
│       └── ingest/ingest_reference.py  # rerunnable ingestion script
└── frontend/
    ├── Dockerfile, package.json, next.config.ts
    ├── app/page.tsx, app/layout.tsx
    ├── components/ChatMessage.tsx, ToolCallDisplay.tsx
    └── lib/backend-chat-transport.ts   # custom ChatTransport → FastAPI SSE
```

## Key implementation details

**vLLM service** — mounts the HF snapshot read-only, serves it directly (no re-download):
```yaml
vllm:
  image: vllm/vllm-openai:<pin-current-tag>
  ports: ["${VLLM_PORT:-8000}:8000"]
  volumes: ["${MODEL_SNAPSHOT_PATH}:/models/llama-3.1-8b-instruct:ro"]
  environment: [HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1]
  command: >
    --model /models/llama-3.1-8b-instruct --served-model-name llama-3.1-8b-instruct
    --enable-auto-tool-choice --tool-call-parser llama3_json
    --max-model-len ${VLLM_MAX_MODEL_LEN:-16384} --gpu-memory-utilization 0.90
  ipc: host
  deploy: {resources: {reservations: {devices: [{driver: nvidia, count: all, capabilities: [gpu]}]}}}
```
VRAM budget: weights ≈16.1GB bf16, leaving ~5.5GB of the 90%-utilization pool (~21.6GB) for KV cache/overhead — good for ~35-40K tokens of KV capacity. `--max-model-len 16384` is comfortably inside that, with headroom to spare — appropriate since prompt-crafting conversations are short.

**SearXNG** — JSON output is disabled by default; `searxng/settings.yml` must set `search.formats: [html, json]` **and** `server.limiter: false` (both required — limiter blocks non-browser JSON requests even with json format enabled). Verify with `curl "http://localhost:8080/search?q=test&format=json"`.

**Chroma** — `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` with `embedding_functions.DefaultEmbeddingFunction()` (ONNX all-MiniLM-L6-v2, CPU, no Ollama needed) for two collections: `prompt_reference` and `prompt_history`. Set `HOME`/`SENTENCE_TRANSFORMERS_HOME` inside the backend container to a path on the persistent volume so the ~80MB embedding model isn't re-downloaded on every rebuild.

**Agent** — `langgraph.prebuilt.create_react_agent` (ReAct loop, no custom StateGraph needed) with `ChatOpenAI(base_url=VLLM_BASE_URL, ...)` pointed at vLLM, `AsyncSqliteSaver` for thread persistence, and four tools:
- `web_search(query)` → SearXNG `/search?format=json`, top 5 results
- `search_prompt_reference(query)` → Chroma `prompt_reference` collection
- `search_prompt_history(query)` → Chroma `prompt_history` collection
- `save_prompt_to_history(prompt_text, note)` → upserts into `prompt_history`; system prompt instructs the agent to always call this immediately before presenting the final prompt

**System prompt** must explicitly: mandate natural-language Flux/SD3 prose (no `(word:1.3)` weighting, no comma-tag soup, no `masterpiece, best quality` boilerplate); set tool-use policy (reference library first, web search for current/trending info, history search for reuse, always save the final prompt); include a good-vs-bad few-shot example.

**FastAPI SSE endpoint** (`/chat/stream`) — mirrors the research doc's `astream_events` → SSE pattern, emitting `token` / `tool_start` / `tool_end` / `[DONE]` / `error` events with `X-Accel-Buffering: no`.

**Frontend** — the backend emits a custom SSE schema, not Vercel-AI-SDK-native `streamText`, so wire it with a custom `ChatTransport` (`backend-chat-transport.ts`) implementing `sendMessages()` that fetches `/chat/stream` and translates events into `UIMessageChunk`s for `useChat`. **Verify exact `UIMessageChunk` member names against the installed `ai` package's types before finalizing** — they've shifted across v5 minors; if unstable, fall back to a small hand-rolled streaming hook instead of fighting the SDK's types. Browser talks directly to FastAPI (CORS-enabled); no Next.js API route needed. `ToolCallDisplay.tsx` renders the four tools with human-readable labels ("Searching the web…", "Checking reference library…", etc.).

**Reference ingestion** — `backend/app/ingest/ingest_reference.py`, rerunnable/idempotent via content-hash chunk IDs + `upsert`, chunks `.md`/`.txt` files under `data/reference/` with `RecursiveCharacterTextSplitter`. Run via `docker compose run --rm backend python -m app.ingest.ingest_reference`.

## Verification plan

1. `docker compose config` — sanity-check env substitution.
2. Bring up `vllm` alone; watch logs for clean model load (no CUDA/OOM errors).
3. `docker compose exec vllm nvidia-smi` + host `nvidia-smi` — confirm ~16-20GB VRAM in use.
4. `curl localhost:8000/v1/models` and a `/v1/chat/completions` round trip.
5. Bring up `searxng`; `curl "localhost:8080/search?q=test&format=json"` returns JSON (not 403).
6. Bring up `backend`; `curl localhost:8001/healthz`; check logs for Chroma collections created + embedding model cached.
7. Drop a sample `.md` with a distinctive made-up technique term into `data/reference/`; run the ingestion script; confirm chunk count logged.
8. Bring up `frontend`; open `localhost:3000`; send a prompt request; confirm tokens stream and output reads as natural-language Flux-style prose with no legacy SD tag syntax.
9. Ask something requiring current/trending info; confirm `web_search` fires in `ToolCallDisplay` and in SearXNG logs.
10. Ask something matching the seeded reference doc's distinctive term; confirm `search_prompt_reference` fires and the output actually uses that vocabulary.
11. Complete one full turn (triggers `save_prompt_to_history`); start a related follow-up; confirm `search_prompt_history` fires and adapts the earlier prompt.
12. `docker compose restart backend`; continue the same thread in the UI; confirm prior turns persisted (SQLite checkpoint survived).

## Open items to resolve during implementation

- Pin an actual current `vllm/vllm-openai` image tag (placeholder used in the plan).
- Confirm the installed `ai` npm package's `UIMessageChunk` type names before finalizing the custom transport; use the hand-rolled-hook fallback if the SDK's tool-call chunk types prove unstable.
- Decide whether `data/reference/` content should be git-tracked or left local-only (personal reference material).
