# Agentic Chat — Local Flux/SD3 Prompt-Crafting Assistant

A locally hosted agentic chat assistant that helps craft **Flux/SD3-style
natural-language txt2img prompts**. It uses a local LLM, live web search,
and a persistent knowledge base to research and produce prompt text. It
can also **render a crafted prompt into an actual image** on demand, via a
self-contained native-Python txt2img pipeline (no ComfyUI dependency).

## Architecture

```
Browser
  │  (SSE)
  ▼
Next.js frontend (:3000)
  │  POST /chat/stream              │  POST /generate-image/stream
  ▼                                 ▼
FastAPI + LangGraph backend (:8001) ──stop/restart (docker.sock)──┐
  │                              │                                 │
  │  OpenAI-compatible API       │  tool calls        HTTP POST    │
  ▼                              ▼                      ▼          ▼
vLLM (:8000, GPU)          SearXNG (:8080)   ChromaDB   imagegen (GPU,
serving Llama-3.1-8B       web search        (embedded, internal-only)
                                              backend-   Z-Image-Turbo
                                              data vol)  txt2img via
                                              - prompt_  diffusers
                                                reference
                                              - artist_
                                                styles
                                              - prompt_
                                                history
```

Everything runs as five Docker Compose services: `vllm`, `imagegen`,
`searxng`, `backend`, `frontend`. ChromaDB runs **embedded inside the
backend process** (not a separate container) with a persistent Docker
volume.

`vllm` and `imagegen` share the single GPU and cannot both hold their full
working set in VRAM at once, so they never run concurrently: the backend
stops the `vllm` container (via a `docker.sock` mount) before calling
`imagegen`, and restarts it afterward regardless of outcome. See "Image
generation" under **Using the app** for the tradeoffs this implies.

### Request flow

1. User sends a message from the Next.js chat UI.
2. The backend runs a LangGraph ReAct agent (`create_react_agent`) against
   the local model served by vLLM, capped at a 3-tool-call budget per turn
   (enforced by the system prompt) and a hard `recursion_limit=15` on the
   graph itself as a backstop against runaway tool loops.
3. The agent may call up to a few tools per turn:
   - `search_prompt_reference` — semantic search over a local library of
     prompt-engineering reference docs (`data/reference/`, excluding the
     artist/photographer style catalog below)
   - `search_artist_styles` — semantic search over a local catalog of named
     artist/photographer visual styles
     (`data/reference/artist-photographer-styles.md`)
   - `web_search` — live web search via self-hosted SearXNG, for
     current/trending information
   - `search_prompt_history` — semantic search over previously crafted
     prompts, to reuse/adapt past results
   - `save_prompt_to_history` — persists the final prompt for future reuse
     (idempotent: identical prompt text upserts onto the same entry rather
     than duplicating)
4. The backend streams tokens and tool-call events (each tagged with a
   `call_id`, start time, and measured duration) back to the browser over
   Server-Sent Events. If the agent errors out (e.g. hits the recursion
   limit), a friendly error event is streamed instead of failing silently.
5. Conversation state is checkpointed to SQLite per `thread_id`. The
   frontend persists its `thread_id` in `localStorage` and rehydrates the
   visible conversation from the backend on load, so both a page reload
   and a backend restart resume the same conversation.

## Frameworks & tools used

| Layer | Choice | Why |
|---|---|---|
| Model serving | [vLLM](https://github.com/vllm-project/vllm) (`vllm/vllm-openai` Docker image) | Serves the already-downloaded local HF Transformers snapshot directly (no re-download/conversion), OpenAI-compatible API, native tool-calling support |
| Model | `Meta-Llama-3.1-8B-Instruct` (local HF snapshot, bf16) | Already present on disk; strong tool-calling support via vLLM's `llama3_json` parser |
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`create_react_agent`) + [LangChain](https://github.com/langchain-ai/langchain) core | ReAct tool-calling loop with built-in SQLite checkpointing for conversation persistence |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Async SSE streaming endpoint (`/chat/stream`) |
| Web search | [SearXNG](https://github.com/searxng/searxng) (self-hosted) | Private metasearch, no API key or third-party dependency |
| Vector store / RAG | [ChromaDB](https://github.com/chroma-core/chroma), embedded persistent client | Three collections (`prompt_reference`, `artist_styles`, `prompt_history`); local CPU ONNX embeddings (`all-MiniLM-L6-v2`), no extra embedding server needed |
| Frontend | [Next.js](https://nextjs.org/) 16 + [Vercel AI SDK](https://sdk.vercel.ai/) (`ai`, `@ai-sdk/react`) | `useChat` hook driven by a custom `ChatTransport` that adapts the backend's SSE event schema into `UIMessageChunk`s |
| Image generation | [diffusers](https://github.com/huggingface/diffusers) + [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Alibaba Tongyi-MAI, 6.15B param S3-DiT, Qwen3 text encoder) | Native Python txt2img — no ComfyUI dependency; `enable_model_cpu_offload()` keeps peak VRAM manageable on a single GPU shared with vLLM |
| Container orchestration | [docker-py](https://github.com/docker/docker-py), via a `docker.sock` mount into `backend` | Lets the backend stop/restart the sibling `vllm` container around each image generation |
| Orchestration | Docker Compose, with GPU passthrough (`--gpus`) for `vllm` and `imagegen` | Single-command local stack |

## Prerequisites

- Docker Desktop with the WSL2 backend and GPU passthrough enabled
  (verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`)
- An NVIDIA GPU with enough VRAM for the model (this project was built and
  tested against an RTX 3090, 24GB VRAM; bf16 Llama-3.1-8B weights use
  ~15GB, leaving headroom for a 16K-token KV cache). Image generation
  reuses the same GPU (vLLM is stopped while it runs) and needs a
  transformer that fits comfortably alongside CPU-offloaded components —
  Z-Image-Turbo's 6.15B params fit well within a 24GB card.
- A local HF Transformers snapshot of `Meta-Llama-3.1-8B-Instruct` (or
  point `MODEL_SNAPSHOT_PATH` at a different local Llama-3.x snapshot)
- A local `zImageTurbo_turbo.safetensors` checkpoint (or point
  `IMAGEGEN_DIFFUSION_MODELS_PATH`/`IMAGEGEN_UNET_FILENAME` at a different
  Z-Image-Turbo checkpoint) — config/tokenizer/VAE/text-encoder files are
  fetched automatically from the ungated `Tongyi-MAI/Z-Image-Turbo` HF repo
  on first use and cached in a Docker volume
- If running under WSL2, kernel ≥ 4.19.121 (check with `wsl -- uname -r`)
  for vLLM's pinned-memory support
- **WSL2 memory limit ≥ 32GB** (`.wslconfig`, `[wsl2] memory=`). Image
  generation's CPU-offloaded components (text encoder, transformer, VAE)
  need ~15GB resident in RAM at once; a tighter limit shared across all
  five containers risks the Linux OOM killer terminating the `imagegen`
  process mid-generation.

## Starting the stack

1. Copy the env template and point it at your local model snapshot and
   diffusion checkpoint:

   ```
   cp .env.example .env
   # edit .env: set MODEL_SNAPSHOT_PATH to your local HF snapshot directory
   # edit .env: set IMAGEGEN_DIFFUSION_MODELS_PATH to the directory
   #   containing your Z-Image-Turbo checkpoint (IMAGEGEN_UNET_FILENAME
   #   defaults to zImageTurbo_turbo.safetensors)
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
   by content hash). Edit `data/reference/artist-photographer-styles.md` to
   add/edit named artist/photographer styles, then ingest it into its own
   collection separately:

   ```
   docker compose run --rm backend python -m app.ingest.ingest_styles
   ```

4. Open the chat UI:

   ```
   http://localhost:3000
   ```

## Using the app

- **Chat** (`/`) — describe the image you want; the crafted prompt comes
  back in a distinct block with **Copy**, **Store**, and **Generate**
  buttons. The left sidebar ("Actions") lists every tool call made during
  the conversation with its start time and duration, live-updating as
  calls happen.
- **Image generation** — click **Generate** on a prompt block, confirm the
  dialog (generation pauses chat for several minutes — see below), and
  watch live phase progress (pausing chat model → generating → resuming
  chat model) with an elapsed-time counter. The finished image appears
  inline under the prompt and is linked to that prompt's history entry
  (works in either click order — Generate-then-Store or
  Store-then-Generate). Only one generation runs at a time; a second
  Generate click while one is in flight is rejected. Each generation costs
  real chat downtime: `vllm` is stopped to free VRAM, the image is
  rendered (a handful of diffusion steps takes low single-digit minutes,
  plus a one-time few-minutes GPU warmup on the very first generation),
  and `vllm` is reloaded afterward — reload time varies with available RAM
  (see the WSL2 memory prerequisite above).
- **Prompt history** (`/prompt-history`) — browse every prompt the agent
  has saved, newest first, with timestamps, any notes, and the generated
  image if one exists for that prompt. Linked from the chat header.
- **Slash commands** — type `/` in the chat input for an autocomplete
  dropdown (↑/↓ to navigate, Enter/click to run):
  - `/help` — lists available commands
  - `/clear` — starts a brand-new conversation (new thread id; the old
    conversation stays in the backend's SQLite checkpoint but is no
    longer reachable from the UI)
  - `/refine` — asks the agent to enhance the current prompt with more
    vivid, specific visual detail while keeping the same subject/intent
  - `/consistency` — asks the agent to check the current prompt for
    internal contradictions (e.g. clashing lighting/time of day, mismatched
    subject description, conflicting style cues) without changing it; if
    it finds any, it lists them with numbered resolution options and waits
    for you to pick before editing anything
- Errors (e.g. the agent hitting its tool-call recursion limit) surface as
  a dismissible red banner above the input rather than failing silently.

## Services & ports

| Service | Port | Purpose |
|---|---|---|
| `frontend` | 3000 | Chat UI (`/`) and prompt history page (`/prompt-history`) |
| `backend` | 8001 | FastAPI — see endpoints below |
| `vllm` | 8000 | OpenAI-compatible LLM API |
| `imagegen` | — (internal only) | txt2img HTTP API, called only by `backend` |
| `searxng` | 8080 | Web search JSON API |

### Backend endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness check |
| `POST /chat/stream` | SSE chat endpoint; body `{message, thread_id}` |
| `GET /chat/history/{thread_id}` | Reconstructs the user/assistant text exchange for a thread from the LangGraph checkpoint (used to rehydrate the UI on load) |
| `GET /prompt-history` | All saved prompts from the `prompt_history` Chroma collection, newest first, each with an `image_url` if a generated image exists |
| `POST /prompt-history` | Saves a prompt (and optional note) to history; body `{prompt_text, note?}` |
| `POST /generate-image/stream` | SSE orchestration endpoint: stops `vllm`, calls `imagegen`, saves the result, restarts `vllm`. Body `{prompt_text, width?, height?, steps?, guidance?, seed?}`. Rejects a second concurrent call with `409` |
| `GET /generated-images/{hash}` | Serves a generated PNG by its prompt-text hash (`404` if none exists) |

## Useful commands

```
docker compose ps                          # service status
docker compose logs -f backend             # tail backend logs
docker compose restart backend             # restart backend (checkpoints persist)
docker compose down                        # stop everything (volumes kept)
```

## Notes / known limitations

- Single-user local tool: no auth, no multi-tenant isolation.
- The 8B model is capable but not perfect at agentic tool-use discipline.
  In longer/ambiguous conversations it can still occasionally exhaust the
  tool-call budget and hit the graph's recursion limit — this now surfaces
  as a visible error banner (see "Using the app") instead of silently
  producing no output.
- **The reference library ships empty.** `search_prompt_reference` and
  `search_artist_styles` only ever return whatever's actually been
  ingested — with zero or one document in a collection, every query
  returns the same (only) result, which looks like a bug but isn't. Run
  both ingestion commands (see step 3 under "Starting the stack") to
  populate them.
- `data/reference/` and the Chroma persistence volume (`backend-data`) are
  the two places the assistant's "knowledge" lives — back those up if you
  want to keep accumulated history/reference/style material.
- Only the text conversation is rehydrated on reload/reconnect —
  historical tool-call activity (the Actions Pane) is not reconstructed
  from past turns, only new calls going forward.
- vLLM's model load time is variable depending on available RAM for the OS
  page cache (it streams checkpoint shards rather than needing the full
  ~15GB resident in RAM at once, but with less headroom, loads take
  noticeably longer). Keeping Docker Desktop's WSL2 memory limit
  (`.wslconfig`, `[wsl2] memory=`) at **32GB or more** avoids both this
  slowdown and the OOM risk described in Prerequisites.
- **Image generation is single-user, one-at-a-time, txt2img only**: no
  batch generation, no img2img, no LoRA support, no generation queue. This
  is a deliberate scope decision, not a current limitation to be lifted.
- **Image generation and chat cannot run concurrently** — they share the
  one GPU, so every generation pauses chat for several minutes while
  `vllm` is stopped and reloaded. If a client disconnects mid-generation
  (tab closed, network drop), `vllm` still gets restarted (the restart
  logic runs in a `finally` block specifically to guarantee this), but the
  in-flight generation itself is cancelled and its output is lost.
- The `backend` container has the Docker socket (`/var/run/docker.sock`)
  mounted in order to stop/restart the sibling `vllm` container. This
  grants `backend` full control over the Docker daemon (not scoped to
  just `vllm`) — an accepted tradeoff for this single-user, no-external-
  exposure tool, not something to expose beyond localhost.
