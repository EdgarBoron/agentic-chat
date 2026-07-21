import asyncio
import io
import logging

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.pipeline import build_pipeline, generate

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1088
    height: int = 1600
    steps: int = 10
    guidance: float = 0.0
    seed: int | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if _lock.locked():
        logger.debug("Rejecting request, a generation is already in progress")
        raise HTTPException(status_code=409, detail="A generation is already in progress")

    logger.debug(
        "Received generate request: prompt=%r width=%d height=%d steps=%d guidance=%s seed=%s",
        req.prompt, req.width, req.height, req.steps, req.guidance, req.seed,
    )
    async with _lock:
        try:
            image = await asyncio.to_thread(_run_generation, req)
        except Exception as e:  # noqa: BLE001 - surface the real error to the caller
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    logger.debug("Generation complete")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


_pipe = None


def _get_pipeline():
    # Built once and cached from then on. GPU reclaim for vLLM doesn't
    # depend on rebuilding this object each time: enable_model_cpu_offload()
    # already moves weights back to CPU after every forward pass, and the
    # backend separately stops the vllm container outright before calling
    # here. Reusing the pipeline just skips redundant disk I/O.
    global _pipe
    if _pipe is None:
        logger.debug("No cached pipeline, building one now")
        _pipe = build_pipeline()
        logger.debug("Pipeline built and cached")
    return _pipe


def _run_generation(req: GenerateRequest):
    pipe = _get_pipeline()
    try:
        return generate(
            pipe,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
        )
    finally:
        torch.cuda.empty_cache()
