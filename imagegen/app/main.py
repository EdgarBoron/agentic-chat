import asyncio
import io

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.pipeline import build_pipeline, generate

app = FastAPI()

_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance: float = 3.5
    seed: int | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if _lock.locked():
        raise HTTPException(status_code=409, detail="A generation is already in progress")

    async with _lock:
        try:
            image = await asyncio.to_thread(_run_generation, req)
        except Exception as e:  # noqa: BLE001 - surface the real error to the caller
            raise HTTPException(status_code=500, detail=str(e)) from e

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _run_generation(req: GenerateRequest):
    # Built and torn down fresh for every single request — imagegen must
    # be stateless-at-rest so vLLM can safely reclaim the GPU immediately
    # after this returns, not just between logically separate sessions.
    pipe = build_pipeline()
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
        del pipe
        torch.cuda.empty_cache()
