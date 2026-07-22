import logging
import os

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

UNET_DIR = os.environ["IMAGEGEN_UNET_DIR"]
UNET_FILENAME = os.environ.get("IMAGEGEN_UNET_FILENAME", "zImageTurbo_turbo.safetensors")
HF_CACHE_DIR = os.environ.get("IMAGEGEN_HF_CACHE_DIR", "/hf-cache")

# Official, ungated Alibaba Tongyi-MAI release. The local checkpoint
# (zImageTurbo_turbo.safetensors) provides only the transformer weights —
# its tensor shapes were confirmed to match this repo's transformer/config.json
# (dim=3840, cap_feat_dim=2560). Text encoder (Qwen3), tokenizer, VAE, and
# scheduler aren't available locally, so those come from this repo.
CONFIG_REPO = "Tongyi-MAI/Z-Image-Turbo"

DTYPE = torch.bfloat16


def ensure_remote_components() -> str:
    """Download everything except the transformer's weight shards — we
    supply those ourselves from the local safetensors file. Cached under
    HF_CACHE_DIR across calls/restarts (several GB, one-time cost)."""
    logger.debug("Resolving remote components from %s (cache_dir=%s)", CONFIG_REPO, HF_CACHE_DIR)
    repo_dir = snapshot_download(
        repo_id=CONFIG_REPO,
        cache_dir=HF_CACHE_DIR,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "vae/*",
            "transformer/config.json",
        ],
    )
    logger.debug("Remote components resolved to %s", repo_dir)
    return repo_dir


def load_transformer(repo_dir: str) -> ZImageTransformer2DModel:
    path = os.path.join(UNET_DIR, UNET_FILENAME)
    logger.debug("Loading transformer from %s", path)
    transformer = ZImageTransformer2DModel.from_single_file(
        path, config=repo_dir, subfolder="transformer", torch_dtype=DTYPE
    )
    logger.debug("Transformer loaded")
    return transformer


def build_pipeline() -> ZImagePipeline:
    repo_dir = ensure_remote_components()
    transformer = load_transformer(repo_dir)

    # transformer= overrides just that component; text_encoder/tokenizer/
    # vae/scheduler load from the downloaded repo snapshot as usual.
    logger.debug("Assembling ZImagePipeline")
    pipe = ZImagePipeline.from_pretrained(
        repo_dir, transformer=transformer, torch_dtype=DTYPE
    )
    # Swaps each component onto GPU only for its active stage rather than
    # holding everything resident simultaneously — required to fit safely
    # alongside vLLM's VRAM needs, and consistent with imagegen being
    # stateless-at-rest (nothing GPU-resident once a request finishes).
    pipe.enable_model_cpu_offload()
    logger.debug("Pipeline assembled with CPU offload enabled")
    return pipe


def generate(
    pipe: ZImagePipeline,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    # 0.0 disables classifier-free guidance entirely (single forward pass,
    # no negative-prompt encode) — this checkpoint is a guidance-distilled
    # Turbo build, so upstream's own example uses guidance_scale=0.0, and
    # it's what the reference ComfyUI workflow does too (its BasicGuider
    # only ever sets a positive conditioning). Any value > 0 makes the
    # pipeline double every transformer forward pass for true CFG.
    guidance: float,
    seed: int | None,
):
    generator = torch.Generator(device="cpu")
    # torch.Generator() defaults to a fixed constant seed until explicitly
    # (re)seeded — it is NOT randomized on construction. Without this,
    # every seed=None request silently reused the same noise.
    if seed is not None:
        generator = generator.manual_seed(seed)
    else:
        generator.seed()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    return result.images[0]
