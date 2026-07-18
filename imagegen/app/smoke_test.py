"""Standalone verification of the Z-Image loading approach, run before
wiring anything else up:

    docker compose run --rm imagegen python -m app.smoke_test

Loads the transformer from the local checkpoint and builds the full
pipeline, reporting GPU memory at each stage.
"""

import torch

from app import pipeline


def report(label: str) -> None:
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[{label}] allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB")


def main() -> None:
    print("Downloading/locating remote components (text encoder, vae, tokenizer, scheduler)...")
    repo_dir = pipeline.ensure_remote_components()
    print(f"repo_dir={repo_dir}")

    print("\nLoading transformer from local checkpoint...")
    transformer = pipeline.load_transformer(repo_dir)
    param_count = sum(p.numel() for p in transformer.parameters())
    print(f"transformer param count: {param_count / 1e9:.2f}B")
    transformer.to("cuda")
    report("transformer on GPU")
    del transformer
    torch.cuda.empty_cache()
    report("after freeing transformer")

    print("\nBuilding full pipeline (downloads text_encoder/vae weights if not cached)...")
    pipe = pipeline.build_pipeline()
    print("Pipeline assembled successfully.")

    print("\nRunning a short test generation (4 steps, 512x512)...")
    image = pipeline.generate(
        pipe, prompt="a red apple on a wooden table", width=512, height=512,
        steps=4, guidance=1.0, seed=0,
    )
    report("after generation")
    image.save("/tmp/smoke_test_output.png")
    print("Saved /tmp/smoke_test_output.png inside the container.")
    print("\nAll components loaded and a test image was generated successfully.")


if __name__ == "__main__":
    main()
