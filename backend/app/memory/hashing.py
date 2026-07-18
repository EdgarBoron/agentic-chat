import hashlib


def prompt_hash(prompt_text: str) -> str:
    """Content-derived id for a prompt's text. Used as both the Chroma
    history doc id and the generated-image lookup key, so a generated
    image and a stored history entry for the identical prompt text
    always resolve to the same id regardless of which happened first.
    """
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:24]
