import datetime
import pathlib
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generated_images (
    prompt_hash TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    guidance REAL NOT NULL,
    seed INTEGER,
    created_at TEXT NOT NULL
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def save_image(
    db_path: str,
    images_dir: str,
    prompt_hash: str,
    image_bytes: bytes,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
) -> str:
    dir_path = pathlib.Path(images_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    image_path = dir_path / f"{prompt_hash}.png"
    image_path.write_bytes(image_bytes)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO generated_images
                (prompt_hash, image_path, width, height, steps, guidance, seed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prompt_hash) DO UPDATE SET
                image_path=excluded.image_path,
                width=excluded.width,
                height=excluded.height,
                steps=excluded.steps,
                guidance=excluded.guidance,
                seed=excluded.seed,
                created_at=excluded.created_at
            """,
            (
                prompt_hash,
                str(image_path),
                width,
                height,
                steps,
                guidance,
                seed,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
    return str(image_path)


def get_image_path(db_path: str, prompt_hash: str) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT image_path FROM generated_images WHERE prompt_hash = ?",
            (prompt_hash,),
        ).fetchone()
    return row[0] if row else None


def has_image(db_path: str, prompt_hash: str) -> bool:
    return get_image_path(db_path, prompt_hash) is not None
