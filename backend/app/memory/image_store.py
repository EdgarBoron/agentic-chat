import datetime
import pathlib
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generated_images (
    image_id TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL,
    image_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    guidance REAL NOT NULL,
    seed INTEGER,
    created_at TEXT NOT NULL
)
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_generated_images_prompt_hash
    ON generated_images(prompt_hash)
"""


def _migrate_if_needed(conn: sqlite3.Connection) -> None:
    """Upgrades the pre-multi-image schema (prompt_hash as PRIMARY KEY, one
    row per prompt) to the current one (image_id PRIMARY KEY, many rows per
    prompt_hash). Old rows keep their existing image_path (prompt_hash.png)
    as their new image_id — no files need renaming."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='generated_images'"
    ).fetchall()
    if not tables:
        return
    cols = [row[1] for row in conn.execute("PRAGMA table_info(generated_images)")]
    if "image_id" in cols:
        return
    conn.execute("ALTER TABLE generated_images RENAME TO generated_images_v1")
    conn.execute(_SCHEMA)
    conn.execute(_INDEX)
    conn.execute(
        """
        INSERT INTO generated_images
            (image_id, prompt_hash, image_path, width, height, steps, guidance, seed, created_at)
        SELECT prompt_hash, prompt_hash, image_path, width, height, steps, guidance, seed, created_at
        FROM generated_images_v1
        """
    )
    conn.execute("DROP TABLE generated_images_v1")
    conn.commit()


def _connect(db_path: str) -> sqlite3.Connection:
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _migrate_if_needed(conn)
    conn.execute(_SCHEMA)
    conn.execute(_INDEX)
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
    index: int,
) -> str:
    """Persists one generated image as its own file/row. Every call adds a
    new image_id rather than overwriting a prior one for the same prompt, so
    re-generating (or batch-generating) a prompt never destroys earlier
    results. `index` is the image's 1-based position within its generation
    request (2026-06-28_09-15-30_1.png), disambiguating multiple images
    generated within the same second."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_id = f"{timestamp}_{index}"
    dir_path = pathlib.Path(images_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    image_path = dir_path / f"{image_id}.png"
    image_path.write_bytes(image_bytes)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO generated_images
                (image_id, prompt_hash, image_path, width, height, steps, guidance, seed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
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
    return image_id


def get_image_path(db_path: str, image_id: str) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT image_path FROM generated_images WHERE image_id = ?",
            (image_id,),
        ).fetchone()
    return row[0] if row else None


def list_image_ids(db_path: str, prompt_hash: str) -> list[str]:
    """Newest first, for gallery display."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT image_id FROM generated_images
            WHERE prompt_hash = ?
            ORDER BY created_at DESC
            """,
            (prompt_hash,),
        ).fetchall()
    return [row[0] for row in rows]


