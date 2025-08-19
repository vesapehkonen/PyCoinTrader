# paths.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timezone
import os

# Add markers you actually have at your repo root
_PROJECT_MARKERS = ("pyproject.toml", ".git", "models", "scripts", "src", "data")

@lru_cache(maxsize=1)
def project_root(start: Path | None = None) -> Path:
    """Find the project root. Prefers $PROJECT_ROOT, otherwise walks upward.
    Falls back to CWD (never /)."""
    if env := os.getenv("PROJECT_ROOT"):
        return Path(env).expanduser().resolve()

    start = (start or Path(__file__)).resolve()
    for p in (start, *start.parents):
        if any((p / m).exists() for m in _PROJECT_MARKERS):
            return p

    # Safe fallback: current working directory (not "/")
    return Path.cwd().resolve()

def data_dir() -> Path:
    return project_root() / "data"

def outputs_dir() -> Path:
    return data_dir() / "outputs"

def config_dir() -> Path:
    return project_root() / "config"

def models_dir() -> Path:
    return project_root() / "models"

def _next_numeric_id(base: Path) -> int:
    """Return the next integer id (1,2,3,...) based on existing numeric subdirs."""
    base.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for p in base.iterdir():
        if p.is_dir():
            try:
                n = int(p.name)  # only count pure-numeric folder names
            except ValueError:
                continue
            if n > max_id:
                max_id = n
    return max_id + 1 if max_id >= 0 else 1

def make_run_dir(base: Path | None = None,
                 run_id: int | None = None,
                 label: str | None = None):
    """
    Create and return a per-run output directory using numeric ids.
    Returns (run_dir: Path, run_id: str, started_at: datetime).

    - Folder names are purely numeric (no label in the name).
    - If 'run_id' is given, it is used as-is (must be int).
    - A 'timestamp.txt' file is written inside the run folder.
    - 'label' (if provided) is written into 'timestamp.txt' for reference.
    """
    base = (base or outputs_dir()).resolve()
    base.mkdir(parents=True, exist_ok=True)

    # choose id
    if run_id is None:
        candidate = _next_numeric_id(base)
        # race-safe: try to mkdir; if exists, bump and retry
        while True:
            run_dir = base / str(candidate)
            try:
                run_dir.mkdir(exist_ok=False)  # create the unique folder atomically
                picked_id = candidate
                break
            except FileExistsError:
                candidate += 1
    else:
        picked_id = int(run_id)
        run_dir = base / str(picked_id)
        run_dir.mkdir(parents=True, exist_ok=False)  # fail if it already exists

    started_at = datetime.now(timezone.utc)

    # write timestamp file (and optional label)
    (run_dir / "timestamp.txt").write_text(
        f"{started_at.isoformat()}\n" + (f"label={label}\n" if label else ""),
        encoding="utf-8"
    )

    return run_dir, str(picked_id), started_at
