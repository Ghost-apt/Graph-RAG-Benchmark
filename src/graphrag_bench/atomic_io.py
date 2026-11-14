"""Atomic and thread-safe filesystem utilities for Graph RAG Benchmark.

Guarantees atomic file writes using temporary staging and os.replace,
preventing file corruption during unexpected power loss, crashes, or timeouts.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(file_path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Atomically write text content to target file path."""
    dest = Path(file_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in the same directory to guarantee same filesystem partition
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, delete=False, encoding=encoding) as tmp:
        tmp.write(content)
        tmp_name = tmp.name

    try:
        os.replace(tmp_name, dest)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_write_json(
    file_path: Path | str,
    data: Any,
    indent: int = 2,
    encoding: str = "utf-8",
) -> None:
    """Atomically serialize and write data as JSON to target file path."""
    content = json.dumps(data, indent=indent, default=str)
    atomic_write_text(file_path, content, encoding=encoding)
