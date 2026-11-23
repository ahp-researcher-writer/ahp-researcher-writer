"""Load resources from a folder.

Standalone for now: the researcher's output is a file we copy into a
folder. Later, the AHP receiver will fill the same `resources` dict from
the handoff package.
"""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED = {".json", ".md", ".txt"}


def _describe(path: Path, text: str) -> str:
    if path.suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "findings" in data:
                return (
                    f"Research brief on: {data.get('objective', 'unknown topic')} "
                    f"({len(data['findings'])} findings)"
                )
        except json.JSONDecodeError:
            pass
    first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:120] or path.name


def load_resources(folder: str | Path) -> dict:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Resource folder not found: {folder}")
    resources = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED and path.is_file():
            text = path.read_text(encoding="utf-8")
            resources[path.name] = {"description": _describe(path, text), "text": text}
    return resources
