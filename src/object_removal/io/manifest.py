from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_method_manifest(dir_path: Path, *, stage: str, method: str, params: Mapping[str, Any]) -> Path:
    out = dir_path / "method.json"
    payload = {
        "timestamp_utc": utc_now_iso(),
        "stage": stage,
        "method": method,
        "params": dict(params),
    }
    write_json(out, payload)
    return out

