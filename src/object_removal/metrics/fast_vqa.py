"""Optional FastVQA / FasterVQA scoring via the upstream FAST-VQA-and-FasterVQA repo.

Requires a local clone of https://github.com/VQAssessment/FAST-VQA-and-FasterVQA with
dependencies installed (see that repo's README) and checkpoint paths configured in its
YAML ``test_load_path`` fields. This module muxes inpaint frame folders to a short MP4,
then runs ``<python> vqa.py -m <model> -v <mp4> -d <device>`` inside that repo root
(``<python>`` defaults to ``FAST_VQA_PYTHON`` or ``sys.executable``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

_SCORE_RE = re.compile(
    r"The quality score of the video \(range \[0,1\]\) is\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.?",
    re.IGNORECASE,
)


def resolve_fast_vqa_root(*, repo_root: Path, raw_root: str | Path | None = None) -> Optional[Path]:
    raw = str(raw_root or "").strip()
    if raw:
        root = Path(raw)
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        return root

    env_raw = os.environ.get("FAST_VQA_ROOT", "").strip()
    if env_raw:
        root = Path(env_raw)
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        return root

    for rel in (
        "modules/FAST-VQA-and-FasterVQA",
        "third_party/FAST-VQA-and-FasterVQA",
    ):
        cand = (repo_root / rel).resolve()
        if (cand / "vqa.py").is_file():
            return cand
    return None


def resolve_fast_vqa_python(explicit: str | None = None) -> str:
    """Interpreter used to run upstream ``vqa.py`` (may differ from ``sys.executable``)."""
    raw = (explicit or "").strip() or os.environ.get("FAST_VQA_PYTHON", "").strip()
    return raw if raw else sys.executable


def _format_subprocess_fail(returncode: int | None, tail: str) -> str:
    if returncode is not None and returncode < 0:
        sig = -returncode
        return (
            f"fast_vqa_subprocess_rc={returncode} (signal {sig}; "
            f"signal 11 is often SIGSEGV (torch/decord/cuda mismatch vs this Python); "
            f"set FAST_VQA_PYTHON to a venv with fastvqa deps, or eval.fast_vqa_device: cpu):{tail}"
        )
    return f"fast_vqa_subprocess_rc={returncode}:{tail}"


def _extract_score(text: str) -> Optional[float]:
    m = _SCORE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def run_fast_vqa_on_frames_dir(
    frames_dir: Path,
    *,
    fast_vqa_root: Path,
    model: str = "FasterVQA",
    device: str = "cuda",
    fps: float = 24.0,
    python_exe: str | None = None,
) -> Tuple[Optional[float], str]:
    """Return (score_0_1, status) where status explains failures."""
    from object_removal.utils.video import frames_to_mp4

    root = fast_vqa_root.resolve()
    vqa_py = root / "vqa.py"
    if not vqa_py.is_file():
        return None, f"fast_vqa_missing_vqa_py:{root}"

    files = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        return None, "fast_vqa_no_frames"

    td = Path(tempfile.mkdtemp(prefix="fast_vqa_inpaint_"))
    mp4 = td / "inpaint_preview.mp4"
    try:
        frames_to_mp4(frames_dir, mp4, fps=float(fps))
    except Exception as exc:
        return None, f"fast_vqa_mux_failed:{exc}"
    try:
        exe = resolve_fast_vqa_python(python_exe)

        def _run(dev: str) -> subprocess.CompletedProcess[str]:
            cmd = [
                exe,
                str(vqa_py),
                "-m",
                str(model),
                "-v",
                str(mp4),
                "-d",
                dev,
            ]
            return subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )

        dev = (device or "cuda").strip() or "cuda"
        proc = _run(dev)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            tail = out.strip()[-800:]
            # CUDA stack (torch/decord) sometimes segfaults in the eval driver env.
            if dev.lower() == "cuda" and proc.returncode is not None and proc.returncode < 0:
                print(
                    f"[eval] fast_vqa: subprocess died on cuda (rc={proc.returncode}); "
                    "retrying once with device=cpu",
                    flush=True,
                )
                proc2 = _run("cpu")
                out2 = (proc2.stdout or "") + "\n" + (proc2.stderr or "")
                if proc2.returncode == 0:
                    score2 = _extract_score(out2)
                    if score2 is not None:
                        return score2, "ok_cpu_fallback_after_cuda_crash"
                    tail2 = out2.strip()[-800:]
                    return None, f"fast_vqa_parse_failed_after_cpu_retry:{tail2}"
                tail_both = (out + "\n--- cpu retry ---\n" + out2).strip()[-1200:]
                return None, _format_subprocess_fail(proc.returncode, tail_both)
            return None, _format_subprocess_fail(proc.returncode, tail)
        score = _extract_score(out)
        if score is None:
            tail = out.strip()[-800:]
            return None, f"fast_vqa_parse_failed:{tail}"
        return score, "ok"
    finally:
        shutil.rmtree(td, ignore_errors=True)
