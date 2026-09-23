"""Formula pipeline: detect crop → UniMERNet → LaTeX normalize/validate → IR."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import os

from core.ir import BBox, BlockType, Provenance, QualitySignals, RegionBlock
from core.logger import get_logger
from ingestion.region_detect import DetectedRegion, crop_region_bgr

log = get_logger("formula")

_LATEX_FORBIDDEN = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\u024f\\\{\}\^_]")
_LATEX_MATH_SIGNAL = re.compile(
    r"(=|\\frac|\\sqrt|\\sum|\\int|\\prod|\\lim|[_^]|"
    r"[<>≤≥≈≠±∑∫√]|"
    r"[A-Za-z0-9]\s*[+*/]\s*[A-Za-z0-9]|"
    r"[A-Za-z0-9]\s+-\s+[A-Za-z0-9])"
)
_LATEX_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")


def normalize_latex(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("$$", "").replace("$", "")
    s = s.strip()
    # common wrappers
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def latex_to_text_fallback(latex: str) -> str:
    """Rough readable string for embeddings (not perfect math)."""
    t = latex
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", t)
    t = re.sub(r"\\text\{([^{}]+)\}", r"\1", t)
    t = re.sub(r"\\left|\\right", "", t)
    t = re.sub(r"[{}]", "", t)
    t = t.replace("^", "**").replace("_", "")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def validate_latex(latex: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if not latex or len(latex) < 2:
        return False, ["empty"]
    # balanced braces
    bal = 0
    for ch in latex:
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
            if bal < 0:
                return False, ["unbalanced_braces"]
    if bal != 0:
        return False, ["unbalanced_braces"]
    # UniMERNet returns syntactically valid LaTeX for many non-formula crops.
    # Reject prose/reference-like output before it enters the IR as a formula.
    words = _LATEX_WORD.findall(latex)
    has_signal = bool(_LATEX_MATH_SIGNAL.search(latex))
    if not has_signal and len(words) >= 2:
        return False, ["no_math_signal", "prose_like"]
    if len(words) >= 5 and not any(
        token in latex for token in ("\\frac", "\\sqrt", "\\sum", "\\int", "^", "_")
    ):
        return False, ["prose_like"]
    # try pylatexenc if available
    try:
        from pylatexenc.latexwalker import LatexWalker

        LatexWalker(latex, tolerant_parsing=True).get_latex_nodes()
        notes.append("pylatexenc_ok")
        return True, notes
    except ImportError:
        notes.append("pylatexenc_not_installed")
        # heuristic OK if has math-ish content
        if any(tok in latex for tok in ("=", "_", "^", "\\frac", "\\sum", "\\int", "+", "-")):
            return True, notes + ["heuristic_ok"]
        return len(latex) >= 3, notes + ["heuristic_weak"]
    except Exception as exc:  # noqa: BLE001
        return False, [f"pylatexenc_fail:{exc}"]


class UniMERNetRecognizer:
    """UniMERNet wrapper via persistent subprocess in .venv-unimernet.

    Main venv does not have unimernet installed (it conflicts with Docling
    via transformers version). So we spawn a long-running worker process
    that loads the model once and answers requests over stdin/stdout.

    The worker path is auto-detected:
      - env UNIMERNET_WORKER_PYTHON, or
      - .venv-unimernet/Scripts/python.exe (Windows)
      - .venv-unimernet/bin/python (POSIX)
    """

    _WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "unimernet_worker.py"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        config_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.config_path = config_path or ""
        self._proc = None
        self._available = False
        self._init_error: str | None = None
        self._try_load()

    def _find_worker_python(self) -> Path | None:
        env = os.environ.get("UNIMERNET_WORKER_PYTHON")
        if env:
            p = Path(env)
            return p if p.is_file() else None
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / ".venv-unimernet" / "Scripts" / "python.exe",
            root / ".venv-unimernet" / "bin" / "python",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _try_load(self) -> None:
        import json
        import subprocess

        try:
            if not self._WORKER_SCRIPT.is_file():
                raise FileNotFoundError(...)
            py = self._find_worker_python()
            print(f"[UNIMERNET] py = {py}", flush=True)

            if py is None:
                raise FileNotFoundError(...)
            cfg = self.config_path or str(...)
            print(f"[UNIMERNET] cfg = {cfg}", flush=True)
            if not Path(cfg).is_file():
                raise FileNotFoundError(...)

            self._proc = subprocess.Popen(
                [str(py), str(self._WORKER_SCRIPT), str(cfg)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            print("[UNIMERNET] proc started, waiting for ready", flush=True)
            # Worker's model may print init messages to stdout before JSON.
            # Read lines until we find a valid JSON.
            msg = None
            for _ in range(200):
                line = self._proc.stdout.readline()
                print(f"[UNIMERNET] stdout line = {line!r}", flush=True)
                if not line:
                    err = self._proc.stderr.read() if self._proc.stderr else ""
                    raise RuntimeError(f"Worker died on startup: {err[:500]}")
                try:
                    msg = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if msg is None:
                raise RuntimeError("No ready message from worker after 200 lines")
            if not msg.get("ready"):
                raise RuntimeError(f"Worker not ready: {msg.get('error', 'unknown')}")

            self._available = True
            self._init_error = None
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._init_error = str(exc)
            import traceback
            print(f"[UNIMERNET] _try_load failed: {exc}", flush=True)
            print(f"[UNIMERNET] traceback:\n{traceback.format_exc()}", flush=True)
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
                self._proc = None
    @property
    def available(self) -> bool:
        return self._available and self._proc is not None

    def recognize(self, image_bgr: np.ndarray) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError(self._init_error or "UniMERNet unavailable")

        import base64
        import json

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image

        buf = __import__("io").BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        req = json.dumps({"image_b64": b64}, ensure_ascii=False)
        try:
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()

            resp = None
            for _ in range(200):
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("Worker died mid-request")
                try:
                    resp = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if resp is None:
                raise RuntimeError("No JSON response from worker")
        except Exception as exc:  # noqa: BLE001
            self._available = False
            raise RuntimeError(f"Worker communication failed: {exc}") from exc

        if "error" in resp:
            raise RuntimeError(f"Worker error: {resp['error']}")
        return str(resp.get("latex") or ""), float(resp.get("confidence") or 0.85)

    def __del__(self) -> None:
        if getattr(self, "_proc", None) is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
def recognize_formula_vlm(image_bgr: np.ndarray, vlm: Any) -> tuple[str, float]:
    prompt = (
        "Extract the mathematical formula from this image as LaTeX only. "
        "No markdown fences, no explanation. If unreadable write UNREADABLE."
    )
    raw = vlm.read_ndarray(image_bgr, prompt).strip()
    if raw.upper().startswith("UNREADABLE"):
        return "", 0.0
    return normalize_latex(raw), 0.55


def process_formula_region(
    image_bgr: np.ndarray,
    region: DetectedRegion,
    *,
    doc_id: str,
    page: int,
    region_id: str,
    unimer: UniMERNetRecognizer | None = None,
    vlm: Any | None = None,
    highres_scale: float = 1.5,
) -> RegionBlock:
    crop = crop_region_bgr(image_bgr, region, pad=12)
    if crop.size == 0 or min(crop.shape[:2]) < 10:
        return RegionBlock(
            doc_id=doc_id,
            page=page,
            region_id=region_id,
            type=BlockType.FORMULA,
            bbox=region.bbox_pt,
            quality=QualitySignals(structural_ok=False, notes=["empty_crop"]),
            content={},
            provenance=Provenance(method="none"),
        )

    # high-res upsample for tiny formulas
    if min(crop.shape[:2]) < 128:
        crop = cv2.resize(
            crop,
            None,
            fx=highres_scale,
            fy=highres_scale,
            interpolation=cv2.INTER_CUBIC,
        )

    latex_raw = ""
    conf = 0.0
    method = "none"
    notes: list[str] = list(region.notes)

    if unimer and unimer.available:
        try:
            latex_raw, conf = unimer.recognize(crop)
            method = "unimernet"
        except Exception as exc:  # noqa: BLE001
            notes.append(f"unimer_fail:{exc}")
            log.warning("unimer_fail", error=str(exc))

    if (not latex_raw or conf < 0.3) and vlm is not None:
        try:
            latex_raw, conf = recognize_formula_vlm(crop, vlm)
            method = "vlm_recovery" if method != "none" else "vlm"
            notes.append("vlm_used")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vlm_fail:{exc}")

    latex_norm = normalize_latex(latex_raw)
    ok, vnotes = validate_latex(latex_norm)
    notes.extend(vnotes)

    # one retry with stronger upscale if invalid
    if not ok and unimer and unimer.available:
        try:
            crop2 = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            latex_raw2, conf2 = unimer.recognize(crop2)
            latex_norm2 = normalize_latex(latex_raw2)
            ok2, vnotes2 = validate_latex(latex_norm2)
            notes.append("retry_upscale")
            notes.extend(vnotes2)
            if ok2:
                latex_raw, latex_norm, conf, ok = latex_raw2, latex_norm2, conf2, True
                method = "unimernet_retry"
        except Exception as exc:  # noqa: BLE001
            notes.append(f"retry_fail:{exc}")

    text_fb = latex_to_text_fallback(latex_norm) if latex_norm else ""
    return RegionBlock(
        doc_id=doc_id,
        page=page,
        region_id=region_id,
        type=BlockType.FORMULA,
        bbox=region.bbox_pt,
        source="raster",
        quality=QualitySignals(
            confidence=conf,
            structural_ok=ok,
            suspicion=0.0 if ok else 0.7,
            notes=notes,
        ),
        content={
            "latex_raw": latex_raw,
            "latex": latex_norm,
            "text_fallback": text_fb,
            "status": "ok" if ok and latex_norm else "suspicious",
        },
        provenance=Provenance(model=method, method=method, confidence=conf),
    )
