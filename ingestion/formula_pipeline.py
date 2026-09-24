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
from ingestion.models import ExtractStatus
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


_DEGENERATE_SPACING = re.compile(r"(\\[,;:! ]|\\q?quad(?![a-zA-Z])|~)(?:\s*\1){3,}")
_STRUCTURAL_BRACE = re.compile(r"(?<!\\)[{}]")
_ENV_RE = re.compile(r"\\(begin|end)\{([a-zA-Z*]+)\}")


def repair_latex(latex: str) -> tuple[str, list[str]]:
    """Fix UniMERNet output cut by max_seq_len: collapse degenerate `\\! \\! \\! …`
    tails, drop unmatched `}`, close open `{` and `\\begin{...}` environments.

    Returns the fixed LaTeX and notes (`spacing_collapsed`, `braces_repaired`)."""
    notes: list[str] = []
    s = _DEGENERATE_SPACING.sub(r"\1", latex)
    if s != latex:
        notes.append("spacing_collapsed")
        s = re.sub(r"\s+", " ", s).strip()
    out: list[str] = []
    depth = 0
    pos = 0
    for m in _STRUCTURAL_BRACE.finditer(s):
        out.append(s[pos : m.start()])
        if m.group(0) == "{":
            depth += 1
            out.append("{")
        elif depth > 0:
            depth -= 1
            out.append("}")
        pos = m.end()
    out.append(s[pos:])
    fixed = "".join(out).rstrip()
    if depth:
        fixed += " " + " ".join("}" * depth)
    open_envs: list[str] = []
    for kind, name in _ENV_RE.findall(fixed):
        if kind == "begin":
            open_envs.append(name)
        elif open_envs and open_envs[-1] == name:
            open_envs.pop()
    for name in reversed(open_envs):
        fixed += f" \\end{{{name}}}"
    fixed = re.sub(r"\s+", " ", fixed).strip()
    if fixed != s.strip():
        notes.append("braces_repaired")
    return fixed, notes


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
    # balanced braces (escaped \{ \} are delimiters, not groups)
    bal = 0
    for m in _STRUCTURAL_BRACE.finditer(latex):
        if m.group(0) == "{":
            bal += 1
        else:
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


def normalize_and_validate(latex_raw: str) -> tuple[str, bool, list[str]]:
    """normalize_latex -> repair_latex -> validate_latex."""
    latex_norm = normalize_latex(latex_raw)
    notes: list[str] = []
    if latex_norm:
        latex_norm, notes = repair_latex(latex_norm)
    ok, vnotes = validate_latex(latex_norm)
    return latex_norm, ok, notes + vnotes


_SEM_MATH_MARKERS = re.compile(
    r"[=+\-*/^_<>≤≥≠≈]|\\(?:frac|sqrt|int|sum|prod|lim|partial|infty|alpha|beta|gamma|"
    r"theta|lambda|mu|pi|sigma|omega|kappa|rho|hbar|phi|Delta)(?![a-zA-Z])"
)
_SEM_RUSSIAN_WORDS = re.compile(r"[А-Яа-яЁё]{3,}\s+[А-Яа-яЁё]{3,}")
_SEM_PURE_NUMBERS = re.compile(r"[0-9.,=×xX;:%]+")
_SEM_FORMULA_NUMBER = re.compile(r"\(?[IVX]+\.\d+\)?|\(\d+\.\d+\)")
_SEM_LETTER_RUN = re.compile(r"[^\W\d_]{4,}")
_SEM_MATH_WORDS = frozenset(
    {"const", "grad", "arcsin", "arccos", "arctg", "arctan", "sinh", "cosh", "tanh", "sign"}
)
# OCR text of the source spans: Cyrillic words or Greek-lookalike garbage of them.
_SEM_SOURCE_PROSE = re.compile(r"[А-Яа-яЁё]{4,}|[^\W\d_]{5,}")
_SEM_SPACING_CMD = re.compile(r"\\(?:[,;:!]|q?quad(?![a-zA-Z]))|~")
_SEM_RELATION = re.compile(r"=|\\approx|\\le(?:q)?(?![a-zA-Z])|\\ge(?:q)?(?![a-zA-Z])|[<>≤≥≈]")
_SEM_TRAILING_RELATION = re.compile(r"(?:=|\\approx|<|>)\s*[-+.,:;]?\s*$")
_SEM_VALUE_SEPARATORS = re.compile(r"\\(?:qquad|quad|mid|vert)(?![a-zA-Z])|\\\\|[|;&]")
_SEM_VALUE_STRUCTURE = re.compile(r"\\(?:frac|sqrt|int|sum|prod|lim|partial)(?![a-zA-Z])")
_SEM_FRAC_WITH_RELATION = re.compile(r"\\frac\s*\{([^{}]*=[^{}]*)\}\s*\{[^{}]*\}")
_SEM_VALUE_ASSIGNMENT = re.compile(
    r"[A-Za-zА-Яа-яЁёΑ-Ωα-ω]'?(?:_[A-Za-zА-Яа-яЁёΑ-Ωα-ω0-9]{0,3})?"
    r"=[-−]?\d+(?:[.,]\d+)*(?:[A-Za-zА-Яа-яЁё]{1,3}\.?)?"
)
_SEM_VALUE_LEFTOVER = re.compile(r"[\s()\[\].,:;^'`|]*")
_SEM_TINY_W_PX = 20
_SEM_TINY_H_PX = 8
_SEM_TABLE_SHARE = 0.7
_SEM_FIGURE_SHARE = 0.3
_SEM_FIGURE_GAP_PT = 15.0
_SEM_TEXT_CMD = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathtt|mathsf|mathcal|mathscr|text|textbf|textit|"
    r"textrm|operatorname)(?![a-zA-Z])"
)
# Operators that make a text-looking region a formula (issue 011, problem B).
_SEM_HEADER_MATH = re.compile(r"[=+^_]|\\(?:frac|sum|int|sqrt|prod)(?![a-zA-Z])")
# Text regions: kept in JSON as captions/headers, not indexed as formulas.
_CAPTION_REASONS = frozenset({"caption_or_header", "figure_caption"})


def _latex_plain(latex: str) -> str:
    """Latex without environments, commands, spacing and braces (for value checks)."""
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[^{}]*\})?", " ", latex)
    s = _SEM_SPACING_CMD.sub(" ", s)
    s = re.sub(r"\\\\|\\ ", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}&]", " ", s)
    return re.sub(r"\s+", "", s)


def _latex_letter_runs(latex: str) -> list[str]:
    """Letter runs as they read in the source: UniMERNet spells text as `a b c` or `a \\ b`."""
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[^{}]*\})?", "|", latex)
    s = s.replace("\\\\", "|")
    s = s.replace("\\ ", "")
    s = re.sub(r"\\[a-zA-Z]+", "|", s)
    s = re.sub(r"\\.", "|", s)
    s = s.replace("~", "|")
    s = re.sub(r"\s+", "", s)
    return [m for m in _SEM_LETTER_RUN.findall(s) if m.lower() not in _SEM_MATH_WORDS]


def _latex_group(s: str, i: int) -> tuple[str, int]:
    """Argument starting at s[i] (a {...} group or a single token) and the index after it."""
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s):
        return "", i
    if s[i] != "{":
        m = re.match(r"\\[a-zA-Z]+|.", s[i:])
        tok = m.group(0) if m else ""
        return tok, i + len(tok)
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    return s[i + 1 :], len(s)


def _has_degenerate_fraction(latex: str) -> bool:
    for m in re.finditer(r"\\frac(?![a-zA-Z])", latex):
        num, i = _latex_group(latex, m.end())
        den, _ = _latex_group(latex, i)
        for part in (num, den):
            if not re.search(r"[A-Za-z0-9]", part):
                return True
    return False


def _is_truncated(latex: str) -> bool:
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[^{}]*\})?", " ", latex)
    s = _SEM_SPACING_CMD.sub(" ", s)
    s = re.sub(r"[{}]", " ", s)
    return bool(_SEM_TRAILING_RELATION.search(s)) or _has_degenerate_fraction(latex)


def _is_numeric_value(latex: str) -> bool:
    """`m_к = 0,00052 | m_к = 0,00088`, `(N = 200000, C = 1225 кг)`: values, not formulas."""
    s = _SEM_FRAC_WITH_RELATION.sub(r" \1 ", latex)
    if _SEM_VALUE_STRUCTURE.search(s):
        return False
    s = _SEM_VALUE_SEPARATORS.sub(";", s)
    plain = _latex_plain(s)
    found = False
    for piece in plain.split(";"):
        rest, n = _SEM_VALUE_ASSIGNMENT.subn("", piece)
        found = found or n > 0
        if not _SEM_VALUE_LEFTOVER.fullmatch(rest):
            return False
    return found


def _source_text(notes: list[str]) -> str:
    return " | ".join(n[len("text=") :] for n in notes if n.startswith("text="))


def _share_inside(inner: BBox, outer: BBox) -> float:
    w = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    h = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    area = max(1e-6, (inner.x2 - inner.x1) * (inner.y2 - inner.y1))
    return w * h / area


def _is_near_figure(bbox: BBox, figure: BBox) -> bool:
    """Overlaps the figure or sits right below it."""
    if _share_inside(bbox, figure) >= _SEM_FIGURE_SHARE:
        return True
    x_overlap = max(0.0, min(bbox.x2, figure.x2) - max(bbox.x1, figure.x1))
    if x_overlap < _SEM_FIGURE_SHARE * max(1e-6, bbox.x2 - bbox.x1):
        return False
    return figure.y2 - 2.0 <= bbox.y1 <= figure.y2 + _SEM_FIGURE_GAP_PT


def _strip_text_commands(latex: str) -> str:
    """Latex without `\\mathrm{...}`-like text groups, environments and spacing."""
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[^{}]*\})?", " ", latex)
    out: list[str] = []
    i = 0
    for m in _SEM_TEXT_CMD.finditer(s):
        if m.start() < i:
            continue
        out.append(s[i : m.start()])
        _arg, i = _latex_group(s, m.end())
    out.append(s[i:])
    return "".join(out)


def _has_text_content(latex: str, source_text: str) -> bool:
    return bool(
        _SEM_RUSSIAN_WORDS.search(latex)
        or _latex_letter_runs(latex)
        or any(
            m.lower() not in _SEM_MATH_WORDS
            for m in _SEM_SOURCE_PROSE.findall(source_text or "")
        )
    )


def is_real_formula_semantic(
    latex: str,
    *,
    bbox_pt: BBox | None = None,
    page_figures: list[BBox] | None = None,
    page_tables: list[BBox] | None = None,
    source_text: str = "",
    raw_crop_wh: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """Semantic check after `validate_latex`: UniMERNet returns valid LaTeX for prose,
    captions and table cells, so syntax alone does not make a formula.

    Returns (is_formula, reason); reason is empty when the formula passes.
    """
    if raw_crop_wh is not None:
        w, h = raw_crop_wh
        if w < _SEM_TINY_W_PX or h < _SEM_TINY_H_PX:
            return False, "tiny_crop"

    plain = _latex_plain(latex)
    has_markers = bool(_SEM_MATH_MARKERS.search(latex))
    if _SEM_FORMULA_NUMBER.fullmatch(plain):
        return False, "formula_number"

    if bbox_pt is not None:
        if any(_is_near_figure(bbox_pt, f) for f in page_figures or []) and _has_text_content(
            latex, source_text
        ):
            return False, "figure_caption"
        in_table = any(
            _share_inside(bbox_pt, t) >= _SEM_TABLE_SHARE for t in page_tables or []
        )
        if in_table and (
            not has_markers or _SEM_PURE_NUMBERS.fullmatch(plain) or _is_numeric_value(latex)
        ):
            return False, "table_value"

    if plain and _SEM_PURE_NUMBERS.fullmatch(plain):
        return False, "pure_numbers"
    if _is_numeric_value(latex):
        return False, "numeric_value"
    if _is_truncated(latex):
        return False, "truncated"
    text_reason = ""
    if _SEM_RUSSIAN_WORDS.search(latex):
        text_reason = "russian_words"
    elif _latex_letter_runs(latex):
        text_reason = "long_letter_run"
    elif any(
        m.lower() not in _SEM_MATH_WORDS for m in _SEM_SOURCE_PROSE.findall(source_text or "")
    ):
        text_reason = "prose_source_text"
    if text_reason:
        # Headings and captions: all letters are inside \mathrm{...}, no operators outside.
        if not _SEM_HEADER_MATH.search(_strip_text_commands(latex)):
            return False, "caption_or_header"
        return False, text_reason
    if not has_markers:
        return False, "no_math_markers"
    return True, ""


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
    page_figures: list[BBox] | None = None,
    page_tables: list[BBox] | None = None,
    dpi: int = 200,
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

    preview_h, preview_w = crop.shape[:2]
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

    latex_norm, ok, vnotes = normalize_and_validate(latex_raw)
    notes.extend(vnotes)

    # one retry with stronger upscale if invalid
    if not ok and unimer and unimer.available:
        try:
            crop2 = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            latex_raw2, conf2 = unimer.recognize(crop2)
            latex_norm2, ok2, vnotes2 = normalize_and_validate(latex_raw2)
            notes.append("retry_upscale")
            notes.extend(vnotes2)
            if ok2:
                latex_raw, latex_norm, conf, ok = latex_raw2, latex_norm2, conf2, True
                method = "unimernet_retry"
        except Exception as exc:  # noqa: BLE001
            notes.append(f"retry_fail:{exc}")

    status = "ok" if ok and latex_norm else "suspicious"
    suspicion = 0.0 if ok else 0.7
    notes.append(f"preview_wh={preview_w}x{preview_h}")
    if status == "ok":
        scale = dpi / 72.0
        bbox = region.bbox_pt
        sem_ok, sem_reason = is_real_formula_semantic(
            latex_norm,
            bbox_pt=bbox,
            page_figures=page_figures,
            page_tables=page_tables,
            source_text=_source_text(region.notes),
            raw_crop_wh=(
                int(round((bbox.x2 - bbox.x1) * scale)),
                int(round((bbox.y2 - bbox.y1) * scale)),
            ),
        )
        if not sem_ok:
            status = (
                ExtractStatus.CAPTION_OR_HEADER.value
                if sem_reason in _CAPTION_REASONS
                else ExtractStatus.SUSPICIOUS_SEMANTIC.value
            )
            suspicion = 0.6
            notes.append(f"semantic_reason={sem_reason}")

    text_fb = latex_to_text_fallback(latex_norm) if latex_norm else ""
    content: dict[str, Any] = {
        "latex_raw": latex_raw,
        "latex": latex_norm,
        "text_fallback": text_fb,
        "status": status,
    }
    if status == ExtractStatus.CAPTION_OR_HEADER.value:
        # UniMERNet LaTeX of text is garbage; the OCR text of the spans is the content.
        content["text"] = _source_text(region.notes)
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
            suspicion=suspicion,
            notes=notes,
        ),
        content=content,
        provenance=Provenance(model=method, method=method, confidence=conf),
    )
