"""Formula pipeline: detect crop → UniMERNet → LaTeX normalize/validate → IR."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
    """UniMERNet wrapper for unimernet 0.2.3.

    Uses the low-level API: Config → tasks.setup_task → task.build_model
    → load_processor. Avoids demo.ImageProcessor because it opens a GUI
    window (cv2.imshow) and blocks on cv2.waitKey(0).
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        config_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.config_path = config_path or ""
        self._model = None
        self._vis_processor = None
        self._device = None
        self._available = False
        self._init_error: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import argparse
            import torch
            from unimernet.common.config import Config
            import unimernet.tasks as tasks
            from unimernet.processors import load_processor
            from pathlib import Path

            cfg_path = self.config_path or str(
                Path(self.model_name or "").parent / "configs" / "demo.yaml"
            )
            if not Path(cfg_path).is_file():
                raise FileNotFoundError(
                    f"UniMERNet config not found: {cfg_path}. "
                    "Set UNIMERNET_CONFIG_PATH."
                )

            args = argparse.Namespace(cfg_path=cfg_path, options=None)
            cfg = Config(args)

            task = tasks.setup_task(cfg)
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = task.build_model(cfg).to(self._device)
            self._model.eval()

            self._vis_processor = load_processor(
                "formula_image_eval",
                cfg.config.datasets.formula_rec_eval.vis_processor.eval,
            )

            self._available = True
            self._init_error = None
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._available

    def recognize(self, image_bgr: np.ndarray) -> tuple[str, float]:
        if not self._available or self._model is None:
            raise RuntimeError(self._init_error or "UniMERNet unavailable")

        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # vis_processor returns a tensor; add batch dim, move to device
        tensor = self._vis_processor(pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model.generate({"image": tensor})

        # output is a dict with "pred_str": list[str]
        if isinstance(output, dict):
            preds = output.get("pred_str") or []
            latex = preds[0] if preds else ""
        else:
            latex = str(output)

        return str(latex or ""), 0.85
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
    if crop.size == 0:
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
