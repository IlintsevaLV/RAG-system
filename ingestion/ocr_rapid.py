"""RapidOCR wrapper (PP-OCRv5 Cyrillic) for pages routed to OCR.

Models are loaded from OCR_MODEL_DIR (default data/models/ocr) to avoid
ModelScope downloads on locked-down work PCs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from core.text_norm import normalize_ocr_text
from ingestion.models import OCRLine, OCRPageResult
from ingestion.ocr_reading_order import rebuild_ocr_text
from ingestion.preprocess import horizontal_bands

# Expected offline files (PP-OCRv5 mobile + Cyrillic rec)
REQUIRED_MODELS = (
    "ch_PP-OCRv5_det_mobile.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "cyrillic_PP-OCRv5_rec_mobile.onnx",
)


@dataclass
class RapidOCRConfig:
    use_gpu: bool = False
    max_side_len: int = 4000
    band_trigger_px: int = 2800
    band_height: int = 1600
    band_overlap: float = 0.12
    lang: str = "cyrillic"
    model_dir: str | None = None


def default_model_dir() -> Path:
    try:
        from core.config import get_settings

        return Path(get_settings().ocr_model_dir)
    except Exception:
        return Path("data/models/ocr")


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    return Path(model_dir) if model_dir else default_model_dir()


def check_local_models(model_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_MODELS if not (model_dir / name).is_file()]
    return missing


def _try_build_engine(cfg: RapidOCRConfig) -> tuple[Any, str]:
    """Build RapidOCR from local ONNX files; do not rely on ModelScope."""
    model_dir = resolve_model_dir(cfg.model_dir)
    missing = check_local_models(model_dir)

    last_err: Exception | None = None
    try:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        params: dict[str, Any] = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CYRILLIC,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Global.max_side_len": cfg.max_side_len,
            "Global.model_root_dir": str(model_dir.resolve()),
        }
        if not missing:
            params.update(
                {
                    "Det.model_path": str((model_dir / REQUIRED_MODELS[0]).resolve()),
                    "Cls.model_path": str((model_dir / REQUIRED_MODELS[1]).resolve()),
                    "Rec.model_path": str((model_dir / REQUIRED_MODELS[2]).resolve()),
                }
            )
        if cfg.use_gpu:
            params["EngineConfig.onnxruntime.use_cuda"] = True

        engine = RapidOCR(params=params)
        return engine, "rapidocr"
    except Exception as exc:  # noqa: BLE001
        last_err = exc

    # Legacy package (optional)
    try:
        from rapidocr_onnxruntime import RapidOCR as LegacyRapidOCR

        kwargs: dict[str, Any] = {}
        if not missing:
            kwargs = {
                "det_model_path": str(model_dir / REQUIRED_MODELS[0]),
                "rec_model_path": str(model_dir / REQUIRED_MODELS[2]),
            }
        return LegacyRapidOCR(**kwargs), "rapidocr_onnxruntime"
    except Exception as exc:  # noqa: BLE001
        last_err = last_err or exc

    hint = (
        f"RapidOCR models missing or download blocked (ModelScope).\n"
        f"  model_dir: {model_dir.resolve()}\n"
        f"  missing: {missing or '(files present but engine failed)'}\n"
        f"  Fix on a PC with internet, then copy folder to work PC:\n"
        f"    python -m scripts.prepare_ocr_models\n"
        f"  Or set OCR_MODEL_DIR to a folder with:\n"
        f"    {', '.join(REQUIRED_MODELS)}\n"
        f"  Last error: {last_err}"
    )
    raise RuntimeError(hint) from last_err


@lru_cache(maxsize=4)
def get_ocr_engine(
    use_gpu: bool = False,
    max_side_len: int = 4000,
    model_dir: str = "",
) -> tuple[Any, str]:
    return _try_build_engine(
        RapidOCRConfig(
            use_gpu=use_gpu,
            max_side_len=max_side_len,
            model_dir=model_dir or None,
        )
    )


def _quad_to_xyxy(box: Any) -> tuple[float, float, float, float]:
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _parse_rapidocr_output(result: Any) -> list[tuple[Any, str, float]]:
    """Normalize various RapidOCR return shapes to (box, text, score)."""
    if result is None:
        return []

    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and txts is not None:
        out = []
        for i, text in enumerate(txts):
            box = boxes[i] if i < len(boxes) else None
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            if box is None or text is None:
                continue
            out.append((box, str(text), score))
        return out

    rows = result[0] if isinstance(result, tuple) else result
    if not rows:
        return []
    out = []
    for row in rows:
        if row is None:
            continue
        if len(row) >= 3:
            box, text, score = row[0], row[1], row[2]
        elif len(row) == 2:
            box, text = row[0], row[1]
            score = 0.0
        else:
            continue
        out.append((box, str(text), float(score) if score is not None else 0.0))
    return out


def _run_engine(engine: Any, image_bgr: np.ndarray) -> list[tuple[Any, str, float]]:
    result = engine(image_bgr)
    return _parse_rapidocr_output(result)


def px_bbox_to_pdf_points(
    bbox_px: tuple[float, float, float, float],
    *,
    dpi: int,
) -> tuple[float, float, float, float]:
    scale = 72.0 / float(dpi)
    x0, y0, x1, y1 = bbox_px
    return x0 * scale, y0 * scale, x1 * scale, y1 * scale


def recognize_image(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    page: int,
    doc_id: str,
    cfg: RapidOCRConfig | None = None,
    y_offset_px: float = 0.0,
) -> list[OCRLine]:
    cfg = cfg or RapidOCRConfig()
    model_dir = str(resolve_model_dir(cfg.model_dir).resolve())
    engine, engine_name = get_ocr_engine(
        use_gpu=cfg.use_gpu,
        max_side_len=cfg.max_side_len,
        model_dir=model_dir,
    )
    rows = _run_engine(engine, image_bgr)
    lines: list[OCRLine] = []
    for box, text, score in rows:
        text_n = normalize_ocr_text(text)
        if not text_n:
            continue
        x0, y0, x1, y1 = _quad_to_xyxy(box)
        y0 += y_offset_px
        y1 += y_offset_px
        bbox_px = (x0, y0, x1, y1)
        bbox_pt = px_bbox_to_pdf_points(bbox_px, dpi=dpi)
        lines.append(
            OCRLine(
                text=text_n,
                text_raw=text,
                bbox_px=bbox_px,
                bbox_pt=bbox_pt,
                confidence=score,
                engine=engine_name,
            )
        )
    lines.sort(key=lambda ln: (ln.bbox_px[1], ln.bbox_px[0]))
    return lines


def recognize_page_image(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    page: int,
    doc_id: str,
    cfg: RapidOCRConfig | None = None,
    bands: list[tuple[int, int, np.ndarray]] | None = None,
) -> OCRPageResult:
    cfg = cfg or RapidOCRConfig()
    h, w = image_bgr.shape[:2]
    use_bands = bands
    if use_bands is None and max(h, w) >= cfg.band_trigger_px:
        use_bands = horizontal_bands(
            image_bgr, band_height=cfg.band_height, overlap=cfg.band_overlap
        )

    if use_bands:
        lines: list[OCRLine] = []
        for y0, _y1, band_img in use_bands:
            lines.extend(
                recognize_image(
                    band_img,
                    dpi=dpi,
                    page=page,
                    doc_id=doc_id,
                    cfg=cfg,
                    y_offset_px=float(y0),
                )
            )
        lines = _dedupe_overlap_lines(lines)
        lines.sort(key=lambda ln: (ln.bbox_px[1], ln.bbox_px[0]))
    else:
        lines = recognize_image(
            image_bgr, dpi=dpi, page=page, doc_id=doc_id, cfg=cfg
        )

    confs = [ln.confidence for ln in lines]
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    text, ordered, ro = rebuild_ocr_text(lines, float(w))
    text = normalize_ocr_text(text)
    return OCRPageResult(
        doc_id=doc_id,
        page=page,
        lines=ordered or lines,
        text=text,
        mean_confidence=mean_conf,
        line_count=len(ordered or lines),
        used_bands=bool(use_bands),
        image_size=(w, h),
        dpi=dpi,
        n_columns=ro.n_columns,
        formula_suspect=ro.formula_suspect,
        short_fragment_ratio=ro.short_fragment_ratio,
        reading_order_notes=list(ro.notes or []),
    )


def _dedupe_overlap_lines(lines: list[OCRLine], iou_thresh: float = 0.5) -> list[OCRLine]:
    kept: list[OCRLine] = []
    for ln in sorted(lines, key=lambda x: -x.confidence):
        drop = False
        for k in kept:
            if _bbox_iou(ln.bbox_px, k.bbox_px) >= iou_thresh and (
                ln.text == k.text or ln.text in k.text or k.text in ln.text
            ):
                drop = True
                break
        if not drop:
            kept.append(ln)
    return kept


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
