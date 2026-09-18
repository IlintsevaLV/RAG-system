"""Table pipeline: Docling → PP-Structure → img2table → VLM recovery → markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.ir import BlockType, Provenance, QualitySignals, RegionBlock
from core.logger import get_logger
from ingestion.region_detect import DetectedRegion, crop_region_bgr

log = get_logger("table")


@dataclass
class TableParseResult:
    markdown: str
    html: str = ""
    n_rows: int = 0
    n_cols: int = 0
    method: str = ""
    confidence: float = 0.0
    notes: list[str] | None = None


def _md_quality(md: str) -> tuple[float, int, int, list[str]]:
    notes: list[str] = []
    if not md or not md.strip():
        return 0.0, 0, 0, ["empty"]
    lines = [ln for ln in md.splitlines() if ln.strip()]
    rows = [ln for ln in lines if "|" in ln and not re.match(r"^\s*\|?\s*-+", ln)]
    if not rows:
        return 0.1, 0, 0, ["no_pipe_rows"]
    cols = [ln.count("|") - (1 if ln.strip().startswith("|") else 0) for ln in rows]
    # normalize col count
    n_cols = int(np.median(cols)) if cols else 0
    n_rows = len(rows)
    consistency = float(sum(1 for c in cols if abs(c - n_cols) <= 1) / max(1, len(cols)))
    # nonempty cells estimate
    cells = []
    for ln in rows:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        cells.extend(parts)
    nonempty = sum(1 for c in cells if c) / max(1, len(cells))
    score = 0.45 * consistency + 0.40 * nonempty + 0.15 * min(1.0, n_rows / 5.0)
    if n_cols < 2 or n_rows < 2:
        score *= 0.5
        notes.append("too_small")
    notes.append(f"rows={n_rows} cols={n_cols} consistency={consistency:.2f}")
    return float(score), n_rows, n_cols, notes


def parse_table_docling(image_bgr: np.ndarray, tmp_dir: Path) -> TableParseResult | None:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        log.info("docling_not_installed")
        return None
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        img_path = tmp_dir / "table_crop.png"
        cv2.imwrite(str(img_path), image_bgr)
        conv = DocumentConverter()
        result = conv.convert(str(img_path))
        doc = result.document
        # export markdown; prefer table-focused export if available
        md = ""
        if hasattr(doc, "export_to_markdown"):
            md = doc.export_to_markdown() or ""
        html = ""
        if hasattr(doc, "export_to_html"):
            html = doc.export_to_html() or ""
        score, nr, nc, notes = _md_quality(md)
        return TableParseResult(
            markdown=md,
            html=html,
            n_rows=nr,
            n_cols=nc,
            method="docling",
            confidence=score,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("docling_fail", error=str(exc))
        return TableParseResult(
            markdown="", method="docling", confidence=0.0, notes=[f"fail:{exc}"]
        )


def parse_table_ppstructure(image_bgr: np.ndarray) -> TableParseResult | None:
    """PaddleOCR PP-Structure table engine (optional)."""
    try:
        from paddleocr import PPStructure  # type: ignore
    except ImportError:
        try:
            from paddlex import create_pipeline  # noqa: F401
        except ImportError:
            log.info("ppstructure_not_installed")
            return None
        return TableParseResult(
            markdown="",
            method="ppstructure",
            confidence=0.0,
            notes=["paddlex_present_but_adapter_not_wired"],
        )

    try:
        engine = PPStructure(show_log=False, table=True, ocr=True, layout=False)
        result = engine(image_bgr)
        html = ""
        md = ""
        for block in result or []:
            if block.get("type") == "table":
                res = block.get("res") or {}
                html = res.get("html") or html
        if html:
            md = _html_table_to_markdown(html)
        score, nr, nc, notes = _md_quality(md)
        return TableParseResult(
            markdown=md,
            html=html,
            n_rows=nr,
            n_cols=nc,
            method="ppstructure",
            confidence=score,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ppstructure_fail", error=str(exc))
        return TableParseResult(
            markdown="", method="ppstructure", confidence=0.0, notes=[f"fail:{exc}"]
        )


def parse_table_img2table(image_bgr: np.ndarray) -> TableParseResult | None:
    try:
        from img2table.document import Image as Img2TableImage
        from img2table.ocr import TesseractOCR
    except ImportError:
        log.info("img2table_not_installed")
        return None
    try:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            return None
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(buf.tobytes())
            path = f.name
        doc = Img2TableImage(path)
        ocr = TesseractOCR(lang="rus+eng")
        tables = doc.extract_tables(ocr=ocr)
        if not tables:
            return TableParseResult(
                markdown="", method="img2table", confidence=0.0, notes=["no_tables"]
            )
        # take largest
        best = max(tables, key=lambda t: t.bbox.area if hasattr(t.bbox, "area") else 0)
        df = best.df
        md = df.to_markdown(index=False) if df is not None else ""
        score, nr, nc, notes = _md_quality(md.replace("---", "|---"))  # type: ignore[arg-type]
        # pandas markdown uses --- without pipes sometimes
        if not md:
            md = df.to_csv(index=False) if df is not None else ""
            score, nr, nc, notes = 0.4, len(df), len(df.columns), ["csv_fallback"]
        return TableParseResult(
            markdown=md,
            n_rows=nr or (len(df) if df is not None else 0),
            n_cols=nc or (len(df.columns) if df is not None else 0),
            method="img2table",
            confidence=max(score, 0.45),
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("img2table_fail", error=str(exc))
        return TableParseResult(
            markdown="", method="img2table", confidence=0.0, notes=[f"fail:{exc}"]
        )


def parse_table_vlm(image_bgr: np.ndarray, vlm: Any) -> TableParseResult:
    prompt = (
        "Extract the table from this image as GitHub-flavored Markdown only. "
        "Use | columns | and a header separator row. "
        "Do not invent cells. If unreadable write UNREADABLE."
    )
    raw = (vlm.read_ndarray(image_bgr, prompt) or "").strip()
    if raw.upper().startswith("UNREADABLE"):
        return TableParseResult(
            markdown="", method="vlm", confidence=0.0, notes=["unreadable"]
        )
    # strip fences
    raw = re.sub(r"^```(?:markdown)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    score, nr, nc, notes = _md_quality(raw)
    return TableParseResult(
        markdown=raw,
        n_rows=nr,
        n_cols=nc,
        method="vlm",
        confidence=min(score, 0.75),
        notes=notes + ["vlm_recovery"],
    )


def _html_table_to_markdown(html: str) -> str:
    try:
        from html.parser import HTMLParser

        rows: list[list[str]] = []
        cur: list[str] = []
        capture = False
        buf: list[str] = []

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                nonlocal capture, cur, buf
                if tag in ("td", "th"):
                    capture = True
                    buf = []
                if tag == "tr":
                    cur = []

            def handle_endtag(self, tag):
                nonlocal capture, cur, buf
                if tag in ("td", "th") and capture:
                    cur.append(re.sub(r"\s+", " ", "".join(buf)).strip())
                    capture = False
                if tag == "tr" and cur:
                    rows.append(cur)
                    cur = []

            def handle_data(self, data):
                if capture:
                    buf.append(data)

        P().feed(html)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        norm = [r + [""] * (width - len(r)) for r in rows]
        header = norm[0]
        sep = ["---"] * width
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for r in norm[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)
    except Exception:
        return ""


def process_table_region(
    image_bgr: np.ndarray,
    region: DetectedRegion,
    *,
    doc_id: str,
    page: int,
    region_id: str,
    cache_dir: Path,
    vlm: Any | None = None,
    enable_docling: bool = True,
    enable_ppstructure: bool = True,
    enable_img2table: bool = True,
    accept_threshold: float = 0.45,
) -> RegionBlock:
    crop = crop_region_bgr(image_bgr, region, pad=10)
    notes: list[str] = list(region.notes)
    attempts: list[TableParseResult] = []

    tmp = cache_dir / "tables" / doc_id / f"p{page:04d}_{region_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp / "crop.png"), crop)

    if enable_docling:
        r = parse_table_docling(crop, tmp)
        if r:
            attempts.append(r)
            notes.append(f"docling_score={r.confidence:.3f}")
            if r.confidence >= accept_threshold and r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)

    if enable_ppstructure:
        r = parse_table_ppstructure(crop)
        if r:
            attempts.append(r)
            notes.append(f"ppstructure_score={r.confidence:.3f}")
            if r.confidence >= accept_threshold and r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)

    if enable_img2table:
        r = parse_table_img2table(crop)
        if r:
            attempts.append(r)
            notes.append(f"img2table_score={r.confidence:.3f}")
            if r.confidence >= accept_threshold and r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)

    if vlm is not None:
        try:
            r = parse_table_vlm(crop, vlm)
            attempts.append(r)
            notes.append(f"vlm_score={r.confidence:.3f}")
            if r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vlm_fail:{exc}")

    # best effort among attempts
    best = max(attempts, key=lambda a: a.confidence) if attempts else None
    if best and best.markdown.strip():
        notes.append("best_effort_below_threshold")
        return _to_block(doc_id, page, region_id, region, best, notes, suspicious=True)

    return RegionBlock(
        doc_id=doc_id,
        page=page,
        region_id=region_id,
        type=BlockType.TABLE,
        bbox=region.bbox_pt,
        quality=QualitySignals(
            structural_ok=False, suspicion=1.0, notes=notes + ["all_backends_failed"]
        ),
        content={"markdown": "", "status": "failed", "attempts": [a.method for a in attempts]},
        provenance=Provenance(method="none"),
    )


def _to_block(
    doc_id: str,
    page: int,
    region_id: str,
    region: DetectedRegion,
    result: TableParseResult,
    notes: list[str],
    *,
    suspicious: bool = False,
) -> RegionBlock:
    all_notes = list(notes) + list(result.notes or [])
    return RegionBlock(
        doc_id=doc_id,
        page=page,
        region_id=region_id,
        type=BlockType.TABLE,
        bbox=region.bbox_pt,
        source="raster",
        quality=QualitySignals(
            confidence=result.confidence,
            structural_ok=not suspicious and result.confidence >= 0.45,
            suspicion=0.6 if suspicious else 0.0,
            notes=all_notes,
        ),
        content={
            "markdown": result.markdown,
            "html": result.html,
            "n_rows": result.n_rows,
            "n_cols": result.n_cols,
            "status": "suspicious" if suspicious else "ok",
        },
        provenance=Provenance(
            model=result.method, method=result.method, confidence=result.confidence
        ),
    )


def process_figure_region(
    image_bgr: np.ndarray,
    region: DetectedRegion,
    *,
    doc_id: str,
    page: int,
    region_id: str,
    cache_dir: Path,
    vlm: Any | None = None,
) -> RegionBlock:
    """Minimal figure handling: save crop + optional VLM caption."""
    crop = crop_region_bgr(image_bgr, region, pad=6)
    out_dir = cache_dir / "figures" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"p{page:04d}_{region_id}.png"
    cv2.imwrite(str(crop_path), crop)
    caption = ""
    method = "crop_only"
    if vlm is not None:
        try:
            caption = vlm.read_ndarray(
                crop,
                "Describe this figure in one short Russian or English sentence. "
                "If it is a chart, mention axes topic. No markdown.",
            ).strip()
            method = "vlm_caption"
        except Exception as exc:  # noqa: BLE001
            caption = ""
            method = f"crop_only:{exc}"
    return RegionBlock(
        doc_id=doc_id,
        page=page,
        region_id=region_id,
        type=BlockType.FIGURE,
        bbox=region.bbox_pt,
        quality=QualitySignals(confidence=region.score),
        content={"caption": caption, "crop_path": str(crop_path), "status": "ok"},
        provenance=Provenance(method=method, confidence=region.score),
    )
