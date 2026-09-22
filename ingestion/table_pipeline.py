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
    n_cols = int(np.median(cols)) if cols else 0
    n_rows = len(rows)
    consistency = float(sum(1 for c in cols if abs(c - n_cols) <= 1) / max(1, len(cols)))
    cells: list[str] = []
    for ln in rows:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        cells.extend(parts)
    nonempty = sum(1 for c in cells if c) / max(1, len(cells))
    meaningful = sum(1 for c in cells if any(ch.isalnum() for ch in c)) / max(1, len(cells))
    score = (
        0.35 * consistency
        + 0.30 * nonempty
        + 0.20 * meaningful
        + 0.15 * min(1.0, n_rows / 5.0)
    )
    long_cells = sum(1 for c in cells if len(c) > 180)
    if long_cells and n_rows < 5:
        score *= 0.35
        notes.append("prose_like_cells")
    # TOC-like markdown: dotted leaders or trailing page numbers in most rows.
    toc_hits = sum(
        1
        for c in cells
        if "..." in c or "…" in c or re.search(r"\.\s*\.\s*\.", c)
    )
    page_num_hits = sum(1 for c in cells if re.fullmatch(r"\d{1,3}", c or ""))
    if toc_hits >= max(2, n_rows // 2) or (
        page_num_hits >= max(3, n_rows // 2) and n_cols <= 3
    ):
        score *= 0.15
        notes.append("toc_like_markdown")
    if meaningful < 0.25:
        score *= 0.4
        notes.append("low_meaningful_fill")
    if n_cols < 2 or n_rows < 2:
        score *= 0.5
        notes.append("too_small")
    # Extremely wide sparse grids from false line detection.
    if n_cols >= 10 and meaningful < 0.35:
        score *= 0.25
        notes.append("sparse_wide_grid")
    notes.append(
        f"rows={n_rows} cols={n_cols} consistency={consistency:.2f} "
        f"meaningful={meaningful:.2f}"
    )
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



def _upsample_cell(cell: np.ndarray, min_side: int = 48) -> np.ndarray:
    """Tiny table cells (sub/superscripts, Greek letters) need upsampling."""
    if cell is None or cell.size == 0:
        return cell
    h, w = cell.shape[:2]
    m = min(h, w)
    if m <= 0 or m >= min_side:
        return cell
    scale = max(2.0, float(min_side) / float(m))
    return cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _cell_visual_formula_score(cell: np.ndarray) -> float:
    """Heuristic: compact ink with superscripts / stacked glyphs looks like math."""
    if cell is None or cell.size == 0:
        return 0.0
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0
    ink = gray < max(40, int(np.median(gray) - 20))
    density = float(ink.mean())
    if density < 0.01 or density > 0.55:
        return 0.0
    # vertical variance of ink -> stacked exponents / fractions
    row_energy = ink.sum(axis=1)
    active = np.flatnonzero(row_energy > 0)
    if len(active) < 3:
        return 0.0
    span = (active[-1] - active[0] + 1) / max(1, h)
    score = 0.0
    if density < 0.28:
        score += 0.35
    if span > 0.55 and h / max(w, 1) > 0.55:
        score += 0.35
    edges = cv2.Canny(gray, 50, 150)
    if edges.mean() > 8:
        score += 0.2
    return min(1.0, score)


def _recognize_table_cell(
    cell: np.ndarray,
    *,
    dpi: int,
    unimer: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """OCR a cell; optionally recover in-cell formulas as LaTeX for Markdown."""
    from ingestion.ocr_rapid import RapidOCRConfig, recognize_image
    from ingestion.tech_symbols import (
        has_greek,
        looks_like_formula_text,
        wrap_formula_for_markdown,
    )

    meta: dict[str, Any] = {"kind": "text", "latex": "", "text": ""}
    if cell is None or cell.size == 0:
        return "", meta

    work = _upsample_cell(cell)
    # Start with the Cyrillic head. Running the Greek head on every ordinary
    # Russian cell produces visually plausible but incorrect Greek garbage.
    base_lines = recognize_image(
        work,
        dpi=dpi,
        page=0,
        doc_id="table_cell",
        cfg=RapidOCRConfig(max_side_len=2000, enable_latin=False, enable_greek=False),
    )
    # Multi-line cells: keep spaces (not newlines) so Markdown stays one row.
    text = " ".join(ln.text.strip() for ln in base_lines if ln.text.strip()).strip()
    meta["text"] = text
    visual = _cell_visual_formula_score(cell)
    from ingestion.tech_symbols import script_shares, choose_ocr_candidate

    base_has_cyr = any("\u0400" <= ch <= "\u04ff" for ch in text)
    base_conf = (
        sum(ln.confidence for ln in base_lines) / len(base_lines)
        if base_lines
        else 0.0
    )
    _, base_lat, _ = script_shares(text)
    # Secondary heads are reserved for math-like cells and clear English /
    # Latin-script cells. Ordinary Cyrillic prose stays on the Cyrillic head.
    formula_like = (
        looks_like_formula_text(text)
        or has_greek(text)
        or (visual >= 0.55 and not base_has_cyr)
    )
    needs_secondary = formula_like or (base_lat >= 0.55 and len(text) >= 8)
    if needs_secondary:
        all_lines = recognize_image(
            work,
            dpi=dpi,
            page=0,
            doc_id="table_cell",
            cfg=RapidOCRConfig(max_side_len=2000, enable_latin=True, enable_greek=True),
        )
        all_text = " ".join(
            ln.text.strip() for ln in all_lines if ln.text.strip()
        ).strip()
        all_conf = (
            sum(ln.confidence for ln in all_lines) / len(all_lines)
            if all_lines
            else 0.0
        )
        chosen, chosen_conf, chosen_lang = choose_ocr_candidate(
            [
                (text, base_conf, "cyrillic"),
                (all_text, all_conf, "latin+greek"),
            ]
        )
        # Never replace a Cyrillic prose reading with a Greek-looking
        # hallucination unless the Cyrillic result itself is math-like.
        if not (
            has_greek(chosen)
            and any("\u0400" <= ch <= "\u04ff" for ch in text)
            and not looks_like_formula_text(text)
        ):
            text = chosen
            base_conf = chosen_conf
        meta["ocr_heads"] = "cyrillic+latin+greek"
        meta["selected_head"] = chosen_lang
    else:
        meta["ocr_heads"] = "cyrillic"
        meta["selected_head"] = "cyrillic"
    meta["text"] = text
    formula_like = (
        looks_like_formula_text(text)
        or has_greek(text)
        or (visual >= 0.55 and not any("\u0400" <= ch <= "\u04ff" for ch in text))
    )

    latex = ""
    if formula_like and unimer is not None:
        try:
            latex = (unimer.recognize(work) or "").strip()
        except Exception as exc:  # noqa: BLE001
            meta["unimer_fail"] = str(exc)
    if latex:
        meta["kind"] = "formula"
        meta["latex"] = latex
        return wrap_formula_for_markdown(text, latex), meta
    if formula_like and text:
        meta["kind"] = "formula_text"
        return wrap_formula_for_markdown(text, None), meta
    return text.replace("|", r"\|"), meta


def parse_table_cells_ocr(
    image_bgr: np.ndarray,
    *,
    dpi: int = 200,
    unimer: Any | None = None,
) -> TableParseResult | None:
    """Recover a lined table by OCR-ing each detected grid cell.

    Handles:
    - tiny cells (upsample)
    - multi-line cells (join)
    - Greek letters via dual OCR
    - formula-like cells -> optional UniMERNet LaTeX inside `$...$`
    """
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        hk = max(w // 35, 18)
        vk = max(h // 35, 18)
        horizontal = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        )
        vertical = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
        )
        x_energy = vertical.sum(axis=0) / 255.0
        y_energy = horizontal.sum(axis=1) / 255.0

        def _peaks(values: np.ndarray, minimum: float) -> list[int]:
            ids = np.flatnonzero(values >= minimum).tolist()
            groups: list[list[int]] = []
            for idx in ids:
                if not groups or idx - groups[-1][-1] > 5:
                    groups.append([idx])
                else:
                    groups[-1].append(idx)
            return [int(sum(g) / len(g)) for g in groups]

        xs = _peaks(x_energy, max(3.0, h * 0.10))
        ys = _peaks(y_energy, max(3.0, w * 0.10))
        if len(xs) < 3 or len(ys) < 3 or len(xs) * len(ys) > 400:
            return None

        rows: list[list[str]] = []
        cell_meta: list[list[dict[str, Any]]] = []
        nonempty = 0
        meaningful = 0
        formula_cells = 0
        for y0, y1 in zip(ys, ys[1:]):
            row: list[str] = []
            meta_row: list[dict[str, Any]] = []
            for x0, x1 in zip(xs, xs[1:]):
                pad_x = max(2, int((x1 - x0) * 0.06))
                pad_y = max(2, int((y1 - y0) * 0.10))
                y_a, y_b = min(h, y0 + pad_y), max(0, y1 - pad_y)
                x_a, x_b = min(w, x0 + pad_x), max(0, x1 - pad_x)
                if y_b <= y_a or x_b <= x_a:
                    cell = image_bgr[0:0, 0:0]
                else:
                    cell = image_bgr[y_a:y_b, x_a:x_b]
                text, meta = _recognize_table_cell(cell, dpi=dpi, unimer=unimer)
                row.append(text)
                meta_row.append(meta)
                nonempty += bool(meta.get("text") or text)
                meaningful += bool(meta.get("text")) and any(
                    ch.isalnum() or ("\u0370" <= ch <= "\u03FF") for ch in meta.get("text", "")
                )
                if meta.get("kind", "").startswith("formula"):
                    formula_cells += 1
            rows.append(row)
            cell_meta.append(meta_row)
        if len(rows) < 2:
            return None
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        md = "\n".join(
            ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
            + ["| " + " | ".join(r) + " |" for r in rows[1:]]
        )
        total_cells = max(1, len(rows) * width)
        fill_ratio = meaningful / total_cells
        score = min(0.95, 0.15 + 0.85 * fill_ratio)
        if formula_cells:
            score = min(0.98, score + 0.05)
        if fill_ratio < 0.22 or meaningful < max(4, int(0.2 * total_cells)):
            return TableParseResult(
                markdown="",
                n_rows=len(rows),
                n_cols=width,
                method="cell_ocr",
                confidence=min(score, 0.20),
                notes=[
                    f"grid={len(xs)}x{len(ys)}",
                    f"nonempty={nonempty}",
                    f"meaningful_ratio={fill_ratio:.3f}",
                    "rejected_sparse_grid",
                ],
            )
        return TableParseResult(
            markdown=md,
            n_rows=len(rows),
            n_cols=width,
            method="cell_ocr",
            confidence=score,
            notes=[
                f"grid={len(xs)}x{len(ys)}",
                f"nonempty={nonempty}",
                f"meaningful_ratio={fill_ratio:.3f}",
                f"formula_cells={formula_cells}",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cell_ocr_fail", error=str(exc))
        return TableParseResult(
            markdown="", method="cell_ocr", confidence=0.0, notes=[f"fail:{exc}"]
        )


def parse_table_from_spans(
    spans: list[Any],
    region: DetectedRegion,
    *,
    page_w: float,
) -> TableParseResult | None:
    """Rebuild Markdown from text/OCR spans inside an unruled table bbox."""
    if not spans:
        return None
    pad = 4.0
    box = region.bbox_pt
    inside = []
    for sp in spans:
        bbox = getattr(sp, "bbox", None)
        text = (getattr(sp, "text", "") or "").strip()
        if bbox is None or not text:
            continue
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        if box.x1 - pad <= cx <= box.x2 + pad and box.y1 - pad <= cy <= box.y2 + pad:
            inside.append(sp)
    if len(inside) < 6:
        return None

    # row cluster
    items = sorted(inside, key=lambda s: (0.5 * (s.bbox[1] + s.bbox[3]), s.bbox[0]))
    rows_sp: list[list[Any]] = []
    for sp in items:
        cy = 0.5 * (sp.bbox[1] + sp.bbox[3])
        if not rows_sp or abs(0.5 * (rows_sp[-1][0].bbox[1] + rows_sp[-1][0].bbox[3]) - cy) > 10:
            rows_sp.append([sp])
        else:
            rows_sp[-1].append(sp)
    if len(rows_sp) < 2:
        return None

    xs = [0.5 * (sp.bbox[0] + sp.bbox[2]) for sp in inside]
    import numpy as np

    hist, edges = np.histogram(xs, bins=max(6, min(14, len(xs))))
    thr = max(2.0, float(hist.max()) * 0.35)
    peaks = [0.5 * (edges[i] + edges[i + 1]) for i, v in enumerate(hist) if v >= thr]
    merged: list[float] = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > 16:
            merged.append(float(p))
        else:
            merged[-1] = 0.5 * (merged[-1] + p)
    if len(merged) < 2:
        return None

    grid: list[list[str]] = []
    for row in rows_sp:
        cells = [""] * len(merged)
        for sp in sorted(row, key=lambda s: s.bbox[0]):
            cx = 0.5 * (sp.bbox[0] + sp.bbox[2])
            j = min(range(len(merged)), key=lambda k: abs(merged[k] - cx))
            from ingestion.tech_symbols import (
                has_greek,
                looks_like_formula_text,
                wrap_formula_for_markdown,
            )
            raw = sp.text.strip()
            # Greek-only OCR output from the EL head is unsafe when there is
            # no equation marker. Preserve recall as UNREADABLE instead of
            # indexing a plausible-looking hallucination.
            greek_text = has_greek(raw) and not any(
                ch in raw for ch in "=^_\\/+-−×÷∑∫√"
            )
            if greek_text and not any("\u0400" <= ch <= "\u04ff" for ch in raw):
                val = "UNREADABLE"
            else:
                val = (
                    wrap_formula_for_markdown(raw)
                    if looks_like_formula_text(raw)
                    else raw.replace("|", r"\|")
                )
            cells[j] = (cells[j] + " " + val).strip() if cells[j] else val
        grid.append(cells)

    # drop empty trailing columns
    while len(grid[0]) > 2 and all(not r[-1] for r in grid):
        grid = [r[:-1] for r in grid]
    width = len(grid[0])
    if width < 2 or len(grid) < 2:
        return None
    nonempty = sum(1 for r in grid for c in r if c)
    if nonempty / max(1, len(grid) * width) < 0.30:
        return None

    md = "\n".join(
        ["| " + " | ".join(grid[0]) + " |",
         "| " + " | ".join(["---"] * width) + " |"]
        + ["| " + " | ".join(r) + " |" for r in grid[1:]]
    )
    score, nr, nc, notes = _md_quality(md)
    return TableParseResult(
        markdown=md,
        n_rows=nr,
        n_cols=nc,
        method="span_cells",
        confidence=max(score, 0.55),
        notes=notes + [f"span_cols={width}", f"span_rows={len(grid)}"],
    )


def parse_table_vlm(image_bgr: np.ndarray, vlm: Any) -> TableParseResult:
    prompt = (
        "Extract the table from this image as GitHub-flavored Markdown only. "
        "Use | columns | and a header separator row. "
        "Preserve Greek letters (α β γ θ λ μ π σ ω) exactly. "
        "If a cell contains a formula, write it as $LaTeX$ inside the cell. "
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
    unimer: Any | None = None,
    spans: list[Any] | None = None,
    page_w: float | None = None,
    enable_docling: bool = True,
    enable_ppstructure: bool = True,
    enable_img2table: bool = True,
    enable_cell_ocr: bool = True,
    accept_threshold: float = 0.45,
) -> RegionBlock:
    crop = crop_region_bgr(image_bgr, region, pad=10)
    notes: list[str] = list(region.notes)
    attempts: list[TableParseResult] = []

    tmp = cache_dir / "tables" / doc_id / f"p{page:04d}_{region_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp / "crop.png"), crop)

    # Unruled / lightly ruled tables: rebuild Markdown from aligned spans first.
    if spans and page_w:
        r = parse_table_from_spans(spans, region, page_w=page_w)
        if r:
            attempts.append(r)
            notes.append(f"span_cells_score={r.confidence:.3f}")
            if r.confidence >= accept_threshold and r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)

    if enable_cell_ocr:
        r = parse_table_cells_ocr(crop, unimer=unimer)
        if r:
            attempts.append(r)
            notes.append(f"cell_ocr_score={r.confidence:.3f}")
            if r.confidence >= accept_threshold and r.markdown.strip():
                return _to_block(doc_id, page, region_id, region, r, notes)

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

    # Prefer an explicit failure over garbage markdown that pollutes RAG.
    usable = [
        a
        for a in attempts
        if a.markdown.strip() and a.confidence >= max(0.28, accept_threshold * 0.55)
    ]
    best = max(usable, key=lambda a: a.confidence) if usable else None
    if best:
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
        content={
            "markdown": "",
            "status": "failed",
            "attempts": [a.method for a in attempts],
        },
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
