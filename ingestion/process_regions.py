"""Orchestrate region detection + formula/table/figure pipelines for a PDF page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from core.config import Settings, get_settings
from core.ir import BlockType, RegionBlock
from core.logger import get_logger
from core.vlm_client import LlamaVisionClient
from ingestion.formula_pipeline import UniMERNetRecognizer, process_formula_region
from ingestion.region_detect import DetectedRegion, detect_page_regions
from ingestion.table_pipeline import process_figure_region, process_table_region

log = get_logger("regions")


def _vlm(settings: Settings) -> LlamaVisionClient | None:
    if not settings.enable_vlm:
        return None
    client = LlamaVisionClient(
        host=settings.model_host,
        port=settings.model_port,
        timeout_s=settings.vlm_timeout_s,
    )
    return client if client.ready() else None


def process_page_special_blocks(
    doc: fitz.Document,
    *,
    doc_id: str,
    page_number: int,
    settings: Settings | None = None,
    unimer: UniMERNetRecognizer | None = None,
    vlm: LlamaVisionClient | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    image_bgr, detections, _seed_blocks, page_spans = detect_page_regions(
        doc,
        page_number,
        doc_id=doc_id,
        dpi=settings.render_dpi,
    )
    page_w = float(doc[page_number - 1].rect.width)
    if vlm is None:
        vlm = _vlm(settings)
    if unimer is None and settings.enable_unimernet:
        unimer = UniMERNetRecognizer(
            settings.unimernet_model,
            config_path=settings.unimernet_config_path,
        )

    blocks: list[RegionBlock] = []
    for i, det in enumerate(detections):
        rid = f"{det.type.value}_{i:02d}"
        if det.type == BlockType.FORMULA:
            blocks.append(
                process_formula_region(
                    image_bgr,
                    det,
                    doc_id=doc_id,
                    page=page_number,
                    region_id=rid,
                    unimer=unimer,
                    vlm=vlm if settings.formula_vlm_recovery else None,
                )
            )
        elif det.type == BlockType.TABLE:
            blocks.append(
                process_table_region(
                    image_bgr,
                    det,
                    doc_id=doc_id,
                    page=page_number,
                    region_id=rid,
                    cache_dir=settings.cache_dir,
                    vlm=vlm if settings.table_vlm_recovery else None,
                    unimer=unimer,
                    spans=page_spans,
                    page_w=page_w,
                    enable_docling=settings.table_enable_docling,
                    enable_ppstructure=settings.table_enable_ppstructure,
                    enable_img2table=settings.table_enable_img2table,
                    enable_cell_ocr=settings.table_enable_cell_ocr,
                    accept_threshold=settings.table_accept_threshold,
                )
            )
        elif det.type == BlockType.FIGURE:
            blocks.append(
                process_figure_region(
                    image_bgr,
                    det,
                    doc_id=doc_id,
                    page=page_number,
                    region_id=rid,
                    cache_dir=settings.cache_dir,
                    vlm=vlm if settings.figure_vlm_caption else None,
                )
            )

    summary = {
        "page": page_number,
        "n_detections": len(detections),
        "n_formulas": sum(1 for b in blocks if b.type == BlockType.FORMULA),
        "n_tables": sum(1 for b in blocks if b.type == BlockType.TABLE),
        "n_figures": sum(1 for b in blocks if b.type == BlockType.FIGURE),
        "table_ok": sum(
            1
            for b in blocks
            if b.type == BlockType.TABLE and b.content.get("status") == "ok"
        ),
        "formula_ok": sum(
            1
            for b in blocks
            if b.type == BlockType.FORMULA and b.content.get("status") == "ok"
        ),
    }
    return {
        "doc_id": doc_id,
        "page": page_number,
        "summary": summary,
        "detections": [
            {
                "type": d.type.value,
                "score": d.score,
                "method": d.method,
                "bbox": d.bbox_pt.model_dump(),
            }
            for d in detections
        ],
        "blocks": [b.model_dump(mode="json") for b in blocks],
    }


def process_pdf_regions(
    path: str | Path,
    *,
    doc_id: str | None = None,
    pages: list[int] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    path = Path(path)
    doc_id = doc_id or path.stem
    vlm = _vlm(settings)
    unimer = (
        UniMERNetRecognizer(
            settings.unimernet_model,
            config_path=settings.unimernet_config_path,
        )
        if settings.enable_unimernet
        else None
    )
    doc = fitz.open(path)
    try:
        total = doc.page_count
        page_nums = pages or list(range(1, total + 1))
        page_nums = [p for p in page_nums if 1 <= p <= total]
        pages_out = []
        for pno in page_nums:
            log.info("regions_page", doc_id=doc_id, page=pno)
            pages_out.append(
                process_page_special_blocks(
                    doc,
                    doc_id=doc_id,
                    page_number=pno,
                    settings=settings,
                    unimer=unimer,
                    vlm=vlm,
                )
            )
        return {
            "doc_id": doc_id,
            "source_path": str(path.resolve()),
            "page_count": total,
            "pages": pages_out,
        }
    finally:
        doc.close()
