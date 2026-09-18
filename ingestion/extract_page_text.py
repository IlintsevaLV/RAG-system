"""Extract RAG-ready page text by class A/B (layer+OCR+norm) or C/D (preprocess+VLM)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from core.config import Settings, get_settings
from core.logger import get_logger
from core.text_norm import normalize_ocr_text
from core.vlm_client import (
    LAYOUT_CHECK_PROMPT,
    VLM_PAGE_READ_PROMPT,
    LlamaVisionClient,
)
from ingestion.layout_columns import enrich_layout_hints, reorder_spans_by_columns
from ingestion.models import (
    DocumentTextExtract,
    ExtractStatus,
    LayoutHints,
    PageClass,
    PageRoute,
    PageTextExtract,
)
from ingestion.ocr_rapid import RapidOCRConfig, recognize_page_image
from ingestion.page_classify import classify_page
from ingestion.preprocess import process_page_image, save_preview
from ingestion.source_analysis import analyze_page

log = get_logger("extract_page")


def _word_jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", a.lower()))
    wb = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", b.lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _parse_layout_vlm(reply: str) -> LayoutHints:
    hints = LayoutHints(vlm_notes=reply.strip()[:1200])
    up = reply.upper()
    if "COLUMNS:" in up:
        line = [ln for ln in reply.splitlines() if "COLUMNS:" in ln.upper()]
        if line:
            m = re.search(r"(\d+)", line[0])
            if m:
                hints.n_columns = max(1, min(4, int(m.group(1))))
    if re.search(r"TABLE:\s*Y", up) or "TABLE: YES" in up:
        hints.likely_table = True
    if re.search(r"UNUSUAL:\s*Y", up) or "UNUSUAL: YES" in up:
        hints.unusual_layout = True
    if re.search(r"MISSING:\s*Y", up) or "MISSING: YES" in up:
        hints.notes.append("vlm_reports_missing_regions")
    return hints


def _get_vlm(settings: Settings) -> LlamaVisionClient | None:
    if not settings.enable_vlm:
        return None
    client = LlamaVisionClient(
        host=settings.model_host,
        port=settings.model_port,
        timeout_s=settings.vlm_timeout_s,
    )
    if not client.ready():
        log.warning("vlm_not_ready", host=settings.model_host, port=settings.model_port)
        return None
    return client


def extract_page(
    doc: fitz.Document,
    *,
    doc_id: str,
    page_number: int,
    settings: Settings | None = None,
    vlm: LlamaVisionClient | None = None,
    save_previews: bool = True,
) -> PageTextExtract:
    settings = settings or get_settings()
    page = doc[page_number - 1]
    analysis = analyze_page(
        page,
        doc_id=doc_id,
        page_number=page_number,
        tlq_threshold=settings.tlq_threshold,
        enable_visual=settings.tlq_enable_visual_agreement,
        lexical_veto_threshold=settings.lexical_veto_threshold,
        lexical_min_tokens=settings.lexical_min_tokens,
    )
    page_class = classify_page(
        analysis.tlq,
        tlq_threshold=settings.tlq_threshold,
        class_b_lex_min=settings.class_b_lex_min,
        class_b_dict_min=settings.class_b_dict_min,
    )

    notes = [f"classified={page_class.value}"]
    notes.extend(analysis.tlq.notes[:5])

    if page_class in (PageClass.A, PageClass.B):
        return _extract_ab(
            doc,
            analysis=analysis,
            page_class=page_class,
            settings=settings,
            vlm=vlm,
            notes=notes,
            save_previews=save_previews,
        )
    return _extract_cd(
        doc,
        analysis=analysis,
        page_class=PageClass.C,
        settings=settings,
        vlm=vlm,
        notes=notes,
        save_previews=save_previews,
    )


def _extract_ab(
    doc: fitz.Document,
    *,
    analysis,
    page_class: PageClass,
    settings: Settings,
    vlm: LlamaVisionClient | None,
    notes: list[str],
    save_previews: bool,
) -> PageTextExtract:
    spans = analysis.text_spans
    # Re-fetch spans if classifier kept layer but analysis dropped spans on OCR route
    if not spans and analysis.tlq.has_text_layer:
        from ingestion.source_analysis import _extract_spans

        text_raw, spans = _extract_spans(doc[analysis.page - 1])
    else:
        text_raw = analysis.extracted_text

    text_cols, layout = reorder_spans_by_columns(spans, analysis.width_pt)
    layout = enrich_layout_hints(spans, analysis.width_pt, layout)
    base_text = text_cols.strip() or text_raw
    text = normalize_ocr_text(base_text)
    text_source = "layer"
    confidence = analysis.tlq.components.lexical_quality
    status = ExtractStatus.OK
    preview_path = None
    provenance: dict[str, Any] = {
        "tlq": analysis.tlq.score,
        "lexical_quality": analysis.tlq.components.lexical_quality,
        "garbage_veto": analysis.tlq.garbage_veto,
    }

    # OCR cross-check for class B (and optional for A if enabled)
    need_ocr = page_class == PageClass.B or settings.ab_always_ocr_check
    ocr_text = ""
    prep_img = None
    if need_ocr:
        prep = process_page_image(
            doc,
            analysis.page,
            route=PageRoute.OCR,
            dpi_good=settings.render_dpi,
            dpi_bad=settings.render_dpi_bad,
            enable_binarize=False,
            force=True,
        )
        prep_img = prep
        if prep.get("image_bgr") is not None:
            if save_previews:
                preview_path = str(
                    settings.cache_dir
                    / "extract"
                    / analysis.doc_id
                    / f"page_{analysis.page:04d}_ab.png"
                )
                save_preview(prep["image_bgr"], preview_path)
            ocr_cfg = RapidOCRConfig(
                use_gpu=settings.enable_gpu_ocr,
                max_side_len=settings.ocr_max_side_len,
                band_trigger_px=settings.ocr_band_trigger_px,
                band_height=settings.ocr_band_height,
                model_dir=str(settings.ocr_model_dir),
            )
            ocr_res = recognize_page_image(
                prep["image_bgr"],
                dpi=int(prep.get("dpi") or settings.render_dpi),
                page=analysis.page,
                doc_id=analysis.doc_id,
                cfg=ocr_cfg,
                bands=prep.get("bands"),
            )
            ocr_text = normalize_ocr_text(ocr_res.text)
            jac = _word_jaccard(text, ocr_text)
            provenance["ocr_agreement"] = round(jac, 4)
            provenance["ocr_lines"] = ocr_res.line_count
            notes.append(f"ocr_agreement={jac:.3f}")

            # Escalate B → C/VLM path when layer strongly disagrees with OCR
            # (US Army-like garbage layer with high TLQ).
            if (
                page_class == PageClass.B
                and jac < settings.ab_escalate_agree
                and ocr_res.line_count >= 8
            ):
                notes.append(
                    f"escalate_B_to_C agreement={jac:.3f}<{settings.ab_escalate_agree}"
                )
                return _extract_cd(
                    doc,
                    analysis=analysis,
                    page_class=PageClass.C,
                    settings=settings,
                    vlm=vlm,
                    notes=notes,
                    save_previews=save_previews,
                )

            if jac < settings.ab_ocr_agree_min and len(ocr_text) > len(text) * 0.6:
                text = ocr_text
                text_source = "layer+ocr"
                notes.append("switched_to_ocr_due_to_low_agreement")
                status = ExtractStatus.SUSPICIOUS
            elif jac < settings.ab_ocr_agree_min:
                status = ExtractStatus.SUSPICIOUS
                notes.append("low_layer_ocr_agreement")
            else:
                text_source = "layer+ocr_check"

    # Light VLM layout control
    if settings.enable_vlm and settings.ab_enable_vlm_check and vlm is not None:
        if prep_img is None:
            prep_img = process_page_image(
                doc,
                analysis.page,
                route=PageRoute.OCR,
                dpi_good=min(150, settings.render_dpi),
                dpi_bad=min(200, settings.render_dpi_bad),
                force=True,
            )
        if prep_img.get("image_bgr") is not None:
            truncated = text[:2500]
            prompt = LAYOUT_CHECK_PROMPT.format(extracted=truncated)
            try:
                reply = vlm.read_ndarray(prep_img["image_bgr"], prompt)
                vlm_layout = _parse_layout_vlm(reply)
                layout.n_columns = max(layout.n_columns, vlm_layout.n_columns)
                layout.likely_table = layout.likely_table or vlm_layout.likely_table
                layout.unusual_layout = layout.unusual_layout or vlm_layout.unusual_layout
                layout.vlm_notes = vlm_layout.vlm_notes
                layout.notes.extend(vlm_layout.notes)
                notes.append("vlm_layout_check_done")
                if "vlm_reports_missing_regions" in layout.notes and ocr_text:
                    text = ocr_text
                    text_source = "layer+ocr"
                    status = ExtractStatus.SUSPICIOUS
                    notes.append("vlm_missing_regions_used_ocr")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"vlm_layout_check_failed:{exc}")

    if layout.likely_table:
        notes.append("likely_table_flagged_for_later_stage")
    if layout.n_columns >= 2:
        notes.append(f"columns={layout.n_columns}")

    return PageTextExtract(
        doc_id=analysis.doc_id,
        page=analysis.page,
        page_class=page_class,
        status=status,
        text=text,
        text_source=text_source,
        layout=layout,
        tlq_score=analysis.tlq.score,
        lexical_quality=analysis.tlq.components.lexical_quality,
        confidence=confidence,
        preview_path=preview_path,
        notes=notes,
        provenance=provenance,
    )


def _extract_cd(
    doc: fitz.Document,
    *,
    analysis,
    page_class: PageClass,
    settings: Settings,
    vlm: LlamaVisionClient | None,
    notes: list[str],
    save_previews: bool,
) -> PageTextExtract:
    prep = process_page_image(
        doc,
        analysis.page,
        route=PageRoute.OCR,
        dpi_good=settings.render_dpi,
        dpi_bad=settings.render_dpi_bad,
        enable_binarize=settings.preprocess_enable_binarize,
        force=True,
    )
    iqs_level = prep["iqs"].level.value if prep.get("iqs") is not None else None
    preview_path = None
    if prep.get("image_bgr") is not None and save_previews:
        preview_path = str(
            settings.cache_dir
            / "extract"
            / analysis.doc_id
            / f"page_{analysis.page:04d}_cd.png"
        )
        save_preview(prep["image_bgr"], preview_path)

    provenance: dict[str, Any] = {
        "tlq": analysis.tlq.score,
        "lexical_quality": analysis.tlq.components.lexical_quality,
        "preprocess_steps": prep.get("steps") or [],
        "iqs_level": iqs_level,
    }
    layout = LayoutHints()

    # Primary: VLM
    if settings.enable_vlm and vlm is not None and prep.get("image_bgr") is not None:
        try:
            raw = vlm.read_ndarray(prep["image_bgr"], VLM_PAGE_READ_PROMPT)
            raw = (raw or "").strip()
            if raw.upper().startswith("UNREADABLE") or len(raw) < 40:
                notes.append("vlm_unreadable")
                return PageTextExtract(
                    doc_id=analysis.doc_id,
                    page=analysis.page,
                    page_class=PageClass.D,
                    status=ExtractStatus.FAILED,
                    text="",
                    text_source="vlm",
                    layout=layout,
                    tlq_score=analysis.tlq.score,
                    lexical_quality=analysis.tlq.components.lexical_quality,
                    iqs_level=iqs_level,
                    confidence=0.0,
                    preview_path=preview_path,
                    notes=notes + ["failed_unreadable_like_p572"],
                    provenance=provenance,
                )
            text = normalize_ocr_text(raw)
            # crude column note from VLM length
            if "\n\n" in text:
                layout.notes.append("vlm_paragraph_breaks")
            return PageTextExtract(
                doc_id=analysis.doc_id,
                page=analysis.page,
                page_class=PageClass.C,
                status=ExtractStatus.OK,
                text=text,
                text_source="vlm",
                layout=layout,
                tlq_score=analysis.tlq.score,
                lexical_quality=analysis.tlq.components.lexical_quality,
                iqs_level=iqs_level,
                confidence=0.7,
                preview_path=preview_path,
                notes=notes + ["vlm_extract_ok"],
                provenance=provenance,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vlm_failed:{exc}")

    if not settings.enable_vlm or vlm is None:
        notes.append("vlm_disabled_or_unavailable")
        if not settings.cd_ocr_fallback:
            return PageTextExtract(
                doc_id=analysis.doc_id,
                page=analysis.page,
                page_class=PageClass.C,
                status=ExtractStatus.NEEDS_VLM,
                text="",
                text_source="",
                layout=layout,
                tlq_score=analysis.tlq.score,
                lexical_quality=analysis.tlq.components.lexical_quality,
                iqs_level=iqs_level,
                preview_path=preview_path,
                notes=notes,
                provenance=provenance,
            )

    # Fallback OCR when VLM off or failed
    if prep.get("image_bgr") is None:
        return PageTextExtract(
            doc_id=analysis.doc_id,
            page=analysis.page,
            page_class=PageClass.D,
            status=ExtractStatus.FAILED,
            text="",
            text_source="",
            notes=notes + ["no_image_for_ocr"],
            provenance=provenance,
        )

    ocr_cfg = RapidOCRConfig(
        use_gpu=settings.enable_gpu_ocr,
        max_side_len=settings.ocr_max_side_len,
        band_trigger_px=settings.ocr_band_trigger_px,
        band_height=settings.ocr_band_height,
        model_dir=str(settings.ocr_model_dir),
    )
    ocr_res = recognize_page_image(
        prep["image_bgr"],
        dpi=int(prep.get("dpi") or settings.render_dpi),
        page=analysis.page,
        doc_id=analysis.doc_id,
        cfg=ocr_cfg,
        bands=prep.get("bands"),
    )
    text = normalize_ocr_text(ocr_res.text)
    provenance["ocr_mean_confidence"] = ocr_res.mean_confidence
    provenance["ocr_lines"] = ocr_res.line_count

    if ocr_res.line_count < 3 or ocr_res.mean_confidence < settings.cd_fail_ocr_conf:
        return PageTextExtract(
            doc_id=analysis.doc_id,
            page=analysis.page,
            page_class=PageClass.D,
            status=ExtractStatus.FAILED,
            text=text,
            text_source="ocr_fallback",
            layout=layout,
            tlq_score=analysis.tlq.score,
            lexical_quality=analysis.tlq.components.lexical_quality,
            iqs_level=iqs_level,
            confidence=ocr_res.mean_confidence,
            preview_path=preview_path,
            notes=notes + ["ocr_fallback_failed_unreadable"],
            provenance=provenance,
        )

    return PageTextExtract(
        doc_id=analysis.doc_id,
        page=analysis.page,
        page_class=PageClass.C,
        status=ExtractStatus.SUSPICIOUS if not settings.enable_vlm else ExtractStatus.OK,
        text=text,
        text_source="ocr_fallback",
        layout=layout,
        tlq_score=analysis.tlq.score,
        lexical_quality=analysis.tlq.components.lexical_quality,
        iqs_level=iqs_level,
        confidence=ocr_res.mean_confidence,
        preview_path=preview_path,
        notes=notes + ["ocr_fallback_used"],
        provenance=provenance,
    )


def extract_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    pages: list[int] | None = None,
    settings: Settings | None = None,
) -> DocumentTextExtract:
    settings = settings or get_settings()
    path = Path(path)
    doc_id = doc_id or path.stem
    vlm = _get_vlm(settings)
    doc = fitz.open(path)
    try:
        total = doc.page_count
        page_nums = pages or list(range(1, total + 1))
        page_nums = [p for p in page_nums if 1 <= p <= total]
        results: list[PageTextExtract] = []
        for pno in page_nums:
            log.info("extract_page_start", doc_id=doc_id, page=pno)
            results.append(
                extract_page(
                    doc,
                    doc_id=doc_id,
                    page_number=pno,
                    settings=settings,
                    vlm=vlm,
                )
            )
        summary = {
            "pages": len(results),
            "class_A": sum(1 for r in results if r.page_class == PageClass.A),
            "class_B": sum(1 for r in results if r.page_class == PageClass.B),
            "class_C": sum(1 for r in results if r.page_class == PageClass.C),
            "class_D": sum(1 for r in results if r.page_class == PageClass.D),
            "ok": sum(1 for r in results if r.status == ExtractStatus.OK),
            "failed": sum(1 for r in results if r.status == ExtractStatus.FAILED),
            "needs_vlm": sum(1 for r in results if r.status == ExtractStatus.NEEDS_VLM),
        }
        return DocumentTextExtract(
            doc_id=doc_id,
            source_path=str(path.resolve()),
            page_count=total,
            pages=results,
            summary=summary,
        )
    finally:
        doc.close()
