"""Classify PDF pages into A/B/C (D is assigned after failed VLM/OCR)."""

from __future__ import annotations

from ingestion.models import PageClass, TLQResult


def classify_page(
    tlq: TLQResult,
    *,
    tlq_threshold: float = 0.65,
    class_b_lex_min: float = 0.62,
    class_b_dict_min: float = 0.58,
) -> PageClass:
    """Map TLQ + lexical signals to handling class.

    A — clean usable text layer
    B — layer mostly usable (light OCR damage / structured OOV) → layer+OCR+normalize
    C — missing or garbage layer → preprocess + VLM
    D — not decided here (quality gate after extraction)
    """
    lex = tlq.components.lexical_quality
    dict_hit = tlq.components.dict_hit
    oov = tlq.components.oov_hard

    if not tlq.has_text_layer or tlq.char_count < 40:
        return PageClass.C

    if tlq.score < tlq_threshold and (lex is None or lex < 0.55):
        return PageClass.C

    # Clean accept
    if (
        not tlq.garbage_veto
        and tlq.score >= tlq_threshold
        and lex is not None
        and lex >= 0.85
    ):
        return PageClass.A

    # Light damage / structured docs: keep layer path (KT-160G-like)
    if tlq.has_text_layer and tlq.char_count >= 200:
        if lex is not None and dict_hit is not None and oov is not None:
            if lex >= class_b_lex_min and dict_hit >= class_b_dict_min and oov <= 0.32:
                if tlq.score >= 0.55 or tlq.garbage_veto:
                    return PageClass.B
        if not tlq.garbage_veto and tlq.score >= tlq_threshold:
            return PageClass.B

    # Strong garbage or weak layer
    if tlq.garbage_veto or (lex is not None and lex < class_b_lex_min):
        return PageClass.C

    if tlq.score < tlq_threshold:
        return PageClass.C

    return PageClass.B
