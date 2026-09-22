# -*- coding: utf-8 -*-
"""Technical-symbol helpers: Greek letters, math tokens, OCR confusions."""

from __future__ import annotations

import re

# Greek & Coptic block used in aerospace / math docs.
_GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_MATH_MARK_RE = re.compile(
    r"[=^_∞±≤≥≠≈∑∫∏√∂∇·×÷/\\]|\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma|\\theta|\\lambda|\\mu|\\pi|\\sigma|\\omega"
)

# Single-letter OCR confusions typical for Greek variables in scanned tech books.
# Applied only to isolated math tokens, never to multi-letter words.
_LATIN_TO_GREEK_MATH = {
    "a": "α",
    "A": "Α",
    "b": "β",
    "B": "Β",
    "g": "γ",
    "G": "Γ",
    "d": "δ",
    "D": "Δ",
    "e": "ε",
    "z": "ζ",
    "h": "η",
    "th": "θ",
    "i": "ι",
    "k": "κ",
    "l": "λ",
    "L": "Λ",
    "m": "μ",
    "M": "Μ",
    "n": "ν",
    "x": "ξ",
    "X": "Ξ",
    "p": "π",
    "P": "Π",
    "r": "ρ",
    "s": "σ",
    "S": "Σ",
    "t": "τ",
    "T": "Τ",
    "y": "υ",
    "ph": "φ",
    "f": "φ",
    "c": "χ",
    "w": "ω",
    "W": "Ω",
}

# Cyrillic lookalikes often substituted for Greek by a Cyrillic OCR head.
_CYR_TO_GREEK_MATH = {
    "а": "α",
    "А": "Α",
    "в": "β",
    "В": "Β",
    "г": "γ",
    "Г": "Γ",
    "д": "δ",
    "Д": "Δ",
    "е": "ε",
    "Е": "Ε",
    "и": "η",
    "к": "κ",
    "К": "Κ",
    "л": "λ",
    "Л": "Λ",
    "м": "μ",
    "М": "Μ",
    "н": "ν",
    "Н": "Ν",
    "о": "ο",
    "О": "Ο",
    "п": "π",
    "П": "Π",
    "р": "ρ",
    "Р": "Ρ",
    "с": "σ",
    "С": "Σ",
    "т": "τ",
    "Т": "Τ",
    "у": "υ",
    "ф": "φ",
    "Ф": "Φ",
    "х": "χ",
    "Х": "Χ",
    "ш": "ω",
}


def has_greek(text: str) -> bool:
    return bool(_GREEK_RE.search(text or ""))


def greek_ratio(text: str) -> float:
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _GREEK_RE.match(ch)) / len(letters)


def looks_like_formula_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _MATH_MARK_RE.search(t):
        return True
    if has_greek(t) and len(t) <= 48:
        return True
    # short token with digits and letters often is CT=0.2 / ωR etc.
    if len(t) <= 24 and any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t):
        if any(ch in t for ch in "=^_/+-−–—"):
            return True
    return False


def prefer_greek_ocr_candidate(cyr_text: str, cyr_score: float, el_text: str, el_score: float) -> bool:
    """Choose Greek-engine reading when it clearly carries Greek / math content."""
    if not el_text:
        return False
    if has_greek(el_text) and (el_score + 0.05 >= cyr_score or not has_greek(cyr_text)):
        return True
    if looks_like_formula_text(el_text) and el_score >= cyr_score - 0.08:
        return True
    if cyr_score < 0.55 and el_score >= cyr_score + 0.08 and greek_ratio(el_text) > 0:
        return True
    return False


_SINGLE_MATH_TOKEN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё\u0370-\u03FF])"
    r"(th|ph|[A-Za-zА-Яа-яЁё])"
    r"(?![A-Za-zА-Яа-яЁё\u0370-\u03FF])"
)


def recover_greek_math_tokens(text: str) -> str:
    """Map isolated Latin/Cyrillic lookalikes to Greek only in formula-like strings."""
    if not text or not looks_like_formula_text(text):
        return text
    if has_greek(text):
        # already mixed; only fill remaining single-letter holes carefully
        pass

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok in _LATIN_TO_GREEK_MATH:
            return _LATIN_TO_GREEK_MATH[tok]
        if tok in _CYR_TO_GREEK_MATH:
            return _CYR_TO_GREEK_MATH[tok]
        return tok

    return _SINGLE_MATH_TOKEN.sub(_sub, text)


def wrap_formula_for_markdown(
    text: str, latex: str | None = None, *, force: bool = False
) -> str:
    """Emit a table-cell friendly formula representation."""
    if latex and latex.strip():
        body = latex.strip().strip("$")
        return f"${body}$"
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("$") and t.endswith("$"):
        return t
    if force or looks_like_formula_text(t):
        return f"${t}$"
    return t.replace("|", r"\|")


_CYR_RE = re.compile(r"[\u0400-\u04FF]")
_LAT_RE = re.compile(r"[A-Za-z]")


def script_shares(text: str) -> tuple[float, float, float]:
    """Return (cyr_share, lat_share, greek_share) over alphabetic chars."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return 0.0, 0.0, 0.0
    n = len(letters)
    cyr = sum(1 for ch in letters if _CYR_RE.match(ch)) / n
    lat = sum(1 for ch in letters if _LAT_RE.match(ch)) / n
    gre = sum(1 for ch in letters if _GREEK_RE.match(ch)) / n
    return cyr, lat, gre


def prefer_latin_ocr_candidate(
    base_text: str, base_score: float, lat_text: str, lat_score: float
) -> bool:
    """Prefer Latin/English head for long Latin prose or mostly-Latin tokens."""
    if not lat_text:
        return False
    _bc, bl, bg = script_shares(base_text)
    _lc, ll, lg = script_shares(lat_text)
    if lg > 0.15:
        return False  # Greek content belongs to Greek head
    # Long English / Latin lines
    if ll >= 0.70 and len(lat_text) >= 12 and lat_score + 0.03 >= base_score:
        return True
    if ll >= 0.85 and lat_score >= base_score - 0.05:
        return True
    # Base looks Cyrillic-garbled English (low conf, mostly latin lookalikes mangled)
    if bl < 0.35 and ll >= 0.70 and lat_score >= base_score + 0.05:
        return True
    if base_score < 0.50 and lat_score >= base_score + 0.10 and ll >= 0.55:
        return True
    return False


def choose_ocr_candidate(
    candidates: list[tuple[str, float, str]],
) -> tuple[str, float, str]:
    """Pick best (text, score, lang) among overlapping multi-head readings.

    Priority by content:
    1. Greek / formula symbols
    2. Long Latin/English prose
    3. Cyrillic Russian prose
    otherwise highest confidence.
    """
    if not candidates:
        return "", 0.0, "none"
    scored: list[tuple[float, str, float, str]] = []
    for text, conf, lang in candidates:
        if not text:
            continue
        cyr, lat, gre = script_shares(text)
        bonus = 0.0
        longest = max((len(c[0]) for c in candidates if c[0]), default=0)
        if lang == "el" and (gre > 0.05 or looks_like_formula_text(text)):
            bonus += 0.18 + 0.25 * gre
            # Lone Greek letter must not steal a Cyrillic/Latin prose box.
            if len(text) <= 3 and longest >= 8:
                bonus -= 0.40
            has_cyr_alt = any(
                script_shares(t)[0] >= 0.5 for t, _s, _l in candidates if t and t != text
            )
            if len(text) <= 2 and has_cyr_alt:
                bonus -= 0.55
        if lang == "latin" and lat >= 0.55:
            bonus += 0.10 + 0.15 * min(1.0, len(text) / 40.0)
        if lang == "cyrillic" and cyr >= 0.55:
            bonus += 0.12 + 0.10 * min(1.0, len(text) / 40.0)
        if lang == "el" and gre < 0.05 and len(text) > 20:
            bonus -= 0.20
        if lang == "latin" and cyr >= 0.55 and lat < 0.30:
            bonus -= 0.20
        scored.append((conf + bonus, text, conf, lang))
    scored.sort(key=lambda x: x[0], reverse=True)
    _adj, text, conf, lang = scored[0]
    return text, conf, lang

