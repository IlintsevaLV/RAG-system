"""Light text normalization for OCR / text-layer (homoglyphs, whitespace)."""

from __future__ import annotations

import re
import unicodedata

# Visually confusable Latin ↔ Cyrillic pairs (common OCR / PDF embedding bugs).
# Map TO Latin for shared technical tokens (M, A in bolt sizes etc. is context-dependent);
# here we only normalize known OCR confusions inside mostly-Cyrillic words later.
_HOMOGLYPH_TO_CYR = str.maketrans(
    {
        "A": "А",
        "a": "а",
        "B": "В",
        "E": "Е",
        "e": "е",
        "K": "К",
        "M": "М",
        "H": "Н",
        "O": "О",
        "o": "о",
        "P": "Р",
        "p": "р",
        "C": "С",
        "c": "с",
        "T": "Т",
        "X": "Х",
        "x": "х",
        "y": "у",
    }
)

_HOMOGLYPH_TO_LAT = str.maketrans(
    {
        "А": "A",
        "а": "a",
        "В": "B",
        "Е": "E",
        "е": "e",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "о": "o",
        "Р": "P",
        "р": "p",
        "С": "C",
        "с": "c",
        "Т": "T",
        "Х": "X",
        "х": "x",
        "у": "y",
    }
)

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")
_GREEK = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_WS = re.compile(r"[ \t]+")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _script_share(token: str) -> tuple[float, float]:
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return 0.0, 0.0
    cyr = sum(1 for ch in letters if _CYR.match(ch))
    lat = sum(1 for ch in letters if _LAT.match(ch))
    n = len(letters)
    return cyr / n, lat / n


def fix_homoglyphs_in_token(token: str) -> str:
    """Inside a mostly-Cyrillic token, pull Latin lookalikes to Cyrillic (and vice versa).

    Does NOT rewrite mixed technical codes like M12-6g wholesale — those need
    structural validators later. Greek letters are preserved as-is.
    """
    if len(token) < 2:
        return token
    if _GREEK.search(token):
        return token
    cyr_share, lat_share = _script_share(token)
    if cyr_share >= 0.6 and lat_share > 0:
        return token.translate(_HOMOGLYPH_TO_CYR)
    if lat_share >= 0.6 and cyr_share > 0:
        return token.translate(_HOMOGLYPH_TO_LAT)
    return token


_TOKEN = re.compile(r"\S+")

# Spaced display headings: "Ж и р о д и н" / "Р е а к т и в н ы й"
_SPACED_LETTERS = re.compile(
    r"(?<![\wА-Яа-яЁё])"
    r"(?:[A-Za-zА-Яа-яЁё]\s+){2,}[A-Za-zА-Яа-яЁё]"
    r"(?![\wА-Яа-яЁё])"
)

_FOOTER_NOISE = re.compile(
    r"(?im)^(?:www\.vokb-la\.spb\.ru.*|.*Самол[её]т своими руками.*)\s*$"
)
_HEADER_SKB = re.compile(
    r'(?im)^(?:СК[БВ]|CK[BV])\s*[\"«]?Вулкан-Авиа[\"»]?\s*$'
)


def collapse_spaced_letters(text: str) -> str:
    """Join letter-spaced display words common in mid-century Russian books."""

    def _join(m: re.Match[str]) -> str:
        return re.sub(r"\s+", "", m.group(0))

    return _SPACED_LETTERS.sub(_join, text)


def strip_scan_boilerplate(text: str) -> str:
    """Drop repeating scan headers/footers (e.g. vokb-la watermark pages)."""
    lines = []
    for ln in text.splitlines():
        if _FOOTER_NOISE.match(ln.strip()):
            continue
        if _HEADER_SKB.match(ln.strip()):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def normalize_ocr_text(text: str) -> str:
    text = normalize_unicode(text)
    text = collapse_spaced_letters(text)
    text = strip_scan_boilerplate(text)
    text = normalize_whitespace(text)
    text = _TOKEN.sub(lambda m: fix_homoglyphs_in_token(m.group(0)), text)
    # Soft Greek recovery for formula-like OCR fragments (α/β/π/ω…).
    try:
        from ingestion.tech_symbols import recover_greek_math_tokens

        text = recover_greek_math_tokens(text)
    except Exception:
        pass
    return text
