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
    structural validators later.
    """
    if len(token) < 2:
        return token
    cyr_share, lat_share = _script_share(token)
    if cyr_share >= 0.6 and lat_share > 0:
        return token.translate(_HOMOGLYPH_TO_CYR)
    if lat_share >= 0.6 and cyr_share > 0:
        return token.translate(_HOMOGLYPH_TO_LAT)
    return token


_TOKEN = re.compile(r"\S+")


def normalize_ocr_text(text: str) -> str:
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return _TOKEN.sub(lambda m: fix_homoglyphs_in_token(m.group(0)), text)
