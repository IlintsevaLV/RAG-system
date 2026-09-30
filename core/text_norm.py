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

# PDF text layers (LaTeX → PDF) often emit Greek letters that only look like Latin.
# Real formula letters (β λ π ω …) are not in this map.
_GREEK_LOOKALIKE_TO_LAT = {
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
    "α": "a",
    "γ": "g",
    "ε": "e",
    "κ": "k",
    "μ": "m",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "χ": "x",
    "Γ": "G",
}
_LOOKALIKE_CHARS = set(_GREEK_LOOKALIKE_TO_LAT)
_TRANSPARENT = set(" \t\n\r\u00a0.,;:!?\"'«»“”„()[]{}…·-/\\'’`´")
_ABBREV_PUNCT = set(".,")


def _lookalike_side(chars: list[str], index: int, step: int) -> str:
    """Walk away from a lookalike. Spaces, digits and punctuation are transparent."""
    j = index + step
    n = len(chars)
    while 0 <= j < n:
        ch = chars[j]
        if ch in _LOOKALIKE_CHARS or ch.isspace() or ch.isdigit() or ch in _TRANSPARENT:
            j += step
            continue
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return "latin"
        return "block"
    return "edge"


def _in_abbrev_cluster(chars: list[str], index: int) -> bool:
    """ε.γ. is an abbreviation, not a formula: two lookalikes joined by dots."""
    lo = index
    while lo > 0 and (chars[lo - 1] in _LOOKALIKE_CHARS or chars[lo - 1] in _ABBREV_PUNCT):
        lo -= 1
    hi = index
    n = len(chars)
    while hi + 1 < n and (chars[hi + 1] in _LOOKALIKE_CHARS or chars[hi + 1] in _ABBREV_PUNCT):
        hi += 1
    return sum(1 for k in range(lo, hi + 1) if chars[k] in _LOOKALIKE_CHARS) >= 2


def latinize_greek_lookalikes(text: str) -> str:
    """Replace Greek letters that only mimic Latin, and leave formula Greek in place.

    A lookalike is rewritten when a Latin letter is reachable through spaces, digits
    and punctuation, and neither side hits a math sign, Cyrillic, or a real Greek
    letter. ``σ = μ + λ`` stays Greek. ``ARP4754Α`` and ``ε.γ.`` become Latin.
    """
    if not text or not any(ch in _LOOKALIKE_CHARS for ch in text):
        return text
    chars = list(text)
    out: list[str] = []
    for i, ch in enumerate(chars):
        if ch not in _LOOKALIKE_CHARS:
            out.append(ch)
            continue
        left = _lookalike_side(chars, i, -1)
        right = _lookalike_side(chars, i, 1)
        latin_context = left != "block" and right != "block" and (left == "latin" or right == "latin")
        if latin_context or _in_abbrev_cluster(chars, i):
            out.append(_GREEK_LOOKALIKE_TO_LAT[ch])
        else:
            out.append(ch)
    return "".join(out)


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
