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

# Greek letters that PDF text layers use instead of Latin or Cyrillic.
# λ π ω and other real formula letters are absent, so a formula keeps them.
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
    "Σ": "S",
    "Δ": "D",
    "Π": "P",
}
# Same shapes used inside Russian words: β соответствии → в соответствии.
_GREEK_LOOKALIKE_TO_CYR = {
    "β": "в",
    "η": "и",
    "σ": "с",
    "ς": "с",
    "ρ": "р",
    "τ": "т",
    "μ": "м",
    "α": "а",
    "ε": "е",
    "ο": "о",
    "κ": "к",
}
_LOOKALIKE_CHARS = set(_GREEK_LOOKALIKE_TO_LAT) | set(_GREEK_LOOKALIKE_TO_CYR)
_TRANSPARENT = set(" \t\n\r\u00a0.,;:!?\"'«»“”„()[]{}…·-/\\'’`´")
_ABBREV_PUNCT = set(".,")
_CODE_JOIN = set("-_/")
_MATH = set("=<>≤≥≠≈±∓×÷·∙*^_~∼≅≡∝∞∫∑∏√∂∇∈∉⊂⊃∧∨¬|∥°′″")


def _is_latin_letter(ch: str) -> bool:
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def _is_cyrillic_letter(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04FF"


def _is_isolated_latin(chars: list[str], index: int) -> bool:
    """A lone ``O`` in ``(O)`` is a marker, not the script of the sentence."""
    ch = chars[index]
    if not _is_latin_letter(ch):
        return False
    prev = chars[index - 1] if index > 0 else ""
    nxt = chars[index + 1] if index + 1 < len(chars) else ""
    for side in (prev, nxt):
        if side and (_is_latin_letter(side) or side.isdigit()):
            return False
    return True


def _lookalike_side(chars: list[str], index: int, step: int) -> tuple[str, int]:
    """Nearest real script. Spaces, digits, punctuation and lookalikes are transparent."""
    j = index + step
    n = len(chars)
    dist = 0
    while 0 <= j < n:
        ch = chars[j]
        dist += 1
        if ch in _LOOKALIKE_CHARS or ch.isspace() or ch.isdigit() or ch in _TRANSPARENT:
            j += step
            continue
        if _is_isolated_latin(chars, j):
            j += step
            continue
        if _is_latin_letter(ch):
            return "latin", dist
        if _is_cyrillic_letter(ch):
            return "cyrillic", dist
        if ch in _MATH or unicodedata.category(ch) == "Sm":
            return "block", dist
        if "GREEK" in unicodedata.name(ch, ""):
            return "block", dist
        return "block", dist
    return "edge", dist


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


def _in_code_cluster(chars: list[str], index: int) -> bool:
    """Σ1000Δ-Α-… is a technical code: uppercase lookalikes mixed with digits."""
    lo = index
    while lo > 0 and (
        chars[lo - 1] in _LOOKALIKE_CHARS or chars[lo - 1].isdigit() or chars[lo - 1] in _CODE_JOIN
    ):
        lo -= 1
    hi = index
    n = len(chars)
    while hi + 1 < n and (
        chars[hi + 1] in _LOOKALIKE_CHARS or chars[hi + 1].isdigit() or chars[hi + 1] in _CODE_JOIN
    ):
        hi += 1
    has_digit = False
    has_upper = False
    for ch in chars[lo : hi + 1]:
        if ch.isdigit():
            has_digit = True
        if ch in _LOOKALIKE_CHARS and ch.isupper():
            has_upper = True
    return has_digit and has_upper


def _map_lookalike(ch: str, left: str, right: str, left_dist: int, right_dist: int, chars: list[str], index: int) -> str | None:
    if left == "block" or right == "block":
        return None
    if _in_code_cluster(chars, index):
        return _GREEK_LOOKALIKE_TO_LAT.get(ch)
    saw_lat = left == "latin" or right == "latin"
    saw_cyr = left == "cyrillic" or right == "cyrillic"
    if saw_cyr and not saw_lat:
        return _GREEK_LOOKALIKE_TO_CYR.get(ch)
    if saw_lat and not saw_cyr:
        return _GREEK_LOOKALIKE_TO_LAT.get(ch)
    if saw_lat and saw_cyr:
        if left_dist < right_dist:
            prefer_cyr = left == "cyrillic"
        elif right_dist < left_dist:
            prefer_cyr = right == "cyrillic"
        else:
            prefer_cyr = ch in _GREEK_LOOKALIKE_TO_CYR
        if prefer_cyr and ch in _GREEK_LOOKALIKE_TO_CYR:
            return _GREEK_LOOKALIKE_TO_CYR[ch]
        if not prefer_cyr and ch in _GREEK_LOOKALIKE_TO_LAT:
            return _GREEK_LOOKALIKE_TO_LAT[ch]
        # η and β have no Latin twin. A nearer English label must not keep them Greek
        # when the other side is a Russian word.
        if saw_cyr and ch in _GREEK_LOOKALIKE_TO_CYR:
            return _GREEK_LOOKALIKE_TO_CYR[ch]
        return _GREEK_LOOKALIKE_TO_LAT.get(ch)
    if _in_abbrev_cluster(chars, index):
        return _GREEK_LOOKALIKE_TO_LAT.get(ch)
    return None


def normalize_greek_lookalikes(text: str) -> str:
    """Map Greek lookalikes to Latin or Cyrillic. Formula letters stay Greek.

    Cyrillic next to the letter picks the Cyrillic twin (``β соответствии`` →
    ``в соответствии``). Latin, or a digit code such as ``Σ1000Δ``, picks Latin
    (``ARP4754Α``, ``ε.γ.``). ``σ = μ + λ`` is unchanged.
    """
    if not text or not any(ch in _LOOKALIKE_CHARS for ch in text):
        return text
    chars = list(text)
    out: list[str] = []
    for i, ch in enumerate(chars):
        if ch not in _LOOKALIKE_CHARS:
            out.append(ch)
            continue
        left, left_dist = _lookalike_side(chars, i, -1)
        right, right_dist = _lookalike_side(chars, i, 1)
        mapped = _map_lookalike(ch, left, right, left_dist, right_dist, chars, i)
        out.append(mapped if mapped else ch)
    return "".join(out)


def latinize_greek_lookalikes(text: str) -> str:
    """Backward-compatible name. Cyrillic context is mapped to Cyrillic too."""
    return normalize_greek_lookalikes(text)


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
