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


# Greek letters that carry information. Uppercase Α Β Γ Ε Ζ Η Ι Κ Μ Ν Ο Π Ρ Τ Υ Φ Χ and
# omicron look like Latin/Cyrillic letters: the Greek head emits them for Russian text.
_DISTINCTIVE_GREEK = set("αβγδεζηθικλμνξπρσςτυφχψωΔΘΛΞΣΨΩ")
_GREEK_RUN_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")
_LETTER_WORD_RE = re.compile(r"[^\W\d_]+")
_CYR_WORD_RE = re.compile(r"[\u0400-\u04FF]{3,}")
# In running (non-formula) text Greek letters come alone or in pairs (α, ωR);
# a longer run means the Greek head misread Cyrillic.
MAX_GREEK_RUN_IN_TEXT = 2
_MIXED_WORD_MIN_LEN = 4

_SKELETON = str.maketrans(
    {
        **dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "a6brdeex3uuknmhonpcmyfxu4wwbbbeor")),
        **dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", "a6brdeex3uuknmhonpctyfxu4wwbbbeor")),
        **dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω", "abydezhoiknmvxonpccmyfxyw")),
        **dict(zip("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ", "abrdezhoiknmnxonpctyfxyw")),
    }
)
# Lone Cyrillic letters that are not Russian words or abbreviations (п., ч, г, м…):
# the Cyrillic head's reading of ω and φ.
_CYR_FOR_GREEK = frozenset("шф")
_MATH_CONTEXT_RE = re.compile(r"[=<>±+×·^_]")
# Latin-head spellings of Cyrillic letters: Ы -> bI/LI, Л -> JI/J, Ю -> IO.
_LATIN_DIGRAPHS = (("li", "b"), ("bi", "b"), ("bl", "b"), ("ji", "n"), ("j", "n"), ("io", "o"))


def max_greek_run(text: str) -> int:
    return max((len(m) for m in _GREEK_RUN_RE.findall(text or "")), default=0)


def has_distinctive_greek(text: str) -> bool:
    return any(ch in _DISTINCTIVE_GREEK for ch in text or "")


_LOWER_NON_GREEK_RUN_RE = re.compile(r"[a-zа-яё]{3,}")


def has_mixed_script_word(text: str, min_len: int = _MIXED_WORD_MIN_LEN) -> bool:
    """`Kaτeropus`: Greek inside a Latin/Cyrillic word. Formula tokens such as `CNπR`
    have no lowercase Latin/Cyrillic run and are not counted."""
    for word in _LETTER_WORD_RE.findall(text or ""):
        if len(word) < min_len:
            continue
        cyr, lat, gre = script_shares(word)
        if gre > 0 and (cyr > 0 or lat > 0) and _LOWER_NON_GREEK_RUN_RE.search(word):
            return True
    return False


def is_garbled_greek(text: str) -> bool:
    """Greek-head reading that is really misread Cyrillic/Latin text."""
    if not has_greek(text):
        return False
    return (
        max_greek_run(text) > MAX_GREEK_RUN_IN_TEXT
        or has_mixed_script_word(text)
        or not has_distinctive_greek(text)
    )


def lookalike_skeleton(text: str) -> str:
    """Common form of Cyrillic/Greek/Latin lookalikes: `температуру` == `memnepamypy`."""
    s = (text or "").translate(_SKELETON).lower()
    for src, dst in _LATIN_DIGRAPHS:
        s = s.replace(src, dst)
    return "".join(ch for ch in s if ch.isalnum())


def is_lookalike_reading(a: str, b: str, threshold: float = 0.6) -> bool:
    """True when two head readings are the same glyphs in different scripts."""
    from difflib import SequenceMatcher

    sa, sb = lookalike_skeleton(a), lookalike_skeleton(b)
    if len(sa) < 2 or len(sb) < 2:
        return sa == sb and bool(sa)
    return SequenceMatcher(None, sa, sb, autojunk=False).ratio() >= threshold


def is_russian_reading(text: str, conf: float) -> bool:
    cyr, _lat, _gre = script_shares(text)
    return conf >= 0.5 and cyr >= 0.5 and bool(_CYR_WORD_RE.search(text or ""))


def splice_greek_tokens(base: str, greek: str) -> str:
    """Put isolated Greek symbols from the Greek head into a Cyrillic-head line.

    Only 1–2 letter Greek tokens (`α`, `ωR`, `λ1`) replace short base tokens. Russian
    words (`для`, `до`) are never replaced; lone Cyrillic letters (`и`, `п.`, `ч`) only
    next to an operator (`при а = 5°`), except `ш`/`ф` which are not words.
    """
    b_toks, g_toks = base.split(), greek.split()
    if not b_toks or len(b_toks) != len(g_toks):
        return base
    out: list[str] = []
    for i, (bt, gt) in enumerate(zip(b_toks, g_toks)):
        letters = _LETTER_WORD_RE.findall(gt)
        n_letters = sum(len(w) for w in letters)
        take = (
            bt != gt
            and has_distinctive_greek(gt)
            and max_greek_run(gt) <= MAX_GREEK_RUN_IN_TEXT
            and n_letters <= 3
            and sum(ch.isalpha() for ch in bt) <= 3
            and not has_mixed_script_word(gt)
            # a Latin letter read by the Cyrillic head is a real Latin variable
            and (script_shares(bt)[0] > 0 or not any(ch.isalpha() for ch in bt))
        )
        if take:
            core = bt.strip(".,;:()-—")
            cyr_letters = [ch for ch in core if "\u0400" <= ch <= "\u04ff"]
            letters_bt = [ch for ch in core if ch.isalpha()]
            if len(cyr_letters) >= 2 and len(cyr_letters) == len(letters_bt):
                take = False
            elif len(letters_bt) == 1 and cyr_letters and cyr_letters[0].lower() not in _CYR_FOR_GREEK:
                neighbours = " ".join(b_toks[max(0, i - 1) : i + 2])
                take = bool(_MATH_CONTEXT_RE.search(neighbours)) and "." not in bt
        out.append(gt if take else bt)
    return " ".join(out)


def choose_ocr_candidate(
    candidates: list[tuple[str, float, str]],
) -> tuple[str, float, str]:
    """Pick best (text, score, lang) among overlapping multi-head readings.

    A Russian reading of the Cyrillic head is kept against lookalike readings of the
    Latin/Greek heads (`memnepamypy`, `ΟΒΟΡΥΔΟΒΑΗΚΑ`); isolated Greek symbols are spliced
    into it. Otherwise, by content:
    1. Greek / formula symbols
    2. Long Latin/English prose
    3. Cyrillic Russian prose
    otherwise highest confidence.
    """
    if not candidates:
        return "", 0.0, "none"
    base = next(
        ((t, c, lang) for t, c, lang in candidates if lang == "cyrillic" and t), None
    )
    if base is not None and is_russian_reading(base[0], base[1]):
        base_len = len(base[0].replace(" ", ""))
        rest = [
            (t, c, lang)
            for t, c, lang in candidates
            if t
            and lang != "cyrillic"
            # a much shorter reading lost letters (`Крылов А.Н.` -> `K A.H.`)
            and len(t.replace(" ", "")) >= 0.8 * base_len
            and not is_lookalike_reading(base[0], t)
            and not (lang == "el" and is_garbled_greek(t))
        ]
        greek = next((t for t, _c, lang in candidates if lang == "el" and t), "")
        base_text = splice_greek_tokens(base[0], greek) if greek else base[0]
        if not rest:
            return base_text, base[1], "cyrillic"
        candidates = [(base_text, base[1], "cyrillic"), *rest]
    elif base is not None and script_shares(base[0])[0] >= 0.5:
        # `П.4.6.1` vs Latin `N.4.6.1`: same glyphs, the Cyrillic head knows the script
        candidates = [
            (t, c, lang)
            for t, c, lang in candidates
            if lang != "latin" or not is_lookalike_reading(base[0], t)
        ]

    scored: list[tuple[float, str, float, str]] = []
    for text, conf, lang in candidates:
        if not text:
            continue
        cyr, lat, gre = script_shares(text)
        bonus = 0.0
        longest = max((len(c[0]) for c in candidates if c[0]), default=0)
        if lang == "el" and is_garbled_greek(text):
            bonus -= 0.60
        elif lang == "el" and (gre > 0.05 or looks_like_formula_text(text)):
            bonus += 0.18 + 0.25 * gre
            # Lone Greek letter must not steal a Cyrillic/Latin prose box.
            if len(text) <= 3 and longest >= 8:
                bonus -= 0.40
            has_cyr_alt = any(
                is_russian_reading(t, s) for t, s, _l in candidates if t and t != text
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

