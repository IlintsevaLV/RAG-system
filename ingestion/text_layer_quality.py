"""Detect garbage / OCR-like embedded PDF text layers (lexical quality)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

LangHint = Literal["en", "ru", "mixed", "unknown"]

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-/]{1,}|[A-Za-zА-Яа-яЁё]{2,}")
_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")

# Technical designations — do not treat as OOV garbage
_TECHNICAL = re.compile(
    r"""(?ix)
    ^(
        mil(-[a-z0-9]+)+
      | amcp(-[0-9]+)*
      | astm[a-z0-9\-]*
      | gost[a-z0-9\-]*
      | iso[0-9\-]+
      | iec[0-9\-]+
      | kt-?[0-9a-z\-]+
      | ap-?[0-9a-z\-]+
      | faa[a-z0-9\-]*
      | nasa[a-z0-9\-]*
      | doi[:0-9.]+
      | [a-z]?[0-9]+([.\-][0-9a-z]+)+
      | [0-9]+([.,][0-9]+)?(w|kw|mhz|hz|v|a|mm|cm|m|kg|lb|in|ft|psi|°?[cf])?
      | w/in\.?
      | ref\.?
    )$
    """
)

# Tokens that look like broken OCR inside an otherwise alphabetic word
_GARBAGE_SHAPE = re.compile(
    r"""(?x)
    [A-Za-z]{2,}[0-9]{2,}[A-Za-z]+   # dM11be style mid-digit
  | [A-Za-z]*[~^*=_]{1,}[A-Za-z]*    # tgm~p=c
  | (.)\1{3,}                        # aaaa
    """
)


class LexicalQualityResult(BaseModel):
    lexical_quality: float
    garbage_score: float
    dict_hit: float
    typo_ratio: float
    oov_hard: float
    alpha_token_count: int
    lang_hint: LangHint = "unknown"
    garbage_veto: bool = False
    sample_bad_tokens: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _wordfreq():
    try:
        from wordfreq import zipf_frequency

        return zipf_frequency
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("pip install wordfreq") from exc


def detect_lang_hint(text: str) -> LangHint:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "unknown"
    cyr = sum(1 for ch in letters if _CYR.match(ch))
    lat = sum(1 for ch in letters if _LAT.match(ch))
    total = cyr + lat
    if total == 0:
        return "unknown"
    cyr_share = cyr / total
    if cyr_share >= 0.6:
        return "ru"
    if cyr_share <= 0.2:
        return "en"
    return "mixed"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _is_technical(token: str) -> bool:
    return bool(_TECHNICAL.match(token))


def _zipf(token: str, lang: str) -> float:
    zipf_frequency = _wordfreq()
    # wordfreq wants lowercase; keep original for Cyrillic lower
    return float(zipf_frequency(token.lower(), lang))


def _best_zipf(token: str, lang_hint: LangHint) -> float:
    if lang_hint == "ru":
        langs = ("ru", "en")
    elif lang_hint == "en":
        langs = ("en", "ru")
    else:
        langs = ("en", "ru")
    return max(_zipf(token, lg) for lg in langs)


def _is_typo_shaped(token: str) -> bool:
    """Cheap OCR-typo shape heuristics (no full spellchecker)."""
    if _GARBAGE_SHAPE.search(token):
        return True
    # vowel scarcity in Latin tokens length>=5
    if _LAT.search(token) and not _CYR.search(token) and len(token) >= 5:
        vowels = sum(1 for ch in token.lower() if ch in "aeiouy")
        if vowels == 0:
            return True
        if vowels / len(token) < 0.12 and len(token) >= 7:
            return True
    if _CYR.search(token) and len(token) >= 5:
        vowels = sum(1 for ch in token.lower() if ch in "аеёиоуыэюяaeiouy")
        if vowels == 0:
            return True
    return False


def lexical_quality(
    text: str,
    *,
    min_tokens_for_veto: int = 30,
    veto_threshold: float = 0.45,
    dict_zipf: float = 3.0,
    weak_zipf: float = 1.5,
) -> LexicalQualityResult:
    """Score whether embedded text looks like real language vs OCR garbage."""
    notes: list[str] = []
    lang = detect_lang_hint(text)
    tokens = tokenize(text)
    # Prefer alphabetic-heavy tokens for dictionary checks
    alpha_tokens = [t for t in tokens if sum(ch.isalpha() for ch in t) >= 2]
    if not alpha_tokens:
        return LexicalQualityResult(
            lexical_quality=0.0,
            garbage_score=1.0,
            dict_hit=0.0,
            typo_ratio=0.0,
            oov_hard=0.0,
            alpha_token_count=0,
            lang_hint=lang,
            garbage_veto=False,
            notes=["no alpha tokens"],
        )

    dict_hits = 0
    weak_hits = 0
    oov = 0
    typos = 0
    technical = 0
    bad_samples: list[str] = []

    for tok in alpha_tokens:
        if _is_technical(tok):
            technical += 1
            dict_hits += 1  # count as acceptable
            continue
        z = _best_zipf(tok, lang)
        if z >= dict_zipf:
            dict_hits += 1
        elif z >= weak_zipf:
            weak_hits += 1
            if _is_typo_shaped(tok):
                typos += 1
                if len(bad_samples) < 12:
                    bad_samples.append(tok)
        else:
            oov += 1
            if _is_typo_shaped(tok) or z == 0.0:
                typos += 1
            if len(bad_samples) < 12:
                bad_samples.append(tok)

    n = len(alpha_tokens)
    dict_hit = dict_hits / n
    typo_ratio = typos / n
    oov_hard = oov / n
    # weak hits partially count
    dict_hit_soft = (dict_hits + 0.4 * weak_hits) / n

    quality = 0.55 * dict_hit_soft + 0.25 * (1.0 - typo_ratio) + 0.20 * (1.0 - oov_hard)
    quality = float(max(0.0, min(1.0, quality)))
    garbage = 1.0 - quality

    veto = bool(n >= min_tokens_for_veto and quality < veto_threshold)
    if technical:
        notes.append(f"technical_tokens={technical}")
    if veto:
        notes.append(
            f"garbage_veto lexical_quality={quality:.3f} < {veto_threshold} (n={n})"
        )

    return LexicalQualityResult(
        lexical_quality=quality,
        garbage_score=garbage,
        dict_hit=round(dict_hit, 4),
        typo_ratio=round(typo_ratio, 4),
        oov_hard=round(oov_hard, 4),
        alpha_token_count=n,
        lang_hint=lang,
        garbage_veto=veto,
        sample_bad_tokens=bad_samples,
        notes=notes,
    )
