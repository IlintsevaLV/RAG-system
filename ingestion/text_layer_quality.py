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

_GARBAGE_SHAPE = re.compile(
    r"""(?x)
    [A-Za-z]{2,}[0-9]{2,}[A-Za-z]+
  | [A-Za-z]*[~^*=_]{1,}[A-Za-z]*
  | (.)\1{3,}
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


# S1000D / XML-ish / code identifiers (false OOV on structured docs)
_STRUCTURED_ID = re.compile(
    r"""(?x)
    ^(
        [a-z]+([A-Z][a-z0-9]+)+     # camelCase: descrWire, sbConcurrentSbInfo
      | [A-Z]{2,}[A-Z0-9_\-]{2,}    # EXPON, SUMOLD, BOVERA
      | <.?/?[A-Za-z][\w\-:.]*>?    # residual tag crumbs
      | S1000D[R]?[\w\-]*
    )$
    """
)


def _is_technical(token: str) -> bool:
    if _TECHNICAL.match(token):
        return True
    if _STRUCTURED_ID.match(token):
        return True
    # strip angle brackets if present
    bare = token.strip("<>/")
    return bool(bare != token and (_TECHNICAL.match(bare) or _STRUCTURED_ID.match(bare)))


def _zipf(token: str, lang: str) -> float:
    zipf_frequency = _wordfreq()
    return float(zipf_frequency(token.lower(), lang))


def _best_zipf(token: str, lang_hint: LangHint) -> float:
    if lang_hint == "ru":
        langs = ("ru", "en")
    elif lang_hint == "en":
        langs = ("en", "ru")
    else:
        langs = ("en", "ru")
    # Strip dangling hyphen from PDF line-wrap: "impor-"
    base = token[:-1] if token.endswith("-") and len(token) > 3 else token
    return max(_zipf(base, lg) for lg in langs)


def _is_typo_shaped(token: str) -> bool:
    if _GARBAGE_SHAPE.search(token):
        return True
    letters = "".join(ch for ch in token if ch.isalpha())
    if len(letters) < 5:
        return False
    if _LAT.search(token) and not _CYR.search(token):
        vowels = sum(1 for ch in letters.lower() if ch in "aeiouy")
        if vowels == 0:
            return True
        if vowels / len(letters) < 0.12 and len(letters) >= 7:
            return True
    if _CYR.search(token):
        vowels = sum(1 for ch in letters.lower() if ch in "аеёиоуыэюяaeiouy")
        if vowels == 0:
            return True
    return False


def _alpha_len(token: str) -> int:
    return sum(ch.isalpha() for ch in token)


def lexical_quality(
    text: str,
    *,
    min_tokens_for_veto: int = 25,
    veto_threshold: float = 0.75,
    dict_zipf: float = 3.0,
    weak_zipf: float = 1.5,
    min_token_alpha: int = 5,
    oov_veto_ratio: float = 0.18,
) -> LexicalQualityResult:
    """Score embedded text: real language vs OCR-like garbage.

    Short function words (the/of/a) inflate scores on mixed garbage layers,
    so metrics are computed primarily on tokens with >= min_token_alpha letters.
    """
    notes: list[str] = []
    lang = detect_lang_hint(text)
    tokens = tokenize(text)
    alpha_tokens = [t for t in tokens if _alpha_len(t) >= 2]
    # Content tokens — ignore short stopword-like noise for garbage detection
    content_tokens = [t for t in alpha_tokens if _alpha_len(t) >= min_token_alpha]

    if not content_tokens:
        # Fall back to all alpha tokens if page has only short words
        content_tokens = alpha_tokens
        notes.append("fallback_to_short_tokens")

    if not content_tokens:
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

    for tok in content_tokens:
        if _is_technical(tok):
            technical += 1
            dict_hits += 1
            continue
        z = _best_zipf(tok, lang)
        if z >= dict_zipf:
            dict_hits += 1
        elif z >= weak_zipf:
            weak_hits += 1
            if _is_typo_shaped(tok) or tok.endswith("-"):
                typos += 1
                if len(bad_samples) < 12:
                    bad_samples.append(tok)
        else:
            oov += 1
            typos += 1  # hard OOV on long tokens ≈ OCR garbage signature
            if len(bad_samples) < 12:
                bad_samples.append(tok)

    n = len(content_tokens)
    dict_hit = dict_hits / n
    typo_ratio = typos / n
    oov_hard = oov / n
    dict_hit_soft = (dict_hits + 0.35 * weak_hits) / n

    quality = 0.60 * dict_hit_soft + 0.25 * (1.0 - oov_hard) + 0.15 * (1.0 - typo_ratio)
    quality = float(max(0.0, min(1.0, quality)))
    garbage = 1.0 - quality

    veto = False
    if n >= min_tokens_for_veto:
        if quality < veto_threshold:
            veto = True
            notes.append(
                f"garbage_veto lexical_quality={quality:.3f} < {veto_threshold} (n={n})"
            )
        elif oov_hard >= oov_veto_ratio:
            veto = True
            notes.append(
                f"garbage_veto oov_hard={oov_hard:.3f} >= {oov_veto_ratio} (n={n})"
            )

    if technical:
        notes.append(f"technical_tokens={technical}")
    notes.append(f"content_tokens_ge_{min_token_alpha}={n}")

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
