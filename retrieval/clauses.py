"""Split admitted page text into numbered clauses.

Lettered subpoints ``(a)`` / ``а)`` stay inside the numbered пункт.
A clause that runs onto the next admitted page keeps one citation
with a page range. A gap in page numbers starts a new chunk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from core.text_norm import normalize_greek_lookalikes

MAX_CHUNK_CHARS = 1800

_LEADER = re.compile(r"\.{4,}|…{2,}")
_PREFIX = (
    r"(?:глава|раздел|подраздел|пункт|статья|section|chapter|п)"
)
_HEADING = re.compile(
    rf"^(?:(?P<prefix>{_PREFIX})\.?\s+)?(?P<num>\d+(?:\.\d+)*)(?P<trail>\.)?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_LONE_ITEM = re.compile(r"^(\d{1,3})\.$")
_FURNITURE = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^страница\s+\d+\s+из\s+\d+$", re.IGNORECASE),
    re.compile(r"^стр\.?\s*[ivxlcdm\d]+$", re.IGNORECASE),
    re.compile(r"^\d+\s*[-–—]\s*\d+$"),
)
_LETTERS = re.compile(r"[A-Za-zА-Яа-яЁё]")


@dataclass
class AdmittedPage:
    doc_id: str
    source_file: str
    page: int
    page_class: str
    text: str


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page_start: int
    page_end: int
    page_class: str
    clause_id: str | None
    part: int
    parts: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TextChunk:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source_file=data.get("source_file") or data["doc_id"],
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            page_class=data.get("page_class") or "",
            clause_id=data.get("clause_id"),
            part=int(data.get("part") or 1),
            parts=int(data.get("parts") or 1),
            text=data.get("text") or "",
        )


def join_hyphens(text: str) -> str:
    """Restore words broken at a line end. The source PDF splits ``си- / стема``."""
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"([A-Za-zА-Яа-яЁё])[ \t]*-[ \t]*\n[ \t]*([A-Za-zА-Яа-яЁё])",
        r"\1\2",
        text,
    )
    return text


def is_toc_page(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    leaders = sum(1 for ln in lines if _LEADER.search(ln))
    return leaders >= 4 and leaders / len(lines) >= 0.4


def _is_furniture(line: str) -> bool:
    return any(pat.match(line) for pat in _FURNITURE)


def heading_num(line: str, next_line: str | None) -> str | None:
    """Return the clause number if ``line`` opens a пункт, else None."""
    raw = line.strip()
    lone = _LONE_ITEM.match(raw)
    if lone:
        if next_line and len(_LETTERS.findall(next_line)) >= 4:
            return lone.group(1)
        return None

    match = _HEADING.match(raw)
    if not match:
        return None
    if _LEADER.search(raw):
        return None
    num = match.group("num")
    title = (match.group("title") or "").strip()
    prefix = match.group("prefix")
    if title and re.fullmatch(r"\d{2,4}", title):
        return None
    if num.startswith("0"):
        return None
    has_dot = "." in num
    if not has_dot and not prefix and not title:
        return None
    if not has_dot and not prefix:
        try:
            if int(num) > 400:
                return None
        except ValueError:
            return None
        letters = _LETTERS.findall(title)
        if len(letters) < 4 or not title[:1].isupper():
            return None
    if has_dot and not title and not prefix:
        if any(len(part) > 4 for part in num.split(".")):
            return None
    return num


def _clean_lines(page: AdmittedPage) -> list[tuple[int, str, str]]:
    text = join_hyphens(page.text)
    rows: list[tuple[int, str, str]] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line or _is_furniture(line):
            continue
        rows.append((page.page, page.page_class, line))
    return rows


def _letter_count(text: str) -> int:
    return len(_LETTERS.findall(text))


def _pack_lines(
    lines: list[tuple[int, str, str]],
    *,
    limit: int,
) -> list[list[tuple[int, str, str]]]:
    parts: list[list[tuple[int, str, str]]] = []
    buf: list[tuple[int, str, str]] = []
    size = 0
    for row in lines:
        extra = len(row[2]) + (1 if buf else 0)
        if buf and size + extra > limit:
            parts.append(buf)
            buf = []
            size = 0
            extra = len(row[2])
        buf.append(row)
        size += extra
    if buf:
        parts.append(buf)
    return parts


def _chunk_from_part(
    *,
    doc_id: str,
    source_file: str,
    clause_id: str | None,
    heading: str | None,
    part_lines: list[tuple[int, str, str]],
    part: int,
    parts: int,
    seen_ids: set[str],
) -> TextChunk | None:
    body = "\n".join(row[2] for row in part_lines).strip()
    if part > 1 and heading and clause_id and not body.startswith(heading):
        body = f"{heading}\n{body}"
    body = normalize_greek_lookalikes(body)
    letters = _letter_count(body)
    if clause_id:
        if letters < 8:
            return None
    elif letters < 40:
        return None
    pages = [row[0] for row in part_lines]
    classes = {row[1] for row in part_lines}
    page_class = next(iter(classes)) if len(classes) == 1 else "AB"
    base = f"{doc_id}|{pages[0]}-{pages[-1]}|{clause_id or 'preamble'}|{part}"
    chunk_id = base
    n = 2
    while chunk_id in seen_ids:
        chunk_id = f"{base}#{n}"
        n += 1
    seen_ids.add(chunk_id)
    return TextChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        page_start=pages[0],
        page_end=pages[-1],
        page_class=page_class,
        clause_id=clause_id,
        part=part,
        parts=parts,
        text=body,
    )


def chunk_document(pages: list[AdmittedPage], *, limit: int = MAX_CHUNK_CHARS) -> list[TextChunk]:
    """Group consecutive admitted pages of one document into clause chunks."""
    if not pages:
        return []
    ordered = sorted(pages, key=lambda p: p.page)
    doc_id = ordered[0].doc_id
    source_file = ordered[0].source_file
    flat: list[tuple[int, str, str]] = []
    for page in ordered:
        flat.extend(_clean_lines(page))

    groups: list[tuple[str | None, list[tuple[int, str, str]]]] = []
    current_id: str | None = None
    current: list[tuple[int, str, str]] = []

    def flush() -> None:
        nonlocal current_id, current
        if current:
            groups.append((current_id, current))
        current_id = None
        current = []

    for i, row in enumerate(flat):
        nxt = flat[i + 1][2] if i + 1 < len(flat) else None
        if current and row[0] > current[-1][0] + 1:
            flush()
        num = heading_num(row[2], nxt)
        if num:
            flush()
            current_id = num
            current = [row]
            continue
        if not current:
            current_id = None
            current = [row]
        else:
            current.append(row)
    flush()

    seen: set[str] = set()
    chunks: list[TextChunk] = []
    for clause_id, lines in groups:
        packed = _pack_lines(lines, limit=limit)
        heading = lines[0][2] if clause_id else None
        built: list[TextChunk] = []
        for part_lines in packed:
            item = _chunk_from_part(
                doc_id=doc_id,
                source_file=source_file,
                clause_id=clause_id,
                heading=heading,
                part_lines=part_lines,
                part=len(built) + 1,
                parts=0,
                seen_ids=seen,
            )
            if item is not None:
                built.append(item)
        total = len(built)
        for item in built:
            item.parts = total
            chunks.append(item)
    return chunks
