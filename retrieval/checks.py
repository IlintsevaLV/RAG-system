"""Synthetic checks for admission, clause splits, and lexical search."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from retrieval.clauses import chunk_document, heading_num, join_hyphens
from retrieval.engine import (
    BuildReport,
    SearchIndex,
    admit_document,
    build_chunks,
    load_index,
    search,
    write_index,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def run_checks() -> int:
    joined = join_hyphens("Каждая гидравлическая си-\nстема должна вы-\nдерживать давление.")
    if "система" not in joined or "выдерживать" not in joined or "-\n" in joined:
        _fail(f"hyphen join failed: {joined!r}")

    if heading_num("(a) Конструкция.", None) is not None:
        _fail("lettered subpoint became a clause")
    if heading_num("27.1435. Гидравлические системы", None) != "27.1435":
        _fail("regulation number was not a heading")
    if heading_num("13.5.8", "Меры безопасности") != "13.5.8":
        _fail("bare dotted number was not a heading")
    if heading_num("60 мин в штиле", None) is not None:
        _fail("lowercase continuation became a heading")
    if heading_num("1.", "Классификация комплектующих изделий") != "1":
        _fail("numbered item with title on the next line was missed")
    if heading_num("Глава 4 Приложение 6. Эксплуатация воздушных судов", None) != "4":
        _fail("chapter heading was not a clause")

    nlg = (
        "84\n"
        "27.1435. Гидравлические системы\n"
        "(a) Конструкция. Каждая гидравлическая си-\n"
        "стема должна быть сконструирована так, чтобы она вы-\n"
        "держивала нагрузку.\n"
        "(b) Испытания. Каждая гидравлическая система должна быть испытана.\n"
    )
    pages = [
        {
            "doc_id": "НЛГ-27",
            "page": 10,
            "page_class": "A",
            "status": "ok",
            "text": nlg,
            "provenance": {"low_confidence": False, "garbage_veto": False},
        },
        {
            "doc_id": "НЛГ-27",
            "page": 11,
            "page_class": "B",
            "status": "ok",
            "text": "Продолжение описания испытаний гидравлической системы без нового номера.",
            "provenance": {},
        },
        {
            "doc_id": "НЛГ-27",
            "page": 12,
            "page_class": "C",
            "status": "ok",
            "text": "27.9999. Этот текст класса C не должен попасть в индекс гидравлики.",
            "provenance": {},
        },
        {
            "doc_id": "НЛГ-27",
            "page": 13,
            "page_class": "A",
            "status": "suspicious",
            "text": "27.1500. Сомнительная страница не индексируется, хотя номер пункта есть.",
            "provenance": {},
        },
        {
            "doc_id": "НЛГ-27",
            "page": 14,
            "page_class": "A",
            "status": "ok",
            "text": "27.1600. Страница с мусорным слоем не индексируется вообще.",
            "provenance": {"low_confidence": True},
            "notes": ["low_confidence"],
        },
    ]
    toc = "\n".join(f"{i}. Раздел номер {i} " + "." * 12 + f" {i}" for i in range(1, 9))
    pages.append(
        {
            "doc_id": "НЛГ-27",
            "page": 2,
            "page_class": "A",
            "status": "ok",
            "text": toc,
            "provenance": {},
        }
    )
    report = BuildReport()
    admitted = admit_document("НЛГ-27", "NLG.pdf", pages, report)
    page_nums = [p.page for p in admitted]
    if page_nums != [10, 11]:
        _fail(f"admission filter failed: {page_nums} report={report.to_dict()}")

    chunks = chunk_document(admitted)
    by_clause = {c.clause_id: c for c in chunks}
    if "27.1435" not in by_clause:
        _fail(f"missing clause, got {list(by_clause)}")
    body = by_clause["27.1435"].text
    if "(a)" not in body or "(b)" not in body:
        _fail("subpoints fell out of the clause")
    if "система должна" not in body or "выдерживала" not in body:
        _fail(f"hyphen was not restored inside the clause: {body!r}")
    if by_clause["27.1435"].page_start != 10 or by_clause["27.1435"].page_end != 11:
        _fail(
            f"cross-page range {by_clause['27.1435'].page_start}-{by_clause['27.1435'].page_end}"
        )
    if any("27.9999" in c.text or "27.1500" in c.text or "27.1600" in c.text for c in chunks):
        _fail("filtered pages leaked into chunks")

    index = SearchIndex.from_chunks(chunks)
    hits = search(index, "пункт 27.1435 гидравлические системы", top_k=3)
    if not hits or hits[0].chunk.clause_id != "27.1435":
        _fail(f"clause query missed: {[(h.chunk.clause_id, h.score) for h in hits]}")
    if "класс C" in hits[0].chunk.text:
        _fail("answer text is not the admitted clause")
    missing = search(index, "квантовая телепортация марсианского двигателя", top_k=3)
    if missing:
        _fail("unrelated query returned a chunk")
    partial = search(index, "гидравлическая телепортация", top_k=3)
    if partial:
        _fail("one shared word was enough to return a chunk")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pages_dir = root / "pages"
        pages_dir.mkdir()
        (pages_dir / "НЛГ-27_pages.json").write_text(
            json.dumps(
                {"doc_id": "НЛГ-27", "source_path": "C:/raw/НЛГ-27.pdf", "pages": pages},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        built, built_report = build_chunks(pages_dir)
        if built_report.skipped_toc != 1 or built_report.skipped_class != 1:
            _fail(f"report counts {built_report.to_dict()}")
        if not any(c.clause_id == "27.1435" for c in built):
            _fail("build_chunks dropped the clause")
        out = root / "index"
        write_index(built, built_report, out)
        loaded = load_index(out)
        again = search(loaded, "гидравлические системы", top_k=1)
        if not again or again[0].chunk.clause_id != "27.1435":
            _fail("reloaded index missed the clause")

    print("check: ok")
    return 0
