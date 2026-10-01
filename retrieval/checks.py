"""Synthetic checks for admission, clause splits, and lexical search."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.text_norm import normalize_greek_lookalikes
from retrieval.clauses import AdmittedPage, TextChunk, chunk_document, heading_num, join_hyphens
from retrieval.engine import (
    BuildReport,
    SearchIndex,
    admit_document,
    build_chunks,
    load_index,
    search,
    tokenize,
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

    alpha = "ARP4754\u0391"
    sigma = "NASA\u2019\u03c3 STI"
    eg = "\u03b5.\u03b3."
    if normalize_greek_lookalikes(f"any {alpha} requirement") != "any ARP4754A requirement":
        _fail("Greek alpha inside ARP4754A was kept")
    if normalize_greek_lookalikes("\u03b1 key part") != "a key part":
        _fail("Greek alpha in a Latin phrase was kept")
    if normalize_greek_lookalikes(sigma) != "NASA\u2019s STI":
        _fail(f"sigma in NASA was kept: {normalize_greek_lookalikes(sigma)!r}")
    if normalize_greek_lookalikes(f"see {eg} above") != "see e.g. above":
        _fail("epsilon.gamma abbreviation was kept")
    formula = "\u03c3 = \u03bc + \u03bb"
    if normalize_greek_lookalikes(formula) != formula:
        _fail(f"formula was latinized: {normalize_greek_lookalikes(formula)!r}")
    if normalize_greek_lookalikes("\u03b1") != "\u03b1":
        _fail("isolated formula variable was latinized")
    cyr = "changeType \u03b7 (O); \u03b2 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0438\u0438 \u03c3 \u0413\u043b\u0430\u0432\u0430"
    got_cyr = normalize_greek_lookalikes(cyr)
    if "\u03b7" in got_cyr or "\u03b2" in got_cyr or "\u03c3" in got_cyr:
        _fail(f"Cyrillic context kept Greek: {got_cyr!r}")
    if "\u0432 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0438\u0438 \u0441 \u0413\u043b\u0430\u0432\u0430" not in got_cyr:
        _fail(f"Cyrillic twins missing: {got_cyr!r}")
    code = "\u03a31000\u0394-\u0391-08-02-0700-00\u0391-040\u0391-\u0391"
    if normalize_greek_lookalikes(code) != "S1000D-A-08-02-0700-00A-040A-A":
        _fail(f"S1000D code: {normalize_greek_lookalikes(code)!r}")
    if normalize_greek_lookalikes("\u03a31000DR-UACRU-01000-00") != "S1000DR-UACRU-01000-00":
        _fail("capital sigma in S1000DR was kept")
    if tokenize(alpha) != ["arp4754a"]:
        _fail(f"identifier token split: {tokenize(alpha)}")

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

    admitted_arp = [
        AdmittedPage(
            doc_id="APR4754A application",
            source_file="APR4754A application.pdf",
            page=167,
            page_class="A",
            text=(
                "24 Please describe any ARP4754\u0391 requirement validation issues encountered "
                "during the review of this system."
            ),
        ),
        AdmittedPage(
            doc_id="APR4754A application",
            source_file="APR4754A application.pdf",
            page=193,
            page_class="A",
            text=(
                "52 Please discuss in general terms any current or future ARP4754\u0391 applications "
                "that the applicant expects to file."
            ),
        ),
    ]
    arp_chunks = chunk_document(admitted_arp)
    if any("\u0391" in c.text or "\u03b1" in c.text for c in arp_chunks):
        _fail("indexed chunk still contains Greek alpha")
    arp_index = SearchIndex.from_chunks(arp_chunks)
    arp_hits = search(arp_index, "ARP4754A requirement validation issues encountered", top_k=1)
    if not arp_hits or arp_hits[0].chunk.clause_id != "24" or arp_hits[0].chunk.page_start != 167:
        _fail(f"Latin ARP4754A query missed the Greek-layer chunk: {[(h.chunk.clause_id, h.chunk.page_start) for h in arp_hits]}")
    other = search(arp_index, "future ARP4754A applications", top_k=1)
    if not other or other[0].chunk.page_start != 193:
        _fail("second questionnaire item was not separated by its identifier")

    phrase = "Please describe any ARP4754A requirement validation issues encountered"
    short = TextChunk(
        chunk_id="short",
        doc_id="APR4754A application",
        source_file="APR4754A application.pdf",
        page_start=167,
        page_end=167,
        page_class="A",
        clause_id="24",
        part=1,
        parts=1,
        text="24 " + phrase + " during the review of this system.",
    )
    long = TextChunk(
        chunk_id="long",
        doc_id="APR4754A application",
        source_file="APR4754A application.pdf",
        page_start=139,
        page_end=140,
        page_class="A",
        clause_id="178",
        part=1,
        parts=32,
        text=(phrase + "\n") * 40,
    )
    ranked = search(SearchIndex.from_chunks([long, short]), phrase, top_k=1)
    if not ranked or ranked[0].chunk.clause_id != "24":
        _fail(f"long chunk outranked the short exact clause: {[h.chunk.clause_id for h in ranked]}")

    hydro = TextChunk(
        chunk_id="hydro",
        doc_id="НЛГ-27",
        source_file="НЛГ-27.pdf",
        page_start=83,
        page_end=83,
        page_class="A",
        clause_id="27.1435",
        part=1,
        parts=1,
        text="27.1435. Гидравлические системы. Конструкция должна выдерживать нагрузки.",
    )
    other = TextChunk(
        chunk_id="other",
        doc_id="НЛГ-27",
        source_file="НЛГ-27.pdf",
        page_start=10,
        page_end=10,
        page_class="A",
        clause_id="9.1",
        part=1,
        parts=1,
        text="Обучение персонала проводится по отдельной программе аэродрома.",
    )
    para = SearchIndex.from_chunks(
        [hydro, other],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        embedding_model="test",
    )
    paraphrased = search(
        para,
        "требования к прочности гидросистем",
        mode="hybrid",
        query_vector=[1.0, 0.0],
        top_k=1,
    )
    if not paraphrased or paraphrased[0].chunk.clause_id != "27.1435":
        _fail("hybrid missed the hydraulic clause on a paraphrase")
    lexical_only = search(para, "требования к прочности гидросистем", mode="lexical", top_k=1)
    if lexical_only and lexical_only[0].chunk.clause_id == "27.1435":
        _fail("lexical unexpectedly matched the paraphrase; the test no longer separates the modes")
    direct = search(
        para,
        "гидравлические системы 27.1435",
        mode="hybrid",
        query_vector=[0.0, 1.0],
        top_k=1,
    )
    if not direct or direct[0].chunk.clause_id != "27.1435":
        _fail(f"dense vector pulled the direct hit off rank 1: {[h.chunk.clause_id for h in direct]}")

    import retrieval.engine as engine

    saved = engine.embed_passages
    calls: list[int] = []

    def _fake_embed(texts: list[str], model_name: str, *, batch_size: int = 32) -> list[list[float]]:
        calls.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]

    engine.embed_passages = _fake_embed
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = BuildReport()
            write_index([hydro], report, out, embedding_model="intfloat/multilingual-e5-small")
            if calls != [1] or not report.dense:
                _fail(f"first dense build did not embed the chunk: {calls} {report.notes}")
            again_report = BuildReport()
            write_index([hydro], again_report, out, embedding_model="intfloat/multilingual-e5-small")
            if calls != [1]:
                _fail(f"unchanged chunk was embedded again: {calls}")
            loaded = load_index(out)
            if not loaded.vectors or loaded.embedding_model != "intfloat/multilingual-e5-small":
                _fail("dense.npy was not reloaded")
    finally:
        engine.embed_passages = saved

    print("check: ok")
    return 0
