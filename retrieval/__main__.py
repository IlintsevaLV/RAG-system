"""Lexical retrieval over classes A and B.

  python -m retrieval build
  python -m retrieval query "пункт 27.1435 гидравлические системы"
  python -m retrieval probe
  python -m retrieval check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from retrieval.checks import run_checks
from retrieval.engine import (
    build_chunks,
    format_answer,
    load_index,
    probe_recall,
    search,
    write_index,
)


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _pages_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return get_settings().ir_dir / "pages"


def _index_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return get_settings().cache_dir / "text_index"


def _cmd_build(args: argparse.Namespace) -> int:
    pages = _pages_dir(args.pages_dir)
    out = _index_dir(args.index_dir)
    chunks, report = build_chunks(pages)
    model = args.embedding_model if args.dense else None
    if args.dense and not model:
        model = get_settings().embedding_model
    write_index(chunks, report, out, embedding_model=model)
    print(f"Индекс: {out}")
    print(
        f"Файлов JSON: {report.files}, страниц в JSON: {report.pages_seen}, "
        f"допущено A/B: {report.admitted}, чанков: {report.chunks}, документов: {report.docs}"
    )
    print(
        "Отсеяно — другой класс: {skipped_class}, не ok: {skipped_status}, "
        "low_confidence: {skipped_low_confidence}, пустые: {skipped_empty}, "
        "оглавление: {skipped_toc}, без текста пункта: {skipped_short}, "
        "пустой JSON: {empty_json}".format(**report.to_dict())
    )
    if report.dense:
        print(f"Плотные векторы: {model}")
    else:
        print("Плотные векторы не строились. Поиск лексический.")
    for note in report.notes:
        print(note)
    if args.dense and not report.dense:
        return 2
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    query = args.query or sys.stdin.read()
    query = query.strip()
    if not query:
        print("Пустой запрос.")
        return 1
    index = load_index(_index_dir(args.index_dir))
    hits = search(index, query, top_k=args.top)
    if args.json:
        payload = {
            "query": query,
            "found": bool(hits),
            "chunk_count": len(index.chunks),
            "hits": [hit.to_dict() for hit in hits],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_answer(query, hits, chunk_count=len(index.chunks)))
    return 0 if hits else 3


def _cmd_probe(args: argparse.Namespace) -> int:
    index = load_index(_index_dir(args.index_dir))
    result = probe_recall(index, limit=args.limit, top_k=5)
    print(
        f"Пробы: {result['checked']}, попадания: {result['hits']}, "
        f"recall@5: {result['recall_at_5']}"
    )
    for miss in result["misses"]:
        print(f"  промах: {miss}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="Поиск по тексту классов A и B")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Собрать индекс по *_pages.json")
    build.add_argument("--pages-dir", default=None)
    build.add_argument("--index-dir", default=None)
    build.add_argument(
        "--dense",
        action="store_true",
        help="Добавить векторы embedding_model. Нужен sentence-transformers.",
    )
    build.add_argument("--embedding-model", default=None)
    build.set_defaults(func=_cmd_build)

    query = sub.add_parser("query", help="Вернуть чанки без генерации")
    query.add_argument("query", nargs="?", default=None)
    query.add_argument("--top", type=int, default=3)
    query.add_argument("--json", action="store_true")
    query.add_argument("--index-dir", default=None)
    query.set_defaults(func=_cmd_query)

    probe = sub.add_parser("probe", help="Самопроверка: предложение из чанка ищется обратно")
    probe.add_argument("--limit", type=int, default=100)
    probe.add_argument("--index-dir", default=None)
    probe.set_defaults(func=_cmd_probe)

    check = sub.add_parser("check", help="Проверки нарезки и поиска на синтетике")
    check.set_defaults(func=lambda _args: run_checks())

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
