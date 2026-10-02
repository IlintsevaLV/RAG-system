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
    eval_gold,
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
        print(f"Плотные векторы: {model or 'cache'}")
    elif args.dense:
        print("dense skipped")
    else:
        print("Плотные векторы не строились. Поиск лексический.")
    for note in report.notes:
        print(note)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    query = args.query or sys.stdin.read()
    query = query.strip()
    if not query:
        print("Пустой запрос.")
        return 1
    index = load_index(_index_dir(args.index_dir))
    if args.mode in ("dense", "hybrid") and not index.vectors:
        print("Плотные векторы не найдены. Сначала: python -m retrieval build --dense")
        if args.mode == "dense":
            return 2
        print("Используется лексический поиск.")
    hits = search(
        index,
        query,
        top_k=args.top,
        mode=args.mode,
        w_bm25=args.w_bm25,
        w_dense=args.w_dense,
        rrf_k=args.rrf_k,
        min_score=args.min_score if args.mode == "lexical" else 0.0,
    )
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


def _cmd_eval(args: argparse.Namespace) -> int:
    gold = Path(args.gold)
    if not gold.exists():
        print(f"Golden нет: {gold}")
        print("Пока его нет, самопроверка: python -m retrieval probe")
        return 2
    index = load_index(_index_dir(args.index_dir))
    result = eval_gold(index, gold, mode=args.mode, top_k=args.top)
    print(
        f"n={result['n']} mode={args.mode} "
        f"recall@1={result['recall_at_1']} recall@3={result['recall_at_3']} "
        f"recall@5={result['recall_at_5']} MRR={result['mrr']}"
    )
    for miss in result["misses"]:
        print(f"  промах: {miss}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    index = load_index(_index_dir(args.index_dir))
    result = probe_recall(index, limit=args.limit, top_k=5, mode=args.mode)
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
    query.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    query.add_argument("--w-bm25", type=float, default=1.0)
    query.add_argument("--w-dense", type=float, default=1.0)
    query.add_argument("--rrf-k", type=int, default=60)
    query.add_argument(
        "--min-score",
        type=float,
        default=5.0,
        help="Лексический порог BM25. Ниже него — «не найдено». 0 отключает порог.",
    )
    query.set_defaults(func=_cmd_query)

    probe = sub.add_parser("probe", help="Самопроверка: предложение из чанка ищется обратно")
    probe.add_argument("--limit", type=int, default=100)
    probe.add_argument("--index-dir", default=None)
    probe.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    probe.set_defaults(func=_cmd_probe)

    ev = sub.add_parser("eval", help="recall@k и MRR по golden JSON")
    ev.add_argument("--gold", default="data/ir/gold/retrieval_v1.json")
    ev.add_argument("--index-dir", default=None)
    ev.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="hybrid")
    ev.add_argument("--top", type=int, default=5)
    ev.set_defaults(func=_cmd_eval)

    check = sub.add_parser("check", help="Проверки нарезки и поиска на синтетике")
    check.set_defaults(func=lambda _args: run_checks())

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
