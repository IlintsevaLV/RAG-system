"""Admit A/B pages, build a BM25 index, optionally add dense vectors, search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.text_norm import normalize_greek_lookalikes
from retrieval.clauses import AdmittedPage, TextChunk, chunk_document, is_toc_page

# Dotted clause numbers stay whole. Letter+digit codes (ARP4754A) stay whole too:
# a trailing Greek lookalike used to be cut off, so every such question collapsed to "arp".
# A bare number (24, 52) is kept so questionnaire items do not tie with each other.
_TOKEN = re.compile(
    r"\d+(?:\.\d+)+|[a-zа-яё][a-zа-яё0-9]*[a-zа-яё0-9]|\d{2,4}",
    re.IGNORECASE,
)
_CLAUSE_IN_QUERY = re.compile(r"\d+(?:\.\d+)+")
_STOP = frozenset(
    """
    и в во на по с со к ко о об от из за до для при или не ни но а же ли бы
    что как это той том эта эти тот этаже уже еще ещё его её их она они он мы вы ты
    the and of to for in on a an of or is be by as at from with this that
    """.split()
)


@dataclass
class BuildReport:
    files: int = 0
    pages_seen: int = 0
    admitted: int = 0
    chunks: int = 0
    docs: int = 0
    skipped_class: int = 0
    skipped_status: int = 0
    skipped_low_confidence: int = 0
    skipped_empty: int = 0
    skipped_toc: int = 0
    skipped_short: int = 0
    empty_json: int = 0
    dense: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "pages_seen": self.pages_seen,
            "admitted": self.admitted,
            "chunks": self.chunks,
            "docs": self.docs,
            "skipped_class": self.skipped_class,
            "skipped_status": self.skipped_status,
            "skipped_low_confidence": self.skipped_low_confidence,
            "skipped_empty": self.skipped_empty,
            "skipped_toc": self.skipped_toc,
            "skipped_short": self.skipped_short,
            "empty_json": self.empty_json,
            "dense": self.dense,
            "notes": self.notes,
        }


def folded_text(text: str) -> str:
    text = normalize_greek_lookalikes(text).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    folded = folded_text(text)
    return [tok for tok in _TOKEN.findall(folded) if tok not in _STOP]


def query_clause_ids(query: str) -> list[str]:
    return _CLAUSE_IN_QUERY.findall(query)


def _low_confidence(page: dict) -> bool:
    prov = page.get("provenance") or {}
    notes = page.get("notes") or []
    if prov.get("low_confidence") or prov.get("garbage_veto"):
        return True
    return "low_confidence" in notes


def admit_document(
    doc_id: str,
    source_path: str | None,
    pages: list[dict],
    report: BuildReport,
) -> list[AdmittedPage]:
    source_file = Path(source_path).name if source_path else doc_id
    admitted: list[AdmittedPage] = []
    if not pages:
        report.empty_json += 1
        return admitted
    for page in pages:
        report.pages_seen += 1
        if page.get("page_class") not in ("A", "B"):
            report.skipped_class += 1
            continue
        if page.get("status") != "ok":
            report.skipped_status += 1
            continue
        if _low_confidence(page):
            report.skipped_low_confidence += 1
            continue
        text = (page.get("text") or "").strip()
        if len(text) < 40:
            report.skipped_empty += 1
            continue
        if is_toc_page(text):
            report.skipped_toc += 1
            continue
        admitted.append(
            AdmittedPage(
                doc_id=page.get("doc_id") or doc_id,
                source_file=source_file,
                page=int(page["page"]),
                page_class=page["page_class"],
                text=text,
            )
        )
    report.admitted += len(admitted)
    return admitted


def load_admitted(pages_dir: Path, report: BuildReport) -> dict[str, list[AdmittedPage]]:
    grouped: dict[str, list[AdmittedPage]] = {}
    files = sorted(pages_dir.glob("*_pages.json"))
    report.files = len(files)
    if not files:
        report.notes.append(f"no *_pages.json in {pages_dir}")
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.notes.append(f"bad json {path.name}: {exc}")
            continue
        doc_id = data.get("doc_id") or path.name[: -len("_pages.json")]
        pages = admit_document(doc_id, data.get("source_path"), data.get("pages") or [], report)
        for page in pages:
            grouped.setdefault(page.doc_id, []).append(page)
    return grouped


class BM25Index:
    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.dl = [len(doc) for doc in docs]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for doc in docs:
            counts = Counter(doc)
            self.tf.append(dict(counts))
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + (self.n - n + 0.5) / (n + 0.5))

    def score(self, query: list[str], idx: int) -> float:
        if self.avgdl == 0:
            return 0.0
        tf = self.tf[idx]
        dl = self.dl[idx]
        total = 0.0
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
        for term in query:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self.idf(term)
            total += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
        return total

    def candidates(self, query: list[str]) -> set[int]:
        found: set[int] = set()
        wanted = set(query)
        for idx, tf in enumerate(self.tf):
            if wanted.intersection(tf):
                found.add(idx)
        return found


def clause_bonus(clause_id: str | None, asked: list[str]) -> float:
    if not clause_id or not asked:
        return 0.0
    bonus = 0.0
    for num in asked:
        if clause_id == num:
            bonus = max(bonus, 12.0)
        elif clause_id.startswith(num + "."):
            bonus = max(bonus, 4.0)
        elif num.startswith(clause_id + "."):
            bonus = max(bonus, 1.5)
    return bonus


@dataclass
class Hit:
    chunk: TextChunk
    score: float
    method: str

    def to_dict(self) -> dict:
        data = self.chunk.to_dict()
        data["score"] = round(self.score, 4)
        data["method"] = self.method
        return data


@dataclass
class SearchIndex:
    chunks: list[TextChunk]
    bm25: BM25Index
    tokens: list[list[str]]
    vectors: list[list[float]] | None = None
    embedding_model: str | None = None

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        *,
        vectors: list[list[float]] | None = None,
        embedding_model: str | None = None,
    ) -> SearchIndex:
        tokens = [tokenize(chunk.text) for chunk in chunks]
        return cls(
            chunks=chunks,
            bm25=BM25Index(tokens),
            tokens=tokens,
            vectors=vectors,
            embedding_model=embedding_model,
        )


def _query_covered(index: SearchIndex, idx: int, tokens: list[str], asked: list[str]) -> bool:
    """Keep a hit only when the chunk is the named пункт or covers most query words."""
    if clause_bonus(index.chunks[idx].clause_id, asked) > 0:
        return True
    unique = list(dict.fromkeys(tokens))
    if not unique:
        return False
    present = index.bm25.tf[idx]
    matched = sum(1 for tok in unique if present.get(tok))
    if len(unique) == 1:
        return matched == 1
    return matched >= 2 and matched / len(unique) >= 0.5


def _lexical_hits(index: SearchIndex, query: str) -> list[tuple[int, float]]:
    tokens = tokenize(query)
    asked = query_clause_ids(query)
    if not tokens and not asked:
        return []
    lexical: list[tuple[int, float]] = []
    for idx in index.bm25.candidates(tokens or asked):
        if not _query_covered(index, idx, tokens, asked):
            continue
        base = index.bm25.score(tokens, idx) if tokens else 0.0
        score = base + clause_bonus(index.chunks[idx].clause_id, asked)
        if score > 0:
            lexical.append((idx, score))
    phrase = folded_text(query) if len(tokens) >= 3 and len(folded_text(query)) >= 20 else ""

    def _rank(item: tuple[int, float]) -> tuple[int, int, float]:
        idx, score = item
        if not phrase:
            return (0, 0, score)
        text = folded_text(index.chunks[idx].text)
        if phrase in text:
            return (1, -len(text), score)
        return (0, 0, score)

    lexical.sort(key=_rank, reverse=True)
    return lexical


def _as_hits(index: SearchIndex, ranked: list[tuple[int, float]], method: str, top_k: int) -> list[Hit]:
    return [
        Hit(chunk=index.chunks[idx], score=score, method=method)
        for idx, score in ranked[:top_k]
    ]


def search(
    index: SearchIndex,
    query: str,
    *,
    top_k: int = 3,
    pool: int = 50,
    mode: str = "lexical",
    query_vector: list[float] | None = None,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
    rrf_k: int = 60,
) -> list[Hit]:
    """mode: lexical | dense | hybrid. Default stays BM25 even if vectors are loaded."""
    lexical = _lexical_hits(index, query)
    if mode == "lexical" or (mode == "hybrid" and not index.vectors and query_vector is None):
        return _as_hits(index, lexical, "lexical", top_k)

    dense_ranked = _dense_ranking(index, query, query_vector=query_vector)
    if mode == "dense":
        return _as_hits(index, dense_ranked, "dense", top_k)
    if not dense_ranked:
        return _as_hits(index, lexical, "lexical", top_k)

    asked = query_clause_ids(query)
    lex_rank = {idx: rank for rank, (idx, _) in enumerate(lexical[:pool], start=1)}
    dense_rank = {idx: rank for rank, (idx, _) in enumerate(dense_ranked[:pool], start=1)}
    fused: dict[int, float] = {}
    for idx in set(lex_rank) | set(dense_rank):
        score = 0.0
        if idx in lex_rank:
            score += w_bm25 / (rrf_k + lex_rank[idx])
        if idx in dense_rank:
            score += w_dense / (rrf_k + dense_rank[idx])
        score += clause_bonus(index.chunks[idx].clause_id, asked) / 12.0
        fused[idx] = score
    ordered = sorted(
        fused,
        key=lambda idx: (fused[idx], -lex_rank.get(idx, 10**6)),
        reverse=True,
    )
    hits: list[Hit] = []
    for idx in ordered:
        if idx in lex_rank and idx in dense_rank:
            method = "hybrid"
        elif idx in lex_rank:
            method = "lexical"
        else:
            method = "dense"
        hits.append(Hit(chunk=index.chunks[idx], score=fused[idx], method=method))
        if len(hits) >= top_k:
            break
    return hits


def _dense_ranking(
    index: SearchIndex,
    query: str,
    *,
    query_vector: list[float] | None = None,
) -> list[tuple[int, float]]:
    if not index.vectors and query_vector is None:
        return []
    vector = query_vector
    if vector is None:
        if not index.embedding_model:
            return []
        vector = embed_query(query, index.embedding_model)
    if not vector or not index.vectors:
        return []
    scored = [(idx, _dot(vector, doc_vec)) for idx, doc_vec in enumerate(index.vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _e5_prefix(model_name: str, kind: str) -> str:
    if "e5" in model_name.lower():
        return "query: " if kind == "query" else "passage: "
    return ""


def embed_passages(texts: list[str], model_name: str, *, batch_size: int = 32) -> list[list[float]]:
    model = _load_encoder(model_name)
    prefix = _e5_prefix(model_name, "passage")
    rows: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        vectors = model.encode(
            [prefix + text for text in batch],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        rows.extend([float(x) for x in row] for row in vectors)
        if total > batch_size:
            print(f"dense {min(start + batch_size, total)}/{total}", flush=True)
    return rows


def embed_query(query: str, model_name: str) -> list[float] | None:
    try:
        model = _load_encoder(model_name)
    except RuntimeError:
        return None
    prefix = _e5_prefix(model_name, "query")
    vector = model.encode(
        [prefix + query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    return [float(x) for x in vector]


_ENCODER = None
_ENCODER_NAME = ""


def _load_encoder(model_name: str):
    global _ENCODER, _ENCODER_NAME
    if _ENCODER is not None and _ENCODER_NAME == model_name:
        return _ENCODER
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed; dense search stays off"
        ) from exc
    _ENCODER = SentenceTransformer(model_name)
    _ENCODER_NAME = model_name
    return _ENCODER


def build_chunks(pages_dir: Path) -> tuple[list[TextChunk], BuildReport]:
    report = BuildReport()
    grouped = load_admitted(pages_dir, report)
    chunks: list[TextChunk] = []
    for doc_pages in grouped.values():
        chunks.extend(chunk_document(doc_pages))
    report.chunks = len(chunks)
    report.docs = len({chunk.doc_id for chunk in chunks})
    used_pages = {
        (chunk.doc_id, page)
        for chunk in chunks
        for page in range(chunk.page_start, chunk.page_end + 1)
    }
    admitted_pages = {
        (page.doc_id, page.page) for doc_pages in grouped.values() for page in doc_pages
    }
    report.skipped_short = len(admitted_pages - used_pages)
    return chunks, report


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def chunk_text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_dense_cache(out_dir: Path) -> dict[str, list[float]]:
    """Map chunk text hash → vector. A changed chunk is embedded again."""
    ids_path = out_dir / "dense_ids.json"
    npy_path = out_dir / "dense.npy"
    if not ids_path.exists() or not npy_path.exists():
        return {}
    try:
        import numpy as np

        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        arr = np.load(npy_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    cache: dict[str, list[float]] = {}
    if len(ids) != len(arr):
        return {}
    for row, vec in zip(ids, arr):
        sha = row.get("sha256")
        if sha:
            cache[sha] = [float(x) for x in vec]
    return cache


def _load_aligned_dense(out_dir: Path, chunks: list[TextChunk]) -> list[list[float]] | None:
    ids_path = out_dir / "dense_ids.json"
    npy_path = out_dir / "dense.npy"
    if not ids_path.exists() or not npy_path.exists():
        return None
    try:
        import numpy as np

        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        arr = np.load(npy_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if len(ids) != len(chunks) or len(arr) != len(chunks):
        return None
    for i, chunk in enumerate(chunks):
        if ids[i].get("chunk_id") != chunk.chunk_id:
            return None
        if ids[i].get("sha256") != chunk_text_sha(chunk.text):
            return None
    return [[float(x) for x in row] for row in arr]


def _write_dense(
    chunks: list[TextChunk],
    report: BuildReport,
    out_dir: Path,
    embedding_model: str,
) -> None:
    cache = _read_dense_cache(out_dir)
    missing = [i for i, chunk in enumerate(chunks) if chunk_text_sha(chunk.text) not in cache]
    fresh: list[list[float]] = []
    if missing:
        try:
            fresh = embed_passages([chunks[i].text for i in missing], embedding_model)
        except RuntimeError as exc:
            report.notes.append(f"dense skipped: {exc}")
            report.dense = False
            return
        if len(fresh) != len(missing):
            report.notes.append("dense skipped: encoder returned the wrong number of vectors")
            report.dense = False
            return
    elif chunks:
        report.notes.append("dense cache hit, model not loaded")
    rows: list[list[float]] = []
    fresh_at = 0
    ids: list[dict[str, str]] = []
    for chunk in chunks:
        sha = chunk_text_sha(chunk.text)
        if sha in cache:
            rows.append(cache[sha])
        else:
            rows.append(fresh[fresh_at])
            fresh_at += 1
        ids.append({"chunk_id": chunk.chunk_id, "sha256": sha})
    import numpy as np

    np.save(out_dir / "dense.npy", np.asarray(rows, dtype="float32"))
    (out_dir / "dense_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False),
        encoding="utf-8",
    )
    report.dense = True
    report.notes.append(f"dense embedded {len(missing)}, reused {len(chunks) - len(missing)}")


def write_index(
    chunks: list[TextChunk],
    report: BuildReport,
    out_dir: Path,
    *,
    embedding_model: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    old_meta: dict = {}
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_meta = {}
    chunk_path = out_dir / "chunks.jsonl"
    lines = [json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks]
    chunk_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    legacy = out_dir / "vectors.json"
    if legacy.exists():
        legacy.unlink()
    if embedding_model and chunks:
        _write_dense(chunks, report, out_dir, embedding_model)
    model_name = embedding_model if report.dense else None
    if model_name is None and _load_aligned_dense(out_dir, chunks):
        model_name = old_meta.get("embedding_model")
        report.dense = bool(model_name)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": _file_sha(chunk_path),
        "embedding_model": model_name,
        "report": report.to_dict(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index(out_dir: Path) -> SearchIndex:
    chunk_path = out_dir / "chunks.jsonl"
    if not chunk_path.exists():
        raise FileNotFoundError(f"index not found: {chunk_path}. Run: python -m retrieval build")
    chunks: list[TextChunk] = []
    raw = chunk_path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        if line.strip():
            chunks.append(TextChunk.from_dict(json.loads(line)))
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    vectors = _load_aligned_dense(out_dir, chunks)
    model = meta.get("embedding_model") if vectors else None
    return SearchIndex.from_chunks(chunks, vectors=vectors, embedding_model=model)


def probe_recall(index: SearchIndex, *, limit: int = 100, top_k: int = 5, mode: str = "lexical") -> dict:
    """Ask the index with a sentence taken from a chunk. The label is that chunk."""
    pool = [chunk for chunk in index.chunks if len(tokenize(chunk.text)) >= 6]
    checked = 0
    hits_ok = 0
    misses: list[str] = []
    step = max(1, len(pool) // limit) if pool else 1
    for chunk in pool[::step]:
        if checked >= limit:
            break
        query = ""
        for sentence in re.split(r"(?<=[\.!?])\s+", chunk.text):
            if len(tokenize(sentence)) >= 6:
                query = sentence.strip()
                break
        if not query:
            continue
        checked += 1
        found = search(index, query, top_k=top_k, mode=mode)
        ok = any(
            hit.chunk.doc_id == chunk.doc_id
            and hit.chunk.clause_id == chunk.clause_id
            and hit.chunk.page_start <= chunk.page_end
            and chunk.page_start <= hit.chunk.page_end
            for hit in found
        )
        if ok:
            hits_ok += 1
        elif len(misses) < 8:
            misses.append(f"{chunk.doc_id} п.{chunk.clause_id} стр.{chunk.page_start}: {query[:120]}")
    recall = (hits_ok / checked) if checked else 0.0
    return {"checked": checked, "hits": hits_ok, "recall_at_5": round(recall, 4), "misses": misses}


def _gold_match(hit: Hit, item: dict) -> bool:
    if item.get("doc_id") and hit.chunk.doc_id != item["doc_id"]:
        return False
    clause = item.get("clause_id")
    if clause and hit.chunk.clause_id != clause:
        return False
    page = item.get("page")
    if page is not None and not (hit.chunk.page_start <= int(page) <= hit.chunk.page_end):
        return False
    return True


def eval_gold(
    index: SearchIndex,
    gold_path: Path,
    *,
    mode: str = "hybrid",
    top_k: int = 5,
) -> dict:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    recalls = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    missed: list[str] = []
    for item in items:
        hits = search(index, item["query"], top_k=top_k, mode=mode)
        rank = 0
        for i, hit in enumerate(hits, start=1):
            if _gold_match(hit, item):
                rank = i
                break
        if rank:
            rr_sum += 1.0 / rank
            for k in recalls:
                if rank <= k:
                    recalls[k] += 1
        elif len(missed) < 8:
            missed.append(item["query"])
    n = len(items) or 1
    return {
        "n": len(items),
        "recall_at_1": round(recalls[1] / n, 4),
        "recall_at_3": round(recalls[3] / n, 4),
        "recall_at_5": round(recalls[5] / n, 4),
        "mrr": round(rr_sum / n, 4),
        "misses": missed,
    }


def format_hit(hit: Hit, rank: int) -> str:
    chunk = hit.chunk
    if chunk.page_start == chunk.page_end:
        pages = str(chunk.page_start)
    else:
        pages = f"{chunk.page_start}–{chunk.page_end}"
    clause = chunk.clause_id or "без номера пункта"
    part = ""
    if chunk.parts > 1:
        part = f"\nфрагмент: {chunk.part} из {chunk.parts}"
    return (
        f"{rank}. {chunk.doc_id}\n"
        f"файл: {chunk.source_file}\n"
        f"пункт: {clause}\n"
        f"страница: {pages}\n"
        f"класс страницы: {chunk.page_class}\n"
        f"совпадение: {hit.method}, оценка {hit.score:.3f}"
        f"{part}\n"
        f"\n{chunk.text}\n"
    )


def format_answer(query: str, hits: list[Hit], *, chunk_count: int) -> str:
    if not hits:
        return (
            "В корпусе не найдено.\n"
            f"Проиндексировано чанков: {chunk_count}."
        )
    body = "\n".join(format_hit(hit, i) for i, hit in enumerate(hits, start=1))
    return f"Запрос: {query}\nНайдено: {len(hits)}\n\n{body}"
