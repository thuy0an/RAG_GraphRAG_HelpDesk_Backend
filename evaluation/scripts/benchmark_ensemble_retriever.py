"""Benchmark BM25Retriever + EnsembleRetriever voi weight tuning.

Script co the chay tren 1 file mau hoac ingest ca thu muc tai lieu de benchmark
BM25/vector/ensemble tren corpus that.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, List, Tuple

from fastapi import UploadFile

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process
from src.SharedKernel.config.LLMConfig import EmbeddingFactory
from src.SharedKernel.utils.yamlenv import load_env_yaml


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
DEFAULT_DATASET = ROOT / "evaluation" / "dataset" / "benchmark_questions.json"
DEFAULT_DOC = REPO_ROOT / "smoke_test_assets" / "graph_smoke_test.pdf"
DEFAULT_DOCUMENTS_DIR = REPO_ROOT / "AI_HelpDesk_Backend" / "tailieu"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "ensemble_tuning.csv"
DEFAULT_BEST = ROOT / "evaluation" / "results" / "ensemble_best.json"


def _norm(text: str) -> str:
    raw = (text or "").strip().lower()
    no_diacritics = "".join(ch for ch in unicodedata.normalize("NFD", raw) if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", no_diacritics)
    return re.sub(r"\s+", " ", cleaned).strip()


def _norm_keep_diacritics(text: str) -> str:
    raw = (text or "").strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def _point_recall(text: str, points: List[str]) -> float:
    if not points:
        return 0.0
    text_no_diac = _norm(text)
    text_with_diac = _norm_keep_diacritics(text)
    hits = 0
    for p in points:
        p_no_diac = _norm(p)
        p_with_diac = _norm_keep_diacritics(p)
        if not p_no_diac:
            continue
        if p_no_diac in text_no_diac:
            hits += 1
            continue
        if p_with_diac and p_with_diac in text_with_diac:
            hits += 1
            continue
        # token overlap fallback
        p_tokens = set(p_no_diac.split())
        t_tokens = set(text_no_diac.split())
        if p_tokens and len(p_tokens & t_tokens) / len(p_tokens) >= 0.5:
            hits += 1
    return hits / len(points)


def _open_upload_file(path: Path) -> UploadFile:
    return UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()))


def _load_dataset(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("questions", []))


def _list_documents(documents_dir: Path) -> List[Path]:
    if not documents_dir.exists():
        return []
    exts = {".pdf", ".doc", ".docx", ".txt", ".md", ".html"}
    return sorted([p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


async def _load_chunks(doc_path: Path):
    loader = Loader()
    process = Process()
    upload = _open_upload_file(doc_path)
    docs = await loader.load_file(upload)
    split = await process.split_PaC(docs)
    children = split.get("children", [])
    return children


async def _load_chunks_from_documents(documents_dir: Path):
    chunks = []
    for doc_path in _list_documents(documents_dir):
        chunks.extend(await _load_chunks(doc_path))
    return chunks


def _build_retrievers(chunks):
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS
    except Exception as exc:
        raise RuntimeError(f"Retriever dependencies missing: {exc}")

    config = load_env_yaml()
    embedding = EmbeddingFactory.create(config.llm.provider)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    vector_db = FAISS.from_documents(chunks, embedding)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    return bm25, vector_retriever


def _doc_key(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source = str(metadata.get("source", ""))
    page = str(metadata.get("page_number", metadata.get("page", "")))
    content = getattr(doc, "page_content", "")
    return f"{source}|{page}|{hash(content)}"


def _weighted_rrf(bm25_docs, vector_docs, weight_bm25: float, k: int = 60):
    scores = defaultdict(float)
    doc_map = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        key = _doc_key(doc)
        scores[key] += weight_bm25 * (1.0 / (k + rank))
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)
        scores[key] += (1.0 - weight_bm25) * (1.0 / (k + rank))
        doc_map[key] = doc

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in ranked_keys]


def _run_single_retriever(retriever, query: str) -> Tuple[str, float]:
    start = perf_counter()
    docs = retriever.invoke(query)
    elapsed = perf_counter() - start
    context = "\n".join(getattr(doc, "page_content", "") for doc in docs)
    return context, elapsed


def _run_ensemble(bm25, vector_retriever, query: str, weight_bm25: float) -> Tuple[str, float]:
    start = perf_counter()
    bm25_docs = bm25.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    docs = _weighted_rrf(bm25_docs, vector_docs, weight_bm25)
    elapsed = perf_counter() - start
    context = "\n".join(getattr(doc, "page_content", "") for doc in docs[:5])
    return context, elapsed


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["strategy", "weight_bm25", "runs", "avg_latency_s", "avg_point_recall"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_best(path: Path, rows: List[Dict[str, str]]) -> None:
    best = max(rows, key=lambda r: float(r["avg_point_recall"])) if rows else None
    payload = {"best": best, "all": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _async_main(args: argparse.Namespace) -> int:
    dataset = _load_dataset(Path(args.dataset).resolve())
    if args.documents_dir:
        chunks = await _load_chunks_from_documents(Path(args.documents_dir).resolve())
    else:
        chunks = await _load_chunks(Path(args.document).resolve())
    bm25, vector_retriever = _build_retrievers(chunks)

    rows: List[Dict[str, str]] = []

    def evaluate_strategy(name: str, retriever, weight: str = ""):
        latencies = []
        recalls = []
        for item in dataset:
            query = str(item.get("question", ""))
            points = [str(x) for x in item.get("expected_answer_points", []) if x]
            context, elapsed = _run_single_retriever(retriever, query)
            latencies.append(elapsed)
            recalls.append(_point_recall(context, points))
        rows.append(
            {
                "strategy": name,
                "weight_bm25": weight,
                "runs": str(len(latencies)),
                "avg_latency_s": f"{mean(latencies):.4f}" if latencies else "0.0000",
                "avg_point_recall": f"{mean(recalls):.4f}" if recalls else "0.0000",
            }
        )

    evaluate_strategy("BM25", bm25)
    evaluate_strategy("Vector", vector_retriever)

    for weight in [float(x.strip()) for x in args.weights.split(",") if x.strip()]:
        latencies = []
        recalls = []
        for item in dataset:
            query = str(item.get("question", ""))
            points = [str(x) for x in item.get("expected_answer_points", []) if x]
            context, elapsed = _run_ensemble(bm25, vector_retriever, query, weight)
            latencies.append(elapsed)
            recalls.append(_point_recall(context, points))
        rows.append(
            {
                "strategy": "Ensemble",
                "weight_bm25": f"{weight:.2f}",
                "runs": str(len(latencies)),
                "avg_latency_s": f"{mean(latencies):.4f}" if latencies else "0.0000",
                "avg_point_recall": f"{mean(recalls):.4f}" if recalls else "0.0000",
            }
        )

    out_csv = Path(args.output).resolve()
    out_best = Path(args.best).resolve()
    _write_csv(out_csv, rows)
    _write_best(out_best, rows)
    print(f"ensemble_written: {out_csv} | {out_best}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BM25 + EnsembleRetriever with weight tuning")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--document", default=str(DEFAULT_DOC))
    parser.add_argument("--documents-dir", default=str(DEFAULT_DOCUMENTS_DIR), help="Folder with benchmark documents; if set, all supported files in the folder are loaded")
    parser.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--best", default=str(DEFAULT_BEST))
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
