"""Chay benchmark PaCRAG vs GraphRAG va xuat JSON/CSV.

Script nay co muc tieu tao du lieu co cau truc on dinh de chen vao bao cao.
No uu tien tinh lap lai hon la toi uu hieu nang.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

# Thêm thư mục gốc và src vào sys.path để import hoạt động dù chạy từ đâu
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.Features.LangChainAPI.LangChainFacade import LangChainFacade


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "evaluation" / "dataset" / "benchmark_questions.json"
DEFAULT_RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_DOCUMENTS_DIR = ROOT / "tailieu"


@dataclass
class BenchmarkRecord:
    question_id: str
    question_type: str
    difficulty: str
    question: str
    base_pipeline: str
    strategy: str
    pipeline: str
    reranking_enabled: bool
    retrieval_mode: str
    answer: str
    total_time_s: float
    answer_tokens: int
    word_count: int
    retrieved_chunk_count: int
    retrieved_source_count: int
    confidence_score: Optional[float]
    relevance_score: Optional[float]
    source_coverage: Optional[float]
    latency_breakdown_json: str
    metric_groups_json: str


def _load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _list_documents(documents_dir: Path) -> List[Path]:
    if not documents_dir.exists():
        return []
    exts = {".pdf", ".doc", ".docx", ".txt", ".md", ".html"}
    files = [p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _open_upload(path: Path) -> UploadFile:
    return UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()))


async def _prepare_corpus(facade: LangChainFacade, documents: List[Path]) -> List[str]:
    if not documents:
        return []

    source_filters: List[str] = []
    for doc in documents:
        source_name = doc.name
        source_filters.append(source_name)

        try:
            await facade.PaCRAG.clear_vector_store(source=source_name)
        except Exception:
            pass

        try:
            await facade.GraphRAG.delete(source_name)
        except Exception:
            pass

        pac_upload = _open_upload(doc)
        await facade.PaCRAG.index_with_metrics(pac_upload)

        graph_upload = _open_upload(doc)
        await facade.GraphRAG.ingest(graph_upload, source=source_name)

    return source_filters


async def _run_pac(
    facade: LangChainFacade,
    question: str,
    source_filters: Optional[List[str]] = None,
    enable_reranking: bool = False,
    top_k: int = 15,
) -> Dict[str, Any]:
    result = await facade.PaCRAG.retrieve_full(
        question,
        source_filters=source_filters,
        enable_reranking=enable_reranking,
    )
    return result if isinstance(result, dict) else {"answer": _normalize_text(result)}


async def _run_graph(
    facade: LangChainFacade,
    question: str,
    source_filters: Optional[List[str]] = None,
    enable_reranking: bool = False,
    top_k: int = 15,
) -> Dict[str, Any]:
    result = await facade.GraphRAG.retrieve_with_metrics(
        question,
        source_filters=source_filters,
        enable_reranking=enable_reranking,
    )
    return result if isinstance(result, dict) else {"answer": _normalize_text(result)}


def _strategy_label(base_pipeline: str, reranking_enabled: bool) -> str:
    return f"{base_pipeline}+rerank" if reranking_enabled else base_pipeline


def _retrieval_mode(base_pipeline: str) -> str:
    if base_pipeline == "PaCRAG":
        return "hybrid_bm25_vector"
    if base_pipeline == "GraphRAG":
        return "graph_vector"
    return "unknown"


def _extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_metrics = payload.get("retrieval_metrics") or {}
    graph_metrics = payload.get("graph_metrics") or {}
    generation_metrics = payload.get("generation_metrics") or {}
    system_metrics = payload.get("system_metrics") or {}
    return {
        "answer": _normalize_text(payload.get("answer")),
        "total_time_s": float(payload.get("time_total_s") or system_metrics.get("time_total_s") or 0.0),
        "answer_tokens": int(payload.get("answer_tokens") or system_metrics.get("answer_tokens") or 0),
        "word_count": int(payload.get("word_count") or system_metrics.get("word_count") or 0),
        "retrieved_chunk_count": int(payload.get("retrieved_chunk_count") or retrieval_metrics.get("retrieved_chunk_count") or 0),
        "retrieved_source_count": int(
            payload.get("retrieved_source_count")
            or retrieval_metrics.get("retrieved_source_count")
            or graph_metrics.get("source_count")
            or 0
        ),
        "confidence_score": payload.get("confidence_score") or generation_metrics.get("confidence_score"),
        "relevance_score": payload.get("relevance_score") or generation_metrics.get("answer_relevance_proxy") or generation_metrics.get("answer_relevancy"),
        "source_coverage": payload.get("source_coverage") or retrieval_metrics.get("source_coverage") or retrieval_metrics.get("context_recall"),
        "latency_breakdown": payload.get("latency_breakdown") or {},
        "metric_groups": payload.get("metric_groups") or {
            "retrieval_metrics": retrieval_metrics,
            "graph_metrics": graph_metrics,
            "generation_metrics": generation_metrics,
            "system_metrics": system_metrics,
        },
    }


async def _run_benchmark(
    dataset_path: Path,
    documents_dir: Optional[Path] = None,
    reranking_modes: Optional[List[bool]] = None,
    top_k: int = 15,
) -> List[BenchmarkRecord]:
    dataset = _load_dataset(dataset_path)
    questions = dataset.get("questions") or []
    facade = LangChainFacade()
    records: List[BenchmarkRecord] = []
    reranking_modes = reranking_modes or [False, True]

    source_filters: Optional[List[str]] = None
    if documents_dir:
        docs = _list_documents(documents_dir)
        if docs:
            source_filters = await _prepare_corpus(facade, docs)
            print(f"benchmark_corpus_sources: {len(source_filters)}")

    for item in questions:
        question_id = str(item.get("id") or "unknown")
        question = _normalize_text(item.get("question"))
        question_type = _normalize_text(item.get("type") or "unknown")
        difficulty = _normalize_text(item.get("difficulty") or "unknown")

        for reranking_enabled in reranking_modes:
            start = perf_counter()
            pac_payload = await _run_pac(
                facade,
                question,
                source_filters=source_filters,
                enable_reranking=reranking_enabled,
                top_k=top_k,
            )
            pac_elapsed = perf_counter() - start
            pac_metrics = _extract_metrics(pac_payload)
            records.append(
                BenchmarkRecord(
                    question_id=question_id,
                    question_type=question_type,
                    difficulty=difficulty,
                    question=question,
                    base_pipeline="PaCRAG",
                    strategy=_strategy_label("PaCRAG", reranking_enabled),
                    pipeline="PaCRAG",
                    reranking_enabled=reranking_enabled,
                    retrieval_mode=_retrieval_mode("PaCRAG"),
                    answer=pac_metrics["answer"],
                    total_time_s=round(pac_metrics["total_time_s"] or pac_elapsed, 4),
                    answer_tokens=pac_metrics["answer_tokens"],
                    word_count=pac_metrics["word_count"],
                    retrieved_chunk_count=pac_metrics["retrieved_chunk_count"],
                    retrieved_source_count=pac_metrics["retrieved_source_count"],
                    confidence_score=pac_metrics["confidence_score"],
                    relevance_score=pac_metrics["relevance_score"],
                    source_coverage=pac_metrics["source_coverage"],
                    latency_breakdown_json=_safe_json(pac_metrics["latency_breakdown"]),
                    metric_groups_json=_safe_json(pac_metrics["metric_groups"]),
                )
            )

            start = perf_counter()
            graph_payload = await _run_graph(
                facade,
                question,
                source_filters=source_filters,
                enable_reranking=reranking_enabled,
                top_k=top_k,
            )
            graph_elapsed = perf_counter() - start
            graph_metrics = _extract_metrics(graph_payload)
            records.append(
                BenchmarkRecord(
                    question_id=question_id,
                    question_type=question_type,
                    difficulty=difficulty,
                    question=question,
                    base_pipeline="GraphRAG",
                    strategy=_strategy_label("GraphRAG", reranking_enabled),
                    pipeline="GraphRAG",
                    reranking_enabled=reranking_enabled,
                    retrieval_mode=_retrieval_mode("GraphRAG"),
                    answer=graph_metrics["answer"],
                    total_time_s=round(graph_metrics["total_time_s"] or graph_elapsed, 4),
                    answer_tokens=graph_metrics["answer_tokens"],
                    word_count=graph_metrics["word_count"],
                    retrieved_chunk_count=graph_metrics["retrieved_chunk_count"],
                    retrieved_source_count=graph_metrics["retrieved_source_count"],
                    confidence_score=graph_metrics["confidence_score"],
                    relevance_score=graph_metrics["relevance_score"],
                    source_coverage=graph_metrics["source_coverage"],
                    latency_breakdown_json=_safe_json(graph_metrics["latency_breakdown"]),
                    metric_groups_json=_safe_json(graph_metrics["metric_groups"]),
                )
            )

    return records


def _write_outputs(records: List[BenchmarkRecord], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.json"
    summary_path = output_dir / "summary.csv"

    raw_rows = [asdict(record) for record in records]
    raw_path.write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(raw_rows[0].keys()) if raw_rows else [field.name for field in BenchmarkRecord.__dataclass_fields__.values()]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    return {"raw": raw_path, "summary": summary_path}


def _print_summary(records: List[BenchmarkRecord]) -> None:
    grouped: Dict[str, List[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.strategy, []).append(record)

    for pipeline, items in grouped.items():
        if not items:
            continue
        print(f"[{pipeline}]")
        print(f"  runs: {len(items)}")
        print(f"  avg_latency_s: {mean(item.total_time_s for item in items):.4f}")
        print(f"  median_latency_s: {median(item.total_time_s for item in items):.4f}")
        print(f"  avg_retrieved_chunks: {mean(item.retrieved_chunk_count for item in items):.4f}")
        print(f"  avg_confidence: {mean((item.confidence_score or 0.0) for item in items):.4f}")


async def _async_main(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    documents_dir = Path(args.documents_dir).resolve() if args.documents_dir else None
    reranking_modes = [False, True]
    if args.reranking_mode == "baseline":
        reranking_modes = [False]
    elif args.reranking_mode == "rerank":
        reranking_modes = [True]
    records = await _run_benchmark(
        dataset_path,
        documents_dir=documents_dir,
        reranking_modes=reranking_modes,
        top_k=args.top_k,
    )
    _write_outputs(records, output_dir)
    _print_summary(records)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SmartDoc AI benchmark for PaCRAG vs GraphRAG")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to benchmark_questions.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for raw.json and summary.csv")
    parser.add_argument("--documents-dir", default=str(DEFAULT_DOCUMENTS_DIR), help="Folder to ingest benchmark documents before running queries")
    parser.add_argument("--reranking-mode", choices=["baseline", "rerank", "both"], default="both", help="Run baseline, rerank, or both benchmark variants")
    parser.add_argument("--top-k", type=int, default=15, help="Number of chunks to retrieve before reranking (default: 15)")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
