"""Benchmark chunk_size/chunk_overlap cho PaCRAG va GraphRAG.

Script nay chay theo dang grid search tren cac to hop chunking,
index lai du lieu cho moi cau hinh, sau do do query latency trung binh.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, List

from fastapi import UploadFile

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.Features.LangChainAPI.LangChainFacade import LangChainFacade


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
DEFAULT_DATASET = ROOT / "evaluation" / "dataset" / "benchmark_questions.json"
DEFAULT_DOC = REPO_ROOT / "smoke_test_assets" / "graph_smoke_test.pdf"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "chunk_grid_benchmark.csv"


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_dataset(path: Path, max_questions: int) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [str(item.get("question", "")).strip() for item in data.get("questions", []) if item.get("question")]
    return questions[:max_questions] if max_questions > 0 else questions


def _open_upload_file(path: Path) -> UploadFile:
    payload = path.read_bytes()
    return UploadFile(filename=path.name, file=io.BytesIO(payload))


async def _benchmark_pac(
    facade: LangChainFacade,
    doc_path: Path,
    questions: List[str],
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> Dict[str, Any]:
    filename = doc_path.name
    await facade.PaCRAG.clear_vector_store(source=filename)

    upload_file = _open_upload_file(doc_path)
    ingest_metrics = await facade.PaCRAG.index_with_metrics(
        upload_file,
        parent_chunk_size=parent_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_size,
        child_chunk_overlap=child_overlap,
    )

    latencies = []
    retrieved_chunks = []
    for query in questions:
        start = perf_counter()
        result = await facade.PaCRAG.retrieve_full(query, source_filter=filename)
        elapsed = perf_counter() - start
        latencies.append(float(result.get("time_total_s") or elapsed))
        retrieved_chunks.append(int(result.get("retrieved_chunk_count") or 0))

    return {
        "pipeline": "PaCRAG",
        "parent_chunk_size": parent_size,
        "parent_chunk_overlap": parent_overlap,
        "child_chunk_size": child_size,
        "child_chunk_overlap": child_overlap,
        "graph_chunk_size": "",
        "graph_chunk_overlap": "",
        "ingest_time_total_s": ingest_metrics.get("time_total_s", ""),
        "avg_query_latency_s": round(mean(latencies), 4) if latencies else 0.0,
        "median_query_latency_s": round(median(latencies), 4) if latencies else 0.0,
        "avg_retrieved_chunks": round(mean(retrieved_chunks), 4) if retrieved_chunks else 0.0,
    }


async def _benchmark_graph(
    facade: LangChainFacade,
    doc_path: Path,
    questions: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    filename = doc_path.name
    await facade.GraphRAG.delete(filename)

    upload_file = _open_upload_file(doc_path)
    ingest_metrics = await facade.GraphRAG.ingest(
        upload_file,
        source=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    latencies = []
    retrieved_chunks = []
    for query in questions:
        start = perf_counter()
        result = await facade.GraphRAG.retrieve_with_metrics(query, source=filename)
        elapsed = perf_counter() - start
        latencies.append(float(result.get("time_total_s") or elapsed))
        retrieved_chunks.append(int(result.get("retrieved_chunk_count") or 0))

    return {
        "pipeline": "GraphRAG",
        "parent_chunk_size": "",
        "parent_chunk_overlap": "",
        "child_chunk_size": "",
        "child_chunk_overlap": "",
        "graph_chunk_size": chunk_size,
        "graph_chunk_overlap": chunk_overlap,
        "ingest_time_total_s": ingest_metrics.get("time_total_s", ""),
        "avg_query_latency_s": round(mean(latencies), 4) if latencies else 0.0,
        "median_query_latency_s": round(median(latencies), 4) if latencies else 0.0,
        "avg_retrieved_chunks": round(mean(retrieved_chunks), 4) if retrieved_chunks else 0.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pipeline",
        "parent_chunk_size",
        "parent_chunk_overlap",
        "child_chunk_size",
        "child_chunk_overlap",
        "graph_chunk_size",
        "graph_chunk_overlap",
        "ingest_time_total_s",
        "avg_query_latency_s",
        "median_query_latency_s",
        "avg_retrieved_chunks",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def _async_main(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).resolve()
    doc_path = Path(args.document).resolve()
    output_path = Path(args.output).resolve()

    questions = _load_dataset(dataset_path, args.max_questions)
    if not questions:
        raise RuntimeError("No questions found in dataset")

    facade = LangChainFacade()
    rows: List[Dict[str, Any]] = []

    for p_size in _parse_int_list(args.pac_parent_sizes):
        for p_overlap in _parse_int_list(args.pac_parent_overlaps):
            for c_size in _parse_int_list(args.pac_child_sizes):
                for c_overlap in _parse_int_list(args.pac_child_overlaps):
                    row = await _benchmark_pac(
                        facade,
                        doc_path,
                        questions,
                        parent_size=p_size,
                        parent_overlap=p_overlap,
                        child_size=c_size,
                        child_overlap=c_overlap,
                    )
                    rows.append(row)

    for g_size in _parse_int_list(args.graph_chunk_sizes):
        for g_overlap in _parse_int_list(args.graph_chunk_overlaps):
            row = await _benchmark_graph(
                facade,
                doc_path,
                questions,
                chunk_size=g_size,
                chunk_overlap=g_overlap,
            )
            rows.append(row)

    _write_csv(output_path, rows)
    print(f"chunk_grid_written: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chunk-size/chunk-overlap grid")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--document", default=str(DEFAULT_DOC))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--pac-parent-sizes", default="1024,2048")
    parser.add_argument("--pac-parent-overlaps", default="200,400")
    parser.add_argument("--pac-child-sizes", default="128,256,512")
    parser.add_argument("--pac-child-overlaps", default="25,50,100")
    parser.add_argument("--graph-chunk-sizes", default="300,400,600,800")
    parser.add_argument("--graph-chunk-overlaps", default="50,100")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
