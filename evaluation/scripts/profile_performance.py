"""Performance profiling cho embedding va retrieval latency.

Script do latencies theo batch query va xuat ra profile_report.json + markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.Features.LangChainAPI.LangChainFacade import LangChainFacade
DEFAULT_DATASET = ROOT / "evaluation" / "dataset" / "benchmark_questions.json"
DEFAULT_JSON = ROOT / "evaluation" / "results" / "profile_report.json"
DEFAULT_MD = ROOT / "evaluation" / "results" / "profile_report.md"


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    items = sorted(values)
    idx = int((len(items) - 1) * p)
    return items[idx]


def _load_queries(dataset_path: Path, max_questions: int) -> List[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = [str(item.get("question", "")).strip() for item in data.get("questions", []) if item.get("question")]
    return questions[:max_questions] if max_questions > 0 else questions


def _recommend(embed_p95: float, retrieval_p95: float) -> List[str]:
    tips = []
    if embed_p95 > 2.0:
        tips.append("Embedding latency cao: can can nhac cache embedding query va dung model embedding nhe hon.")
    if retrieval_p95 > 5.0:
        tips.append("Retrieval latency cao: giam k, toi uu index Redis, va gioi han source filter theo document.")
    if not tips:
        tips.append("Embedding va retrieval latency dang o muc on dinh trong profile hien tai.")
    return tips


async def _async_main(args: argparse.Namespace) -> int:
    queries = _load_queries(Path(args.dataset).resolve(), args.max_questions)
    if not queries:
        raise RuntimeError("No queries found for profiling")

    facade = LangChainFacade()
    embed_times = []
    retrieve_times = []

    for query in queries:
        t0 = perf_counter()
        facade.embedding.embed_query(query)
        embed_times.append(perf_counter() - t0)

        t1 = perf_counter()
        await facade.PaCRAG.redis_vs_repo.hybrid_retriver(query=query, k=args.k)
        retrieve_times.append(perf_counter() - t1)

    embed_avg = mean(embed_times)
    retrieve_avg = mean(retrieve_times)
    embed_p95 = _percentile(embed_times, 0.95)
    retrieve_p95 = _percentile(retrieve_times, 0.95)

    recommendations = _recommend(embed_p95, retrieve_p95)
    payload = {
        "queries": len(queries),
        "embedding": {
            "avg_s": round(embed_avg, 4),
            "p95_s": round(embed_p95, 4),
        },
        "retrieval": {
            "avg_s": round(retrieve_avg, 4),
            "p95_s": round(retrieve_p95, 4),
            "k": args.k,
        },
        "recommendations": recommendations,
    }

    json_path = Path(args.output_json).resolve()
    md_path = Path(args.output_md).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Performance Profiling Report",
        "",
        f"- Queries: {payload['queries']}",
        f"- Embedding avg/p95 (s): {payload['embedding']['avg_s']} / {payload['embedding']['p95_s']}",
        f"- Retrieval avg/p95 (s): {payload['retrieval']['avg_s']} / {payload['retrieval']['p95_s']}",
        f"- Retrieval k: {payload['retrieval']['k']}",
        "",
        "## Optimization suggestions",
    ]
    for tip in recommendations:
        md_lines.append(f"- {tip}")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"profile_written: {json_path} | {md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile embedding/retrieval latency and suggest optimizations")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max-questions", type=int, default=6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
