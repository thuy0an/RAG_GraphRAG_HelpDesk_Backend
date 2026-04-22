"""Sinh bieu do tu ket qua benchmark.

Doan script nay doc summary.csv va tao cac bieu do so sanh co ban.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "evaluation" / "results" / "summary.csv"
DEFAULT_PLOTS = ROOT / "evaluation" / "plots"


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pipeline"]].append(row)
    return grouped


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib_unavailable: {exc}")
        return 1

    rows = _load_rows(DEFAULT_RESULTS)
    grouped = _group_rows(rows)
    DEFAULT_PLOTS.mkdir(parents=True, exist_ok=True)

    pipelines = [name for name in ("PaCRAG", "GraphRAG") if name in grouped]
    if not pipelines:
        print("no_rows")
        return 1

    latency_means = [sum(float(row["total_time_s"]) for row in grouped[name]) / max(len(grouped[name]), 1) for name in pipelines]
    chunk_means = [sum(float(row["retrieved_chunk_count"]) for row in grouped[name]) / max(len(grouped[name]), 1) for name in pipelines]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pipelines, latency_means, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Average latency per pipeline")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(DEFAULT_PLOTS / "latency_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pipelines, chunk_means, color=["#2ca02c", "#d62728"])
    ax.set_title("Average retrieved chunks per pipeline")
    ax.set_ylabel("Chunks")
    fig.tight_layout()
    fig.savefig(DEFAULT_PLOTS / "retrieval_comparison.png", dpi=200)
    plt.close(fig)

    print("plots_generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
