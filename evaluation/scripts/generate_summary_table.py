"""Tong hop summary.csv thanh bang markdown va LaTeX cho bao cao.

Script nay gom so lieu theo pipeline va loai cau hoi, sau do xuat ra:
- summary_table.md: de doc nhanh va so sanh
- summary_table.tex: de nhung truc tiep vao report_final.tex neu can
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "evaluation" / "results" / "summary.csv"
DEFAULT_OUTPUT_MD = ROOT / "evaluation" / "results" / "summary_table.md"
DEFAULT_OUTPUT_TEX = ROOT / "evaluation" / "results" / "summary_table.tex"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _group_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("pipeline", "unknown")].append(row)
    return grouped


def _build_pipeline_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = _group_rows(rows)
    pipeline_rows: List[Dict[str, str]] = []

    for pipeline in ("PaCRAG", "GraphRAG"):
        items = grouped.get(pipeline, [])
        if not items:
            continue

        latency_values = [_to_float(item.get("total_time_s", "0")) for item in items]
        chunk_values = [_to_float(item.get("retrieved_chunk_count", "0")) for item in items]
        source_values = [_to_float(item.get("retrieved_source_count", "0")) for item in items]
        confidence_values = [_to_float(item.get("confidence_score", "0")) for item in items if item.get("confidence_score") not in (None, "", "None")]
        relevance_values = [_to_float(item.get("relevance_score", "0")) for item in items if item.get("relevance_score") not in (None, "", "None")]
        coverage_values = [_to_float(item.get("source_coverage", "0")) for item in items if item.get("source_coverage") not in (None, "", "None")]

        pipeline_rows.append(
            {
                "pipeline": pipeline,
                "runs": str(len(items)),
                "avg_latency_s": f"{mean(latency_values):.4f}",
                "median_latency_s": f"{median(latency_values):.4f}",
                "avg_retrieved_chunks": f"{mean(chunk_values):.4f}",
                "avg_retrieved_sources": f"{mean(source_values):.4f}",
                "avg_confidence": f"{mean(confidence_values):.4f}" if confidence_values else "N/A",
                "avg_relevance": f"{mean(relevance_values):.4f}" if relevance_values else "N/A",
                "avg_source_coverage": f"{mean(coverage_values):.4f}" if coverage_values else "N/A",
            }
        )

    return pipeline_rows


def _build_question_type_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("question_type", "unknown"), row.get("pipeline", "unknown"))
        grouped[key].append(row)

    output_rows: List[Dict[str, str]] = []
    for (question_type, pipeline), items in sorted(grouped.items()):
        latency_values = [_to_float(item.get("total_time_s", "0")) for item in items]
        chunk_values = [_to_float(item.get("retrieved_chunk_count", "0")) for item in items]
        output_rows.append(
            {
                "question_type": question_type,
                "pipeline": pipeline,
                "runs": str(len(items)),
                "avg_latency_s": f"{mean(latency_values):.4f}",
                "avg_retrieved_chunks": f"{mean(chunk_values):.4f}",
            }
        )

    return output_rows


def _render_markdown(pipeline_rows: List[Dict[str, str]], question_rows: List[Dict[str, str]]) -> str:
    lines = ["# SmartDoc AI Benchmark Summary", "", "## Pipeline comparison", ""]
    lines.append("| Pipeline | Runs | Avg latency (s) | Median latency (s) | Avg retrieved chunks | Avg retrieved sources | Avg confidence | Avg relevance | Avg source coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in pipeline_rows:
        lines.append(
            f"| {row['pipeline']} | {row['runs']} | {row['avg_latency_s']} | {row['median_latency_s']} | {row['avg_retrieved_chunks']} | {row['avg_retrieved_sources']} | {row['avg_confidence']} | {row['avg_relevance']} | {row['avg_source_coverage']} |"
        )

    lines.extend(["", "## By question type", ""])
    lines.append("| Question type | Pipeline | Runs | Avg latency (s) | Avg retrieved chunks |")
    lines.append("|---|---|---:|---:|---:|")
    for row in question_rows:
        lines.append(
            f"| {row['question_type']} | {row['pipeline']} | {row['runs']} | {row['avg_latency_s']} | {row['avg_retrieved_chunks']} |"
        )

    return "\n".join(lines) + "\n"


def _render_latex(pipeline_rows: List[Dict[str, str]], question_rows: List[Dict[str, str]]) -> str:
    pipeline_header = "\\textbf{Pipeline} & \\textbf{Runs} & \\textbf{Avg latency (s)} & \\textbf{Median latency (s)} & \\textbf{Avg chunks} & \\textbf{Avg sources} \\\\" 
    question_header = "\\textbf{Question type} & \\textbf{Pipeline} & \\textbf{Runs} & \\textbf{Avg latency (s)} & \\textbf{Avg chunks} \\\\" 

    lines = [
        "% Auto-generated from evaluation/results/summary.csv",
        "\\begin{table}[h]",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\caption{Tong hop benchmark PaCRAG va GraphRAG}",
        "\\label{tab:auto-benchmark-summary}",
        "\\centering",
        "\\begin{tabular}{|l|c|c|c|c|c|}",
        "\\hline",
        pipeline_header,
        "\\hline",
    ]

    for row in pipeline_rows:
        lines.append(
            f"{row['pipeline']} & {row['runs']} & {row['avg_latency_s']} & {row['median_latency_s']} & {row['avg_retrieved_chunks']} & {row['avg_retrieved_sources']} \\\\" 
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[h]",
            "\\renewcommand{\\arraystretch}{1.2}",
            "\\caption{Tong hop theo loai cau hoi}",
            "\\label{tab:auto-benchmark-by-type}",
            "\\centering",
            "\\begin{tabular}{|l|l|c|c|c|}",
            "\\hline",
            question_header,
            "\\hline",
        ]
    )

    for row in question_rows:
        lines.append(
            f"{row['question_type']} & {row['pipeline']} & {row['runs']} & {row['avg_latency_s']} & {row['avg_retrieved_chunks']} \\\\" 
        )
        lines.append("\\hline")

    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    rows = _load_rows(DEFAULT_RESULTS)
    if not rows:
        print("summary_csv_empty")
        return 1

    pipeline_rows = _build_pipeline_rows(rows)
    question_rows = _build_question_type_rows(rows)

    DEFAULT_OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_MD.write_text(_render_markdown(pipeline_rows, question_rows), encoding="utf-8")
    DEFAULT_OUTPUT_TEX.write_text(_render_latex(pipeline_rows, question_rows), encoding="utf-8")
    print(f"summary_tables_written: {DEFAULT_OUTPUT_MD} | {DEFAULT_OUTPUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
