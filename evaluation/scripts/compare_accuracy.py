"""So sanh accuracy tren ket qua benchmark.

Accuracy duoc tinh theo do phu expected_answer_points trong cau tra loi.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "evaluation" / "dataset" / "benchmark_questions.json"
DEFAULT_RAW = ROOT / "evaluation" / "results" / "raw.json"
DEFAULT_DETAIL = ROOT / "evaluation" / "results" / "accuracy_detail.csv"
DEFAULT_SUMMARY = ROOT / "evaluation" / "results" / "accuracy_summary.csv"


def _norm(text: str) -> str:
    raw = (text or "").strip().lower()
    no_diacritics = "".join(ch for ch in unicodedata.normalize("NFD", raw) if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", no_diacritics)
    return re.sub(r"\s+", " ", cleaned).strip()


def _norm_keep_diacritics(text: str) -> str:
    """Normalize nhưng giữ dấu tiếng Việt để match chính xác hơn."""
    raw = (text or "").strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_overlap_ratio(answer_norm: str, point_norm: str) -> float:
    point_tokens = set(point_norm.split())
    if not point_tokens:
        return 0.0
    answer_tokens = set(answer_norm.split())
    if not answer_tokens:
        return 0.0
    return len(point_tokens & answer_tokens) / len(point_tokens)


def _load_expected(dataset_path: Path) -> Dict[str, List[str]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected: Dict[str, List[str]] = {}
    for row in data.get("questions", []):
        qid = str(row.get("id", ""))
        points = [str(x) for x in row.get("expected_answer_points", []) if x]
        expected[qid] = points
    return expected


def _point_recall(answer: str, points: List[str]) -> float:
    """
    Tính recall dựa trên expected_answer_points.
    Thử match theo 3 cách theo thứ tự ưu tiên:
    1. Exact match sau khi bỏ dấu (no-diacritics)
    2. Exact match giữ dấu tiếng Việt
    3. Token overlap >= 0.5 (hạ từ 0.6 để bắt paraphrase tốt hơn)
    """
    if not points:
        return 0.0
    text_no_diac = _norm(answer)
    text_with_diac = _norm_keep_diacritics(answer)
    hits = 0
    for p in points:
        point_no_diac = _norm(p)
        point_with_diac = _norm_keep_diacritics(p)
        if not point_no_diac:
            continue
        # 1. exact match no-diacritics
        if point_no_diac in text_no_diac:
            hits += 1
            continue
        # 2. exact match with diacritics
        if point_with_diac and point_with_diac in text_with_diac:
            hits += 1
            continue
        # 3. token overlap (hạ ngưỡng xuống 0.5)
        if _token_overlap_ratio(text_no_diac, point_no_diac) >= 0.5:
            hits += 1
            continue
        # 4. token overlap giữ dấu
        if _token_overlap_ratio(text_with_diac, point_with_diac) >= 0.5:
            hits += 1
    return hits / len(points)


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare answer-point accuracy from raw benchmark output")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--detail", default=str(DEFAULT_DETAIL))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    expected_map = _load_expected(Path(args.dataset).resolve())
    raw_rows = json.loads(Path(args.raw).resolve().read_text(encoding="utf-8"))

    detail_rows = []
    grouped = defaultdict(list)

    for row in raw_rows:
        qid = str(row.get("question_id", ""))
        pipeline = str(row.get("strategy") or row.get("pipeline", "unknown"))
        answer = str(row.get("answer", ""))
        points = expected_map.get(qid, [])
        recall = _point_recall(answer, points)

        grouped[pipeline].append(recall)
        detail_rows.append(
            {
                "question_id": qid,
                "strategy": pipeline,
                "pipeline": pipeline,
                "expected_points": str(len(points)),
                "point_recall": f"{recall:.4f}",
            }
        )

    summary_rows = []
    for pipeline, values in grouped.items():
        summary_rows.append(
            {
                "pipeline": pipeline,
                "runs": str(len(values)),
                "avg_point_recall": f"{mean(values):.4f}" if values else "0.0000",
            }
        )

    _write_csv(
        Path(args.detail).resolve(),
        ["question_id", "strategy", "pipeline", "expected_points", "point_recall"],
        detail_rows,
    )
    _write_csv(
        Path(args.summary).resolve(),
        ["pipeline", "runs", "avg_point_recall"],
        summary_rows,
    )
    print(f"accuracy_written: {args.detail} | {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
