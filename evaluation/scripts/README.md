# Evaluation Scripts

Chay cac script nay bang dung backend environment de dam bao du cac thu vien da cai:

```powershell
cd AI_HelpDesk_Backend
venv\Scripts\activate
$env:PYTHONPATH = "src"
```

## Run benchmark

```bash
PYTHONPATH=src python evaluation/scripts/run_benchmark.py
```

Mac dinh script se ingest tai lieu trong `AI_HelpDesk_Backend/tailieu` truoc khi chay query.
Neu muon chi dinh thu muc khac:

```bash
PYTHONPATH=src python evaluation/scripts/run_benchmark.py --documents-dir AI_HelpDesk_Backend/tailieu
```

## Generate plots

```bash
PYTHONPATH=src python evaluation/scripts/generate_plots.py
```

## Generate summary table

```bash
PYTHONPATH=src python evaluation/scripts/generate_summary_table.py
```

## Benchmark chunk_size/overlap grid

```bash
PYTHONPATH=src python evaluation/scripts/benchmark_chunk_grid.py
```

## Compare answer-point accuracy

```bash
PYTHONPATH=src python evaluation/scripts/compare_accuracy.py
```

## BM25 + EnsembleRetriever weight tuning

```bash
PYTHONPATH=src python evaluation/scripts/benchmark_ensemble_retriever.py
```

## Performance profiling and latency optimization hints

```bash
PYTHONPATH=src python evaluation/scripts/profile_performance.py
```

## Output
- `evaluation/results/raw.json`
- `evaluation/results/summary.csv`
- `evaluation/results/summary_table.md`
- `evaluation/results/summary_table.tex`
- `evaluation/results/chunk_grid_benchmark.csv`
- `evaluation/results/accuracy_detail.csv`
- `evaluation/results/accuracy_summary.csv`
- `evaluation/results/ensemble_tuning.csv`
- `evaluation/results/ensemble_best.json`
- `evaluation/results/profile_report.json`
- `evaluation/results/profile_report.md`
- `evaluation/plots/latency_comparison.png`
- `evaluation/plots/retrieval_comparison.png`
