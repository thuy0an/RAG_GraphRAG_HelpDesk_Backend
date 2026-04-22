# SmartDoc AI Benchmark Summary

## Pipeline comparison

| Pipeline | Runs | Avg latency (s) | Median latency (s) | Avg retrieved chunks | Avg retrieved sources | Avg confidence | Avg relevance | Avg source coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PaCRAG | 24 | 45.2654 | 40.7150 | 7.4583 | 4.0417 | 0.7711 | N/A | N/A |
| GraphRAG | 24 | 30.5825 | 32.7700 | 20.0000 | 3.2500 | 0.8417 | N/A | N/A |

## By question type

| Question type | Pipeline | Runs | Avg latency (s) | Avg retrieved chunks |
|---|---|---:|---:|---:|
| multi-hop | GraphRAG | 14 | 35.1886 | 20.0000 |
| multi-hop | PaCRAG | 14 | 51.0671 | 7.4286 |
| single-hop | GraphRAG | 10 | 24.1340 | 20.0000 |
| single-hop | PaCRAG | 10 | 37.1430 | 7.5000 |
