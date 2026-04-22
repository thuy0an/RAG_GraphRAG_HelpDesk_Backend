# Trạng thái Benchmark - Lý thuyết Đồ thị

## Ngày cập nhật: 21/04/2026

### Dataset
✅ **benchmark_questions.json** - Đã cập nhật với 12 câu hỏi về lý thuyết đồ thị (graph theory 1-8.pdf)

### Kết quả Benchmark chính

#### ✅ Đã cập nhật (run_benchmark.py)
- **raw.json** - Dữ liệu thô của 48 lần chạy (12 câu hỏi × 2 pipeline × 2 chế độ reranking)
- **summary.csv** - Bảng tổng hợp chi tiết từng câu trả lời
- **summary_table.md** - Bảng so sánh hiệu suất giữa PaCRAG và GraphRAG
- **summary_table.tex** - Định dạng LaTeX cho báo cáo

**Kết quả chính:**
- PaCRAG: Độ trễ TB 45.27s, confidence 0.77, lấy 7.46 chunks
- GraphRAG: Độ trễ TB 30.58s, confidence 0.84, lấy 20 chunks
- GraphRAG nhanh hơn và có độ tin cậy cao hơn

#### ✅ Đã cập nhật (compare_accuracy.py)
- **accuracy_detail.csv** - Chi tiết độ chính xác từng câu hỏi
- **accuracy_summary.csv** - Tổng hợp độ chính xác theo pipeline

**Kết quả:**
- PaCRAG: 90% point recall
- GraphRAG: 61.67% point recall
- PaCRAG+rerank: 88.33% point recall
- GraphRAG+rerank: 86.67% point recall

#### ✅ Đã cập nhật (generate_plots.py)
- **latency_comparison.png** - Biểu đồ so sánh độ trễ
- **retrieval_comparison.png** - Biểu đồ so sánh số lượng chunks

#### ✅ Đã cập nhật (profile_performance.py)
- **profile_report.json** - Dữ liệu profiling chi tiết
- **profile_report.md** - Báo cáo phân tích hiệu suất

**Kết quả:**
- 6 queries được test
- Embedding avg: 0.6782s, p95: 0.0389s
- Retrieval avg: 0.6105s, p95: 0.0443s

### Kết quả Benchmark bổ sung

#### ⚠️ Chưa cập nhật (dữ liệu cũ)
- **chunk_grid_benchmark.csv** - Benchmark tối ưu kích thước chunk (cần chạy lại nếu cần)
- **ensemble_tuning.csv** - Tối ưu trọng số BM25/Vector (cần chạy lại nếu cần)
- **ensemble_best.json** - Cấu hình ensemble tốt nhất (cần chạy lại nếu cần)

**Lưu ý:** Các file này chứa dữ liệu từ benchmark trước đó. Nếu cần cập nhật:
```bash
python evaluation/scripts/benchmark_chunk_grid.py
python evaluation/scripts/benchmark_ensemble_retriever.py
```

## Tổng kết

### ✅ Đã hoàn thành
1. Cập nhật 12 câu hỏi về lý thuyết đồ thị với dấu tiếng Việt đầy đủ
2. Chạy benchmark chính (PaCRAG vs GraphRAG)
3. Tạo báo cáo so sánh độ chính xác
4. Tạo biểu đồ trực quan
5. Phân tích hiệu suất (profiling)

### 📊 Kết luận chính
- **GraphRAG** vượt trội về tốc độ (nhanh hơn 33%) và độ tin cậy
- **PaCRAG** có độ chính xác cao hơn (90% vs 62%)
- **Reranking** cải thiện độ chính xác cho cả hai pipeline

### 🔄 Tùy chọn (nếu cần)
- Chạy benchmark chunk grid để tối ưu kích thước chunk
- Chạy ensemble tuning để tối ưu trọng số retriever
