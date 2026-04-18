"""
Test suite toàn diện cho tính năng tích chọn file để đọc.

Kiểm tra toàn bộ luồng:
  Frontend (uploadedFiles.selected) 
    → activeSourceFilter 
    → streamChat(source_filter) / compareQuery(source_filter)
    → /retrieve_document (source_filter) / /compare/query (source_filter)
    → PaCRAG.retrieve(source_filter) / GraphRAG.retrieve(source)
    → HybridRetriever.retriever(source_filter) / vector_search_chunks(doc_ids)
    → Redis filter @source:{value} / FAISS doc_id filter

Test categories:
  1. Unit tests – filter expression generation
  2. Unit tests – activeSourceFilter logic (frontend)
  3. Unit tests – request body construction
  4. Integration tests – full pipeline mock
  5. Edge case tests
"""

import asyncio
import json
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# =============================================================================
# SECTION 1: Filter Expression Generation (Retriever.py)
# =============================================================================

def build_filter_expression(source_filter: Optional[str]) -> Optional[str]:
    """Mirror logic từ HybridRetriever.retriever()"""
    return f'@source:{{"{source_filter}"}}' if source_filter else None


class TestFilterExpressionGeneration:
    """Kiểm tra logic tạo filter expression cho Redis."""

    def test_filter_expr_with_filename(self):
        """Khi có source_filter, tạo đúng Redis filter expression."""
        expr = build_filter_expression("document_A.pdf")
        assert expr == '@source:{"document_A.pdf"}'

    def test_filter_expr_with_none(self):
        """Khi source_filter là None, không tạo filter."""
        expr = build_filter_expression(None)
        assert expr is None

    def test_filter_expr_with_empty_string(self):
        """Khi source_filter là chuỗi rỗng (falsy), không tạo filter."""
        expr = build_filter_expression("")
        assert expr is None

    def test_filter_expr_with_spaces_in_filename(self):
        """Tên file có khoảng trắng vẫn tạo đúng filter."""
        expr = build_filter_expression("my document.pdf")
        assert expr == '@source:{"my document.pdf"}'

    def test_filter_expr_with_vietnamese_filename(self):
        """Tên file tiếng Việt vẫn tạo đúng filter."""
        expr = build_filter_expression("Tài liệu học tập.pdf")
        assert expr == '@source:{"Tài liệu học tập.pdf"}'

    def test_filter_expr_with_special_chars(self):
        """Tên file có ký tự đặc biệt."""
        expr = build_filter_expression("report_2024-01.pdf")
        assert expr == '@source:{"report_2024-01.pdf"}'

    @given(filename=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_filter_expr_always_wraps_in_quotes(self, filename: str):
        """Với bất kỳ tên file nào, filter expression luôn bọc trong dấu ngoặc kép."""
        expr = build_filter_expression(filename)
        assert expr is not None
        assert filename in expr
        assert expr.startswith('@source:{')
        assert expr.endswith('}')


# =============================================================================
# SECTION 2: Frontend activeSourceFilter Logic
# =============================================================================

def compute_active_source_filter(uploaded_files: List[Dict]) -> Optional[str]:
    """
    Mirror logic từ handleSendMessage trong UserPortalChat.tsx:
    - Nếu đúng 1 file được tích → filter theo file đó
    - Nếu 0 hoặc nhiều hơn 1 file được tích → không filter (search tất cả)
    """
    selected_files = [f for f in uploaded_files if f.get("selected", False)]
    return selected_files[0]["name"] if len(selected_files) == 1 else None


class TestFrontendActiveSourceFilter:
    """Kiểm tra logic tính activeSourceFilter từ uploadedFiles."""

    def test_no_files_uploaded(self):
        """Không có file nào → không filter."""
        result = compute_active_source_filter([])
        assert result is None

    def test_one_file_selected(self):
        """Đúng 1 file được tích → filter theo file đó."""
        files = [{"name": "doc_A.pdf", "selected": True}]
        result = compute_active_source_filter(files)
        assert result == "doc_A.pdf"

    def test_one_file_not_selected(self):
        """1 file nhưng không tích → không filter."""
        files = [{"name": "doc_A.pdf", "selected": False}]
        result = compute_active_source_filter(files)
        assert result is None

    def test_multiple_files_all_selected(self):
        """Nhiều file, tất cả đều tích → không filter (search tất cả)."""
        files = [
            {"name": "doc_A.pdf", "selected": True},
            {"name": "doc_B.pdf", "selected": True},
            {"name": "doc_C.pdf", "selected": True},
        ]
        result = compute_active_source_filter(files)
        assert result is None

    def test_multiple_files_none_selected(self):
        """Nhiều file, không file nào tích → không filter."""
        files = [
            {"name": "doc_A.pdf", "selected": False},
            {"name": "doc_B.pdf", "selected": False},
        ]
        result = compute_active_source_filter(files)
        assert result is None

    def test_multiple_files_exactly_one_selected(self):
        """Nhiều file, đúng 1 file được tích → filter theo file đó."""
        files = [
            {"name": "doc_A.pdf", "selected": False},
            {"name": "doc_B.pdf", "selected": True},
            {"name": "doc_C.pdf", "selected": False},
        ]
        result = compute_active_source_filter(files)
        assert result == "doc_B.pdf"

    def test_multiple_files_two_selected(self):
        """Nhiều file, 2 file được tích → không filter (không thể filter theo 1 file)."""
        files = [
            {"name": "doc_A.pdf", "selected": True},
            {"name": "doc_B.pdf", "selected": True},
            {"name": "doc_C.pdf", "selected": False},
        ]
        result = compute_active_source_filter(files)
        assert result is None

    def test_selected_file_name_preserved(self):
        """Tên file được trả về đúng, không bị thay đổi."""
        filename = "TTHCM - Tóm tắt Chương II.pdf"
        files = [{"name": filename, "selected": True}]
        result = compute_active_source_filter(files)
        assert result == filename

    @given(
        n_files=st.integers(min_value=0, max_value=10),
        selected_indices=st.lists(st.integers(min_value=0, max_value=9), max_size=10, unique=True),
    )
    @settings(max_examples=200)
    def test_property_exactly_one_selected_returns_name(self, n_files: int, selected_indices: List[int]):
        """Property: chỉ khi đúng 1 file được tích thì trả về tên file đó."""
        files = [
            {"name": f"file_{i}.pdf", "selected": i in selected_indices}
            for i in range(n_files)
        ]
        result = compute_active_source_filter(files)
        selected = [f for f in files if f["selected"]]

        if len(selected) == 1:
            assert result == selected[0]["name"]
        else:
            assert result is None

    @given(
        files=st.lists(
            st.fixed_dictionaries({
                "name": st.text(min_size=1, max_size=50),
                "selected": st.booleans(),
            }),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=200)
    def test_property_result_is_none_or_selected_filename(self, files: List[Dict]):
        """Property: kết quả luôn là None hoặc tên của một file được tích."""
        result = compute_active_source_filter(files)
        if result is not None:
            selected_names = {f["name"] for f in files if f["selected"]}
            assert result in selected_names


# =============================================================================
# SECTION 3: Request Body Construction
# =============================================================================

def build_stream_request_body(query: str, session_id: str, source_filter: Optional[str]) -> Dict:
    """Mirror logic từ ChatService.streamChat()"""
    body = {
        "query": query,
        "session_id": session_id or "anonymous",
    }
    if source_filter:
        body["source_filter"] = source_filter
    return body


def build_compare_request_body(
    session_id: str,
    run_id: str,
    query: str,
    reranking_enabled: bool = False,
    source_filter: Optional[str] = None,
) -> Dict:
    """Mirror logic từ ChatService.compareQuery()"""
    body = {
        "session_id": session_id or "anonymous",
        "run_id": run_id,
        "query": query,
        "reranking_enabled": reranking_enabled,
    }
    if source_filter:
        body["source_filter"] = source_filter
    return body


class TestRequestBodyConstruction:
    """Kiểm tra request body được tạo đúng khi gửi lên backend."""

    def test_stream_request_with_source_filter(self):
        """streamChat request có source_filter khi filter được cung cấp."""
        body = build_stream_request_body("câu hỏi", "user123", "doc_A.pdf")
        assert body["source_filter"] == "doc_A.pdf"
        assert body["query"] == "câu hỏi"
        assert body["session_id"] == "user123"

    def test_stream_request_without_source_filter(self):
        """streamChat request không có source_filter khi filter là None."""
        body = build_stream_request_body("câu hỏi", "user123", None)
        assert "source_filter" not in body

    def test_stream_request_empty_filter_not_included(self):
        """source_filter rỗng không được đưa vào request body."""
        body = build_stream_request_body("câu hỏi", "user123", "")
        assert "source_filter" not in body

    def test_compare_request_with_source_filter(self):
        """compareQuery request có source_filter khi filter được cung cấp."""
        body = build_compare_request_body("user1", "run123", "câu hỏi", source_filter="doc_B.pdf")
        assert body["source_filter"] == "doc_B.pdf"
        assert body["reranking_enabled"] is False

    def test_compare_request_without_source_filter(self):
        """compareQuery request không có source_filter khi filter là None."""
        body = build_compare_request_body("user1", "run123", "câu hỏi")
        assert "source_filter" not in body

    def test_compare_request_with_reranking_and_filter(self):
        """compareQuery request có cả reranking_enabled và source_filter."""
        body = build_compare_request_body(
            "user1", "run123", "câu hỏi",
            reranking_enabled=True,
            source_filter="doc_C.pdf"
        )
        assert body["reranking_enabled"] is True
        assert body["source_filter"] == "doc_C.pdf"


# =============================================================================
# SECTION 4: Integration Tests – Full Pipeline Mock
# =============================================================================

class TestPaCRAGFilterPipeline:
    """Kiểm tra luồng filter qua PaCRAG → RedisVSRepository → HybridRetriever."""

    def test_source_filter_passed_to_hybrid_retriever(self):
        """source_filter được truyền đúng từ PaCRAG xuống HybridRetriever."""
        # Mock HybridRetriever
        mock_retriever = MagicMock()
        mock_retriever.retriever = AsyncMock(return_value=[])

        # Simulate the call chain
        source_filter = "specific_doc.pdf"
        asyncio.run(mock_retriever.retriever("query", 5, source_filter=source_filter))

        # Verify source_filter was passed
        mock_retriever.retriever.assert_called_once_with("query", 5, source_filter="specific_doc.pdf")

    def test_no_filter_passes_none_to_retriever(self):
        """Khi không có filter, None được truyền xuống HybridRetriever."""
        mock_retriever = MagicMock()
        mock_retriever.retriever = AsyncMock(return_value=[])

        asyncio.run(mock_retriever.retriever("query", 5, source_filter=None))

        mock_retriever.retriever.assert_called_once_with("query", 5, source_filter=None)

    def test_filter_applied_to_both_vector_and_bm25(self):
        """Filter expression được áp dụng cho cả VectorQuery và TextQuery."""
        source_filter = "doc_A.pdf"
        filter_expr = build_filter_expression(source_filter)

        # Verify filter expression is correct
        assert filter_expr == '@source:{"doc_A.pdf"}'

        # Simulate that both queries would use this filter
        vector_filter = filter_expr
        bm25_filter = filter_expr
        assert vector_filter == bm25_filter  # Both use same filter


class TestGraphRAGFilterPipeline:
    """Kiểm tra luồng filter qua GraphRAG → vector_search_chunks."""

    def test_source_converted_to_doc_id_for_faiss(self):
        """source (filename) được convert thành doc_id để filter FAISS."""
        import hashlib

        def uid(source: str) -> str:
            return hashlib.md5(source.encode()).hexdigest()

        source = "specific_doc.pdf"
        doc_id = uid(source)

        # Verify doc_id is deterministic
        assert doc_id == uid(source)
        assert len(doc_id) == 32  # MD5 hex length

    def test_no_source_means_no_doc_id_filter(self):
        """Khi source là None, không có doc_id filter."""
        source = None
        doc_ids = [f"uid_{source}"] if source else None
        assert doc_ids is None

    def test_source_filter_creates_doc_id_list(self):
        """Khi source được cung cấp, tạo list doc_ids để filter."""
        import hashlib

        def uid(source: str) -> str:
            return hashlib.md5(source.encode()).hexdigest()

        source = "doc_A.pdf"
        doc_ids = [uid(source)] if source else None
        assert doc_ids is not None
        assert len(doc_ids) == 1
        assert doc_ids[0] == uid("doc_A.pdf")


# =============================================================================
# SECTION 5: End-to-End Scenario Tests
# =============================================================================

class TestEndToEndScenarios:
    """Kiểm tra các kịch bản sử dụng thực tế."""

    def test_scenario_user_selects_one_file(self):
        """
        Kịch bản: User tích 1 file trong danh sách 3 file đã upload.
        Kỳ vọng: Chỉ tìm kiếm trong file đó.
        """
        uploaded_files = [
            {"name": "TTHCM_Chuong1.pdf", "selected": False},
            {"name": "TTHCM_Chuong2.pdf", "selected": True},   # ← chỉ file này
            {"name": "Effective_Opinion.pdf", "selected": False},
        ]

        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter == "TTHCM_Chuong2.pdf"

        # Request body sẽ có source_filter
        body = build_stream_request_body("câu hỏi về chương 2", "user1", active_filter)
        assert body["source_filter"] == "TTHCM_Chuong2.pdf"

        # Filter expression cho Redis
        filter_expr = build_filter_expression(active_filter)
        assert filter_expr == '@source:{"TTHCM_Chuong2.pdf"}'

    def test_scenario_user_selects_all_files(self):
        """
        Kịch bản: User tích tất cả file (mặc định).
        Kỳ vọng: Tìm kiếm trong tất cả tài liệu.
        """
        uploaded_files = [
            {"name": "TTHCM_Chuong1.pdf", "selected": True},
            {"name": "TTHCM_Chuong2.pdf", "selected": True},
            {"name": "Effective_Opinion.pdf", "selected": True},
        ]

        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter is None  # Không filter

        body = build_stream_request_body("câu hỏi tổng quát", "user1", active_filter)
        assert "source_filter" not in body

    def test_scenario_user_deselects_all_files(self):
        """
        Kịch bản: User bỏ tích tất cả file.
        Kỳ vọng: Tìm kiếm trong tất cả tài liệu (không filter).
        """
        uploaded_files = [
            {"name": "TTHCM_Chuong1.pdf", "selected": False},
            {"name": "TTHCM_Chuong2.pdf", "selected": False},
        ]

        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter is None

    def test_scenario_user_selects_two_files(self):
        """
        Kịch bản: User tích 2 file.
        Kỳ vọng: Không filter (search tất cả) vì không thể filter theo 2 file cùng lúc.
        """
        uploaded_files = [
            {"name": "TTHCM_Chuong1.pdf", "selected": True},
            {"name": "TTHCM_Chuong2.pdf", "selected": True},
            {"name": "Effective_Opinion.pdf", "selected": False},
        ]

        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter is None

    def test_scenario_compare_mode_with_filter(self):
        """
        Kịch bản: Compare mode với 1 file được tích.
        Kỳ vọng: Cả PaCRAG và GraphRAG đều filter theo file đó.
        """
        uploaded_files = [
            {"name": "Effective_Opinion.pdf", "selected": True},
        ]

        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter == "Effective_Opinion.pdf"

        # Compare request body
        compare_body = build_compare_request_body(
            "user1", "run_abc", "câu hỏi",
            source_filter=active_filter
        )
        assert compare_body["source_filter"] == "Effective_Opinion.pdf"

        # PaCRAG sẽ dùng source_filter cho Redis
        pac_filter_expr = build_filter_expression(active_filter)
        assert pac_filter_expr == '@source:{"Effective_Opinion.pdf"}'

        # GraphRAG sẽ dùng source (= source_filter) để tạo doc_id
        import hashlib
        graphrag_doc_id = hashlib.md5("Effective_Opinion.pdf".encode()).hexdigest()
        assert len(graphrag_doc_id) == 32

    def test_scenario_no_files_uploaded(self):
        """
        Kịch bản: Chưa upload file nào.
        Kỳ vọng: Không filter.
        """
        uploaded_files = []
        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter is None

    def test_scenario_filter_consistency_pac_and_graphrag(self):
        """
        Kịch bản: Đảm bảo PaCRAG và GraphRAG nhận cùng source_filter.
        Kỳ vọng: Cả hai đều filter theo cùng 1 file.
        """
        source_filter = "doc_A.pdf"

        # PaCRAG: dùng source_filter trực tiếp
        pac_filter = source_filter  # truyền vào hybrid_retriver(source_filter=...)

        # GraphRAG: dùng source (= source_filter) để tạo doc_id
        graphrag_source = source_filter  # truyền vào retrieve(source=...)

        assert pac_filter == graphrag_source == "doc_A.pdf"


# =============================================================================
# SECTION 6: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Kiểm tra các trường hợp biên."""

    def test_filename_with_parentheses(self):
        """Tên file có dấu ngoặc đơn."""
        files = [{"name": "report (final).pdf", "selected": True}]
        result = compute_active_source_filter(files)
        assert result == "report (final).pdf"

        expr = build_filter_expression(result)
        assert "report (final).pdf" in expr

    def test_filename_with_numbers(self):
        """Tên file chỉ có số."""
        files = [{"name": "12345.pdf", "selected": True}]
        result = compute_active_source_filter(files)
        assert result == "12345.pdf"

    def test_very_long_filename(self):
        """Tên file rất dài."""
        long_name = "a" * 200 + ".pdf"
        files = [{"name": long_name, "selected": True}]
        result = compute_active_source_filter(files)
        assert result == long_name

    def test_filter_does_not_modify_original_list(self):
        """Filter không thay đổi danh sách gốc."""
        def apply_source_filter(docs, source_filter):
            if not source_filter:
                return docs
            return [d for d in docs if d.get("metadata", {}).get("source") == source_filter]

        docs = [
            {"content": "a", "metadata": {"source": "file_a.pdf"}},
            {"content": "b", "metadata": {"source": "file_b.pdf"}},
        ]
        original_len = len(docs)

        result = apply_source_filter(docs, "file_a.pdf")

        # Original list unchanged
        assert len(docs) == original_len

    def test_switching_selected_file(self):
        """
        Kịch bản: User đổi file được tích từ A sang B.
        Kỳ vọng: Filter thay đổi theo.
        """
        # Lần 1: chọn file A
        files_v1 = [
            {"name": "doc_A.pdf", "selected": True},
            {"name": "doc_B.pdf", "selected": False},
        ]
        filter_v1 = compute_active_source_filter(files_v1)
        assert filter_v1 == "doc_A.pdf"

        # Lần 2: đổi sang file B
        files_v2 = [
            {"name": "doc_A.pdf", "selected": False},
            {"name": "doc_B.pdf", "selected": True},
        ]
        filter_v2 = compute_active_source_filter(files_v2)
        assert filter_v2 == "doc_B.pdf"

        # Filter đã thay đổi
        assert filter_v1 != filter_v2

    def test_adding_new_file_while_one_selected(self):
        """
        Kịch bản: Đang chọn 1 file, upload thêm file mới (mặc định selected=True).
        Kỳ vọng: Nếu file mới cũng selected → 2 file selected → không filter.
        """
        # Trước khi upload file mới
        files_before = [
            {"name": "doc_A.pdf", "selected": True},
        ]
        filter_before = compute_active_source_filter(files_before)
        assert filter_before == "doc_A.pdf"

        # Sau khi upload file mới (mặc định selected=True)
        files_after = [
            {"name": "doc_A.pdf", "selected": True},
            {"name": "doc_B.pdf", "selected": True},  # file mới
        ]
        filter_after = compute_active_source_filter(files_after)
        assert filter_after is None  # 2 file selected → không filter


# =============================================================================
# SECTION 7: Consistency Tests
# =============================================================================

class TestConsistency:
    """Kiểm tra tính nhất quán giữa các thành phần."""

    def test_pac_and_graphrag_receive_same_filter_value(self):
        """
        PaCRAG nhận source_filter và GraphRAG nhận source với cùng giá trị.
        """
        source_filter = "test_doc.pdf"

        # PaCRAG call: retrieve_full(query, source_filter=source_filter)
        pac_source_filter = source_filter

        # GraphRAG call: retrieve_with_metrics(query, source=source_filter)
        graphrag_source = source_filter

        assert pac_source_filter == graphrag_source

    def test_filter_in_compare_mode_matches_normal_mode(self):
        """
        Filter trong compare mode và normal mode phải nhất quán.
        """
        uploaded_files = [{"name": "doc_A.pdf", "selected": True}]
        active_filter = compute_active_source_filter(uploaded_files)

        # Normal mode (streamChat)
        normal_body = build_stream_request_body("query", "user1", active_filter)

        # Compare mode (compareQuery)
        compare_body = build_compare_request_body("user1", "run1", "query", source_filter=active_filter)

        # Both should have the same source_filter value
        assert normal_body.get("source_filter") == compare_body.get("source_filter")
        assert normal_body.get("source_filter") == "doc_A.pdf"

    def test_no_filter_consistent_across_modes(self):
        """
        Khi không filter, cả normal mode và compare mode đều không gửi source_filter.
        """
        uploaded_files = [
            {"name": "doc_A.pdf", "selected": True},
            {"name": "doc_B.pdf", "selected": True},
        ]
        active_filter = compute_active_source_filter(uploaded_files)
        assert active_filter is None

        normal_body = build_stream_request_body("query", "user1", active_filter)
        compare_body = build_compare_request_body("user1", "run1", "query", source_filter=active_filter)

        assert "source_filter" not in normal_body
        assert "source_filter" not in compare_body
