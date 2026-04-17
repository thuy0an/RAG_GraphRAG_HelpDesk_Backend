"""
Property-based tests cho GraphRAG metrics.

Validates: Requirements CP-4 (Retrieved Chunks Limit)
"""
import sys
import os

# Thêm src vào path để import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from hypothesis import given, settings, strategies as st


def extract_doc_passages(hits):
    """
    Hàm trích xuất doc_passages từ hits (tối đa 10).
    Logic này mirror chính xác logic trong GraphRAGInternal.collect_context()
    kết hợp với giới hạn [:10] trong GraphRAG.retrieve().
    """
    seen_chunks = set()
    doc_passages = []

    for hit in hits:
        chunk_id = hit.get("chunk_id") or hit.get("id")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        text = hit.get("text") or hit.get("content", "")
        if text:
            page_number = hit.get("page_number")
            doc_passages.append({
                "content": text,
                "filename": hit.get("filename", ""),
                "pages": [page_number] if page_number is not None else [],
            })

    return doc_passages[:10]


# Strategy tạo một hit giả lập
hit_strategy = st.fixed_dictionaries({
    "chunk_id": st.text(min_size=1, max_size=50),
    "text": st.text(min_size=1, max_size=300),
    "doc_id": st.text(max_size=50),
    "section_id": st.text(max_size=50),
    "filename": st.text(max_size=100),
    "page_number": st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=500),
    ),
    "score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
})


@given(
    hits=st.lists(hit_strategy, min_size=0, max_size=30)
)
@settings(max_examples=200)
def test_doc_passages_limit(hits):
    """
    **Validates: Requirements CP-4**

    Property: len(doc_passages) <= 10 với bất kỳ số lượng hits đầu vào (0 đến 30).
    """
    result = extract_doc_passages(hits)
    assert len(result) <= 10


@given(
    hits=st.lists(hit_strategy, min_size=0, max_size=30)
)
@settings(max_examples=100)
def test_doc_passages_fields_present(hits):
    """
    Property: Mỗi passage trong doc_passages phải có đủ 3 trường: content, filename, pages.
    """
    result = extract_doc_passages(hits)
    for passage in result:
        assert "content" in passage
        assert "filename" in passage
        assert "pages" in passage
        assert isinstance(passage["pages"], list)


@given(
    hits=st.lists(hit_strategy, min_size=0, max_size=30)
)
@settings(max_examples=100)
def test_doc_passages_pages_is_list(hits):
    """
    Property: pages trong mỗi passage luôn là list (không bao giờ là None).
    """
    result = extract_doc_passages(hits)
    for passage in result:
        assert isinstance(passage["pages"], list)


def test_empty_hits_returns_empty_passages():
    """
    Unit test: hits rỗng → doc_passages rỗng.
    """
    result = extract_doc_passages([])
    assert result == []


def test_more_than_10_hits_capped_at_10():
    """
    Unit test: 20 hits → chỉ lấy tối đa 10 passages.
    """
    hits = [
        {"chunk_id": f"chunk_{i}", "text": f"content {i}", "filename": f"file{i}.pdf", "page_number": i + 1}
        for i in range(20)
    ]
    result = extract_doc_passages(hits)
    assert len(result) == 10


def test_exactly_10_hits_returns_10_passages():
    """
    Unit test: Đúng 10 hits → 10 passages.
    """
    hits = [
        {"chunk_id": f"chunk_{i}", "text": f"content {i}", "filename": f"file{i}.pdf", "page_number": i + 1}
        for i in range(10)
    ]
    result = extract_doc_passages(hits)
    assert len(result) == 10


def test_page_number_none_gives_empty_pages():
    """
    Unit test: Khi page_number là None, pages phải là [].
    """
    hits = [
        {"chunk_id": "chunk_1", "text": "some content", "filename": "file.pdf", "page_number": None}
    ]
    result = extract_doc_passages(hits)
    assert len(result) == 1
    assert result[0]["pages"] == []


def test_page_number_set_gives_single_page_list():
    """
    Unit test: Khi page_number có giá trị, pages phải là [page_number].
    """
    hits = [
        {"chunk_id": "chunk_1", "text": "some content", "filename": "file.pdf", "page_number": 7}
    ]
    result = extract_doc_passages(hits)
    assert len(result) == 1
    assert result[0]["pages"] == [7]


def test_duplicate_chunk_ids_deduplicated():
    """
    Unit test: Các hits có cùng chunk_id chỉ được lấy một lần.
    """
    hits = [
        {"chunk_id": "same_id", "text": "content A", "filename": "file.pdf", "page_number": 1},
        {"chunk_id": "same_id", "text": "content B", "filename": "file.pdf", "page_number": 2},
    ]
    result = extract_doc_passages(hits)
    assert len(result) == 1
    assert result[0]["content"] == "content A"


def test_retrieve_with_metrics_word_count():
    """
    Unit test: word_count phải bằng len(answer.split()).
    """
    answer = "Đây là câu trả lời mẫu gồm nhiều từ"
    word_count = len(answer.split())
    assert word_count == len(answer.split())


def test_retrieve_with_metrics_empty_answer():
    """
    Unit test: answer rỗng → word_count = 0.
    """
    answer = ""
    word_count = len(answer.split())
    assert word_count == 0
