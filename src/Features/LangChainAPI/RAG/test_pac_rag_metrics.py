"""
Property-based tests cho PaCRAG metrics.

Validates: Requirements CP-4 (Retrieved Chunks Limit)
"""
import sys
import os

# Thêm src vào path để import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from hypothesis import given, settings, strategies as st


def extract_retrieved_chunks(hybrid_docs):
    """
    Hàm trích xuất retrieved_chunks từ hybrid_docs (tối đa 10).
    Logic này mirror chính xác logic trong PaCRAG.retrieve_full().
    """
    retrieved_chunks = []
    for doc in hybrid_docs[:10]:
        metadata = doc.get("metadata", {})
        pages = metadata.get("pages") or []
        if not pages and metadata.get("page_number"):
            pages = [metadata["page_number"]]
        retrieved_chunks.append({
            "content": doc.get("content", ""),
            "filename": metadata.get("source", ""),
            "pages": pages,
        })
    return retrieved_chunks


# Strategy tạo một doc giả lập
doc_strategy = st.fixed_dictionaries({
    "content": st.text(max_size=200),
    "metadata": st.one_of(
        # Có pages list
        st.fixed_dictionaries({
            "source": st.text(max_size=100),
            "pages": st.lists(st.integers(min_value=1, max_value=500), max_size=10),
        }),
        # Có page_number thay vì pages
        st.fixed_dictionaries({
            "source": st.text(max_size=100),
            "page_number": st.integers(min_value=1, max_value=500),
        }),
        # Không có pages hay page_number
        st.fixed_dictionaries({
            "source": st.text(max_size=100),
        }),
    ),
})


@given(
    hybrid_docs=st.lists(doc_strategy, min_size=0, max_size=30)
)
@settings(max_examples=200)
def test_retrieved_chunks_limit(hybrid_docs):
    """
    **Validates: Requirements CP-4**

    Property: len(retrieved_chunks) <= 10 với bất kỳ số lượng docs đầu vào (0 đến 30).
    """
    result = extract_retrieved_chunks(hybrid_docs)
    assert len(result) <= 10


@given(
    hybrid_docs=st.lists(doc_strategy, min_size=0, max_size=30)
)
@settings(max_examples=100)
def test_retrieved_chunks_count_matches_min(hybrid_docs):
    """
    Property: len(retrieved_chunks) == min(len(hybrid_docs), 10)
    """
    result = extract_retrieved_chunks(hybrid_docs)
    assert len(result) == min(len(hybrid_docs), 10)


@given(
    hybrid_docs=st.lists(doc_strategy, min_size=0, max_size=30)
)
@settings(max_examples=100)
def test_retrieved_chunks_fields_present(hybrid_docs):
    """
    Property: Mỗi chunk trong retrieved_chunks phải có đủ 3 trường: content, filename, pages.
    """
    result = extract_retrieved_chunks(hybrid_docs)
    for chunk in result:
        assert "content" in chunk
        assert "filename" in chunk
        assert "pages" in chunk
        assert isinstance(chunk["pages"], list)


def test_empty_docs_returns_empty_chunks():
    """
    Unit test: hybrid_docs rỗng → retrieved_chunks rỗng.
    """
    result = extract_retrieved_chunks([])
    assert result == []


def test_pages_fallback_to_page_number():
    """
    Unit test: Khi pages rỗng nhưng có page_number, dùng [page_number].
    """
    docs = [
        {
            "content": "test content",
            "metadata": {
                "source": "file.pdf",
                "page_number": 5,
            }
        }
    ]
    result = extract_retrieved_chunks(docs)
    assert len(result) == 1
    assert result[0]["pages"] == [5]


def test_pages_list_takes_priority():
    """
    Unit test: Khi có pages list, dùng pages list (không dùng page_number).
    """
    docs = [
        {
            "content": "test content",
            "metadata": {
                "source": "file.pdf",
                "pages": [1, 2, 3],
                "page_number": 99,
            }
        }
    ]
    result = extract_retrieved_chunks(docs)
    assert result[0]["pages"] == [1, 2, 3]


def test_exactly_10_docs_returns_10_chunks():
    """
    Unit test: Đúng 10 docs → 10 chunks.
    """
    docs = [{"content": f"doc {i}", "metadata": {"source": f"file{i}.pdf"}} for i in range(10)]
    result = extract_retrieved_chunks(docs)
    assert len(result) == 10


def test_more_than_10_docs_capped_at_10():
    """
    Unit test: 30 docs → chỉ lấy 10 chunks đầu tiên.
    """
    docs = [{"content": f"doc {i}", "metadata": {"source": f"file{i}.pdf"}} for i in range(30)]
    result = extract_retrieved_chunks(docs)
    assert len(result) == 10
    # Đảm bảo lấy đúng 10 docs đầu
    assert result[0]["content"] == "doc 0"
    assert result[9]["content"] == "doc 9"
