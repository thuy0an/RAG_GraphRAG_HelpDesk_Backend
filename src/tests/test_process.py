"""
Unit tests cho Process.split_PaC — chunk strategy.
Không cần LLM hay database.
Chạy: python -m pytest src/tests/test_process.py -v
"""
import asyncio
import pytest
from langchain_core.documents import Document
from src.Features.LangChainAPI.RAG.Process import Process


@pytest.fixture
def process():
    return Process()


def run(coro):
    """Helper chạy coroutine đồng bộ."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_docs(texts: list, source: str = "test.pdf") -> list:
    return [
        Document(
            page_content=text,
            metadata={"page_number": i + 1, "source": source, "content_type": "application/pdf"}
        )
        for i, text in enumerate(texts)
    ]


# ─────────────────────────────────────────────
# split_PaC — basic
# ─────────────────────────────────────────────

def test_split_pac_returns_dict(process):
    docs = _make_docs(["Nội dung tài liệu test " * 50])
    result = run(process.split_PaC(docs))
    assert isinstance(result, dict)
    assert "parent" in result
    assert "children" in result


def test_split_pac_empty_docs(process):
    result = run(process.split_PaC([]))
    assert result == {"parent": [], "children": []}


def test_split_pac_children_have_source_metadata(process):
    docs = _make_docs(["Đây là nội dung tài liệu " * 100], source="document.pdf")
    result = run(process.split_PaC(docs))
    for child in result["children"]:
        assert child.metadata.get("source") == "document.pdf"


def test_split_pac_children_have_pages_metadata(process):
    docs = _make_docs(["Nội dung trang 1 " * 100], source="test.pdf")
    result = run(process.split_PaC(docs))
    for child in result["children"]:
        assert "pages" in child.metadata
        assert isinstance(child.metadata["pages"], list)


def test_split_pac_children_have_parent_id(process):
    docs = _make_docs(["Nội dung " * 200])
    result = run(process.split_PaC(docs))
    for child in result["children"]:
        assert "parent_id" in child.metadata


def test_split_pac_custom_chunk_size(process):
    """Chunk size nhỏ hơn → nhiều chunk hơn."""
    docs = _make_docs(["word " * 500])
    result_default = run(process.split_PaC(docs))
    result_small = run(process.split_PaC(docs, child_chunk_size=100, child_chunk_overlap=10))
    assert len(result_small["children"]) >= len(result_default["children"])


def test_split_pac_parent_count_less_than_children(process):
    """Parent chunks ít hơn hoặc bằng child chunks."""
    docs = _make_docs(["Nội dung dài " * 300])
    result = run(process.split_PaC(docs))
    assert len(result["parent"]) <= len(result["children"])


def test_split_pac_multiple_pages(process):
    """Nhiều trang → metadata pages phải phản ánh đúng."""
    docs = _make_docs([
        "Trang một nội dung " * 50,
        "Trang hai nội dung " * 50,
        "Trang ba nội dung " * 50,
    ])
    result = run(process.split_PaC(docs))
    assert len(result["children"]) > 0
    all_pages = []
    for child in result["children"]:
        all_pages.extend(child.metadata.get("pages", []))
    assert len(all_pages) > 0


# ─────────────────────────────────────────────
# split_PaC — chunk overlap
# ─────────────────────────────────────────────

def test_split_pac_overlap_zero(process):
    """overlap=0 vẫn hoạt động bình thường."""
    docs = _make_docs(["Nội dung test " * 100])
    result = run(process.split_PaC(docs, child_chunk_size=200, child_chunk_overlap=0))
    assert len(result["children"]) > 0


def test_split_pac_large_overlap(process):
    """overlap lớn không gây lỗi."""
    docs = _make_docs(["Nội dung test " * 100])
    result = run(process.split_PaC(docs, child_chunk_size=300, child_chunk_overlap=100))
    assert len(result["children"]) > 0
