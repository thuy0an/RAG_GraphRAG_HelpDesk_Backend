"""
Unit tests cho prompt.py — format_history_block, PaC_template, PaC_template_with_history.
Không cần LLM hay database.
Chạy: pytest src/tests/test_prompt.py -v
"""
import pytest
from src.Features.LangChainAPI.prompt import (
    format_history_block,
    PaC_template,
    PaC_template_with_history,
)


# ─────────────────────────────────────────────
# format_history_block
# ─────────────────────────────────────────────

def _make_turn(turn_id: str, user: str, rag: str = None, graphrag: str = None):
    """Helper tạo list messages cho 1 turn."""
    msgs = [{"turn_id": turn_id, "role": "user", "content": user}]
    if rag:
        msgs.append({"turn_id": turn_id, "role": "assistant_rag", "content": rag})
    if graphrag:
        msgs.append({"turn_id": turn_id, "role": "assistant_graphrag", "content": graphrag})
    return msgs


def test_format_history_block_empty_list():
    assert format_history_block([]) == ""


def test_format_history_block_no_turn_id():
    msgs = [{"role": "user", "content": "hello"}]  # thiếu turn_id
    assert format_history_block(msgs) == ""


def test_format_history_block_rag_key():
    msgs = _make_turn("t1", "Câu hỏi 1", rag="Trả lời RAG")
    result = format_history_block(msgs, role_key="rag_content")
    assert "Người dùng: Câu hỏi 1" in result
    assert "Trợ lý: Trả lời RAG" in result


def test_format_history_block_graphrag_key():
    msgs = _make_turn("t1", "Câu hỏi 1", graphrag="Trả lời GraphRAG")
    result = format_history_block(msgs, role_key="graphrag_content")
    assert "Người dùng: Câu hỏi 1" in result
    assert "Trợ lý: Trả lời GraphRAG" in result


def test_format_history_block_skips_incomplete_turn():
    """Turn chỉ có user, không có assistant → bị bỏ qua."""
    msgs = [{"turn_id": "t1", "role": "user", "content": "Câu hỏi"}]
    result = format_history_block(msgs, role_key="rag_content")
    assert result == ""


def test_format_history_block_multiple_turns_order():
    """Nhiều turn phải theo thứ tự oldest first."""
    msgs = (
        _make_turn("t1", "Q1", rag="A1")
        + _make_turn("t2", "Q2", rag="A2")
    )
    result = format_history_block(msgs, role_key="rag_content")
    pos_q1 = result.index("Q1")
    pos_q2 = result.index("Q2")
    assert pos_q1 < pos_q2


def test_format_history_block_no_trailing_blank_line():
    msgs = _make_turn("t1", "Q1", rag="A1")
    result = format_history_block(msgs, role_key="rag_content")
    assert not result.endswith("\n\n")


def test_format_history_block_wrong_role_key_returns_empty():
    """role_key không khớp với role trong messages → không có assistant → empty."""
    msgs = _make_turn("t1", "Q1", rag="A1")
    # dùng graphrag_content nhưng message chỉ có assistant_rag
    result = format_history_block(msgs, role_key="graphrag_content")
    assert result == ""


def test_format_history_block_mixed_turns():
    """Chỉ include turn có đủ cả user + assistant."""
    msgs = (
        _make_turn("t1", "Q1", rag="A1")          # đủ
        + [{"turn_id": "t2", "role": "user", "content": "Q2"}]  # thiếu assistant
        + _make_turn("t3", "Q3", rag="A3")          # đủ
    )
    result = format_history_block(msgs, role_key="rag_content")
    assert "Q1" in result
    assert "Q2" not in result
    assert "Q3" in result


# ─────────────────────────────────────────────
# PaC_template
# ─────────────────────────────────────────────

def test_pac_template_contains_context():
    tmpl = PaC_template("Đây là ngữ cảnh test")
    assert "Đây là ngữ cảnh test" in tmpl


def test_pac_template_contains_query_placeholder():
    tmpl = PaC_template("context")
    assert "{query}" in tmpl


def test_pac_template_contains_source_instruction():
    tmpl = PaC_template("context")
    assert "Nguồn" in tmpl
    assert "Trang" in tmpl


def test_pac_template_no_history_section():
    tmpl = PaC_template("context")
    assert "Lịch sử hội thoại" not in tmpl


# ─────────────────────────────────────────────
# PaC_template_with_history
# ─────────────────────────────────────────────

def test_pac_template_with_history_empty_history():
    """Khi history_block rỗng, không inject section lịch sử."""
    tmpl = PaC_template_with_history("context", history_block="")
    assert "Lịch sử hội thoại" not in tmpl
    assert "{query}" in tmpl


def test_pac_template_with_history_injects_history():
    history = "Người dùng: Q1\nTrợ lý: A1"
    tmpl = PaC_template_with_history("context", history_block=history)
    assert "Lịch sử hội thoại" in tmpl
    assert "Q1" in tmpl
    assert "A1" in tmpl


def test_pac_template_with_history_escapes_braces():
    """Curly braces trong history không được làm hỏng ChatPromptTemplate."""
    history = "Người dùng: {test}\nTrợ lý: {answer}"
    tmpl = PaC_template_with_history("context", history_block=history)
    # Braces phải được escape thành {{ }}
    assert "{{test}}" in tmpl
    assert "{{answer}}" in tmpl
    # Placeholder {query} vẫn phải còn nguyên
    assert "{query}" in tmpl


def test_pac_template_with_history_contains_context():
    tmpl = PaC_template_with_history("Ngữ cảnh quan trọng", history_block="")
    assert "Ngữ cảnh quan trọng" in tmpl


def test_pac_template_with_history_whitespace_only_history():
    """history_block chỉ có whitespace → không inject section."""
    tmpl = PaC_template_with_history("context", history_block="   \n  ")
    assert "Lịch sử hội thoại" not in tmpl
