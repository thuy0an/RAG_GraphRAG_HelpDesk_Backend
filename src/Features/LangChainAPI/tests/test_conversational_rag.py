"""
Tests cho Conversational RAG (8.2.6).

Chạy: pytest src/Features/LangChainAPI/tests/test_conversational_rag.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.Features.LangChainAPI.prompt import (
    format_history_block,
    PaC_template_with_history,
)
from src.Features.LangChainAPI.RAG.GraphRAGInternal import GraphRAGInternal


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_turns(pairs, role_key="rag_content"):
    """
    Tạo list messages giả lập output của MemoryRepository.get_recent_messages().
    pairs: list of (user_content, assistant_content | None)
    """
    role_map = {
        "rag_content": "assistant_rag",
        "graphrag_content": "assistant_graphrag",
    }
    assistant_role = role_map.get(role_key, "assistant_rag")
    messages = []
    for i, (user, assistant) in enumerate(pairs):
        turn_id = f"turn-{i}"
        messages.append({
            "turn_id": turn_id,
            "role": "user",
            "content": user,
            "timestamp": f"2026-04-17 10:0{i}:00",
        })
        if assistant is not None:
            messages.append({
                "turn_id": turn_id,
                "role": assistant_role,
                "content": assistant,
                "timestamp": f"2026-04-17 10:0{i}:01",
            })
    return messages


# ─── Tests: format_history_block ────────────────────────────────────────────

class TestFormatHistoryBlock:
    def test_empty_turns_returns_empty_string(self):
        """Test 1: Danh sách rỗng → trả về ""."""
        assert format_history_block([]) == ""

    def test_single_complete_turn(self):
        """Test 2: 1 turn đầy đủ → format đúng."""
        turns = make_turns([("Xin chào", "Chào bạn!")])
        result = format_history_block(turns, role_key="rag_content")
        assert "Người dùng: Xin chào" in result
        assert "Trợ lý: Chào bạn!" in result

    def test_turns_with_null_assistant_excluded(self):
        """Test 3: Turn không có assistant content bị bỏ qua."""
        turns = make_turns([
            ("Câu hỏi 1", "Trả lời 1"),
            ("Câu hỏi 2", None),   # NULL assistant
            ("Câu hỏi 3", "Trả lời 3"),
        ])
        result = format_history_block(turns, role_key="rag_content")
        assert "Câu hỏi 1" in result
        assert "Trả lời 1" in result
        assert "Câu hỏi 2" not in result  # bị loại vì không có assistant
        assert "Câu hỏi 3" in result
        assert "Trả lời 3" in result

    def test_all_null_assistant_returns_empty(self):
        """Test 4: Tất cả turns đều không có assistant → trả về ""."""
        turns = make_turns([("Q1", None), ("Q2", None)])
        assert format_history_block(turns, role_key="rag_content") == ""

    def test_rag_role_key_filters_rag_content(self):
        """Test 5: role_key='rag_content' chỉ lấy assistant_rag."""
        # Tạo turns với cả assistant_rag và assistant_graphrag
        turns = [
            {"turn_id": "t1", "role": "user", "content": "Q1", "timestamp": ""},
            {"turn_id": "t1", "role": "assistant_rag", "content": "RAG answer", "timestamp": ""},
            {"turn_id": "t1", "role": "assistant_graphrag", "content": "Graph answer", "timestamp": ""},
        ]
        result = format_history_block(turns, role_key="rag_content")
        assert "RAG answer" in result
        assert "Graph answer" not in result

    def test_graphrag_role_key_filters_graphrag_content(self):
        """Test 6: role_key='graphrag_content' chỉ lấy assistant_graphrag."""
        turns = [
            {"turn_id": "t1", "role": "user", "content": "Q1", "timestamp": ""},
            {"turn_id": "t1", "role": "assistant_rag", "content": "RAG answer", "timestamp": ""},
            {"turn_id": "t1", "role": "assistant_graphrag", "content": "Graph answer", "timestamp": ""},
        ]
        result = format_history_block(turns, role_key="graphrag_content")
        assert "Graph answer" in result
        assert "RAG answer" not in result

    def test_preserves_chronological_order(self):
        """Test 7: Thứ tự turns được bảo toàn (oldest first)."""
        turns = make_turns([
            ("Câu hỏi đầu tiên", "Trả lời đầu tiên"),
            ("Câu hỏi thứ hai", "Trả lời thứ hai"),
        ])
        result = format_history_block(turns, role_key="rag_content")
        pos_first = result.index("Câu hỏi đầu tiên")
        pos_second = result.index("Câu hỏi thứ hai")
        assert pos_first < pos_second

    def test_curly_braces_in_content_not_raising(self):
        """Test 8: Content có { } không gây lỗi khi dùng với ChatPromptTemplate."""
        turns = make_turns([("{tên người}", "Xin chào {bạn}!")])
        # Không raise exception
        result = format_history_block(turns, role_key="rag_content")
        assert "tên người" in result or "{tên người}" in result


# ─── Tests: PaC_template_with_history ───────────────────────────────────────

class TestPaCTemplateWithHistory:
    def test_empty_history_no_section_header(self):
        """Test 9: Khi history rỗng, không có header lịch sử trong prompt."""
        template = PaC_template_with_history("some context", "")
        assert "=== Lịch sử hội thoại ===" not in template

    def test_query_placeholder_preserved(self):
        """Test 10: {query} placeholder vẫn còn trong output."""
        template = PaC_template_with_history("context", "")
        assert "{query}" in template

    def test_history_section_present_when_non_empty(self):
        """Test 11: Khi có history, header/footer xuất hiện."""
        template = PaC_template_with_history("context", "Người dùng: Q\nTrợ lý: A")
        assert "=== Lịch sử hội thoại ===" in template
        assert "=== Kết thúc lịch sử ===" in template

    def test_history_block_before_question(self):
        """Test 12 (Property 3): history_block xuất hiện trước 'Câu hỏi:'."""
        history = "Người dùng: Q\nTrợ lý: A"
        template = PaC_template_with_history("context", history)
        pos_history = template.index("Người dùng: Q")
        pos_question = template.index("Câu hỏi:")
        assert pos_history < pos_question

    def test_curly_braces_in_history_escaped(self):
        """Test 13: { } trong history được escape, không gây lỗi ChatPromptTemplate."""
        from langchain_core.prompts import ChatPromptTemplate
        history = "Người dùng: {tên}\nTrợ lý: Xin chào {bạn}"
        template = PaC_template_with_history("context", history)
        # Không raise exception khi parse template
        prompt = ChatPromptTemplate.from_template(template)
        assert prompt is not None


# ─── Tests: GraphRAGInternal.build_answer_prompt ────────────────────────────

class TestBuildAnswerPrompt:
    def _make_internal(self):
        """Tạo GraphRAGInternal với mock dependencies."""
        mock_neo4j = MagicMock()
        mock_neo4j.execute_query.return_value = []
        mock_provider = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 768
        mock_config = MagicMock()
        mock_config.graph_rag = None

        with patch.object(GraphRAGInternal, "_ensure_indexes"):
            internal = GraphRAGInternal(
                provider=mock_provider,
                embedding=mock_embedding,
                neo4j_store=mock_neo4j,
                config=mock_config,
            )
        return internal

    def test_backward_compatible_no_history(self):
        """Test 14: Gọi không có history_block không raise exception."""
        internal = self._make_internal()
        prompt = internal.build_answer_prompt(
            doc_summary_parts=["summary"],
            section_summaries=["section"],
            graph_facts=["fact"],
            doc_passages=[{"content": "passage", "filename": "doc.pdf", "pages": [1]}],
            question="Câu hỏi?",
        )
        assert "Câu hỏi?" in prompt
        assert "=== Lịch sử hội thoại ===" not in prompt

    def test_history_block_before_question(self):
        """Test 15 (Property 4): history_block xuất hiện trước 'Câu hỏi:'."""
        internal = self._make_internal()
        history = "Người dùng: Q trước\nTrợ lý: A trước"
        prompt = internal.build_answer_prompt(
            doc_summary_parts=["summary"],
            section_summaries=[],
            graph_facts=[],
            doc_passages=[{"content": "passage", "filename": "doc.pdf", "pages": [1]}],
            question="Câu hỏi mới?",
            history_block=history,
        )
        pos_history = prompt.index("Q trước")
        pos_question = prompt.index("Câu hỏi mới?")
        assert pos_history < pos_question

    def test_empty_history_no_section_header(self):
        """Test 16: history_block="" → không có header lịch sử."""
        internal = self._make_internal()
        prompt = internal.build_answer_prompt(
            doc_summary_parts=[],
            section_summaries=[],
            graph_facts=[],
            doc_passages=[],
            question="Q?",
            history_block="",
        )
        assert "=== Lịch sử hội thoại ===" not in prompt

    def test_curly_braces_in_history_not_raising(self):
        """Test 17: { } trong history không gây lỗi str.format()."""
        internal = self._make_internal()
        history = "Người dùng: {tên}\nTrợ lý: Xin chào {bạn}"
        # Không raise KeyError
        prompt = internal.build_answer_prompt(
            doc_summary_parts=[],
            section_summaries=[],
            graph_facts=[],
            doc_passages=[],
            question="Q?",
            history_block=history,
        )
        assert prompt is not None


# ─── Tests: BaseRAG._get_history_limit ──────────────────────────────────────

class TestGetHistoryLimit:
    def test_default_value_when_no_config(self):
        """Test 18: Khi config không có conversational_rag, default=5."""
        from src.Features.LangChainAPI.RAG.BaseRAG import BaseRAG

        class ConcreteRAG(BaseRAG):
            async def index(self, file, **kwargs): pass
            async def retrieve(self, query, **kwargs): pass
            async def delete(self, identifier, **kwargs): pass

        mock_provider = MagicMock()
        mock_embedding = MagicMock()

        with patch("src.Features.LangChainAPI.RAG.BaseRAG.load_env_yaml") as mock_yaml:
            mock_config = MagicMock()
            mock_config.conversational_rag = None
            mock_yaml.return_value = mock_config

            rag = ConcreteRAG(mock_provider, mock_embedding)
            assert rag._get_history_limit() == 5

    def test_reads_from_config(self):
        """Test 19: Đọc đúng giá trị từ config."""
        from src.Features.LangChainAPI.RAG.BaseRAG import BaseRAG

        class ConcreteRAG(BaseRAG):
            async def index(self, file, **kwargs): pass
            async def retrieve(self, query, **kwargs): pass
            async def delete(self, identifier, **kwargs): pass

        mock_provider = MagicMock()
        mock_embedding = MagicMock()

        with patch("src.Features.LangChainAPI.RAG.BaseRAG.load_env_yaml") as mock_yaml:
            mock_conv_cfg = MagicMock()
            mock_conv_cfg.conversation_history_limit = 3
            mock_config = MagicMock()
            mock_config.conversational_rag = mock_conv_cfg
            mock_yaml.return_value = mock_config

            rag = ConcreteRAG(mock_provider, mock_embedding)
            assert rag._get_history_limit() == 3

    def test_zero_disables_history(self):
        """Test 20: limit=0 → disable history."""
        from src.Features.LangChainAPI.RAG.BaseRAG import BaseRAG

        class ConcreteRAG(BaseRAG):
            async def index(self, file, **kwargs): pass
            async def retrieve(self, query, **kwargs): pass
            async def delete(self, identifier, **kwargs): pass

        mock_provider = MagicMock()
        mock_embedding = MagicMock()

        with patch("src.Features.LangChainAPI.RAG.BaseRAG.load_env_yaml") as mock_yaml:
            mock_conv_cfg = MagicMock()
            mock_conv_cfg.conversation_history_limit = 0
            mock_config = MagicMock()
            mock_config.conversational_rag = mock_conv_cfg
            mock_yaml.return_value = mock_config

            rag = ConcreteRAG(mock_provider, mock_embedding)
            assert rag._get_history_limit() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
