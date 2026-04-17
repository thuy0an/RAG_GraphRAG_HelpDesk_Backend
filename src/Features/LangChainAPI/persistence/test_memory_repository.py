"""
Unit tests cho MemoryRepository với schema mới.

Schema: mỗi row = 1 lượt hỏi-đáp
  - user_content    : câu hỏi của user
  - rag_content     : trả lời PaCRAG (upsert độc lập)
  - graphrag_content: trả lời GraphRAG (upsert độc lập)

Chạy: pytest src/Features/LangChainAPI/persistence/test_memory_repository.py -v
"""

import asyncio
import pytest
import tempfile
import os

from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository


# ─── Fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Tạo DB tạm thời cho mỗi test, tự xóa sau khi xong."""
    db_file = tmp_path / "test_chat_history.db"
    return str(db_file)


@pytest.fixture
def repo(tmp_db):
    return MemoryRepository(db_path=tmp_db)


def run(coro):
    """Helper chạy coroutine trong test đồng bộ."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Tests: begin_turn ───────────────────────────────────────────────────────

class TestBeginTurn:
    def test_begin_turn_returns_turn_id(self, repo):
        """Test 1: begin_turn trả về turn_id hợp lệ (non-empty string)."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi đầu tiên"))
        assert isinstance(turn_id, str)
        assert len(turn_id) > 0

    def test_begin_turn_creates_row_with_user_content(self, repo):
        """Test 2: Row được tạo với user_content đúng, rag/graphrag là NULL."""
        turn_id = run(repo.begin_turn("session-1", "xin chào"))
        messages = run(repo.get_history_all("session-1"))

        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "xin chào"
        assert user_msgs[0]["turn_id"] == turn_id

    def test_begin_turn_no_rag_or_graphrag_initially(self, repo):
        """Test 3: Sau begin_turn, không có rag hay graphrag message."""
        run(repo.begin_turn("session-1", "câu hỏi"))
        messages = run(repo.get_history_all("session-1"))

        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]
        assert len(rag_msgs) == 0
        assert len(graph_msgs) == 0

    def test_begin_turn_multiple_turns_same_session(self, repo):
        """Test 4: Nhiều turn trong cùng session → nhiều user message."""
        run(repo.begin_turn("session-1", "câu hỏi 1"))
        run(repo.begin_turn("session-1", "câu hỏi 2"))
        run(repo.begin_turn("session-1", "câu hỏi 3"))

        messages = run(repo.get_history_all("session-1"))
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 3

    def test_begin_turn_different_sessions_isolated(self, repo):
        """Test 5: Các session khác nhau không ảnh hưởng lẫn nhau."""
        run(repo.begin_turn("session-A", "hỏi A"))
        run(repo.begin_turn("session-B", "hỏi B"))

        msgs_a = run(repo.get_history_all("session-A"))
        msgs_b = run(repo.get_history_all("session-B"))

        assert len([m for m in msgs_a if m["role"] == "user"]) == 1
        assert len([m for m in msgs_b if m["role"] == "user"]) == 1
        assert msgs_a[0]["content"] == "hỏi A"
        assert msgs_b[0]["content"] == "hỏi B"


# ─── Tests: update_rag ───────────────────────────────────────────────────────

class TestUpdateRag:
    def test_update_rag_sets_rag_content(self, repo):
        """Test 6: update_rag cập nhật rag_content đúng turn."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_rag(turn_id, "câu trả lời PaCRAG"))

        messages = run(repo.get_history_all("session-1"))
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        assert len(rag_msgs) == 1
        assert rag_msgs[0]["content"] == "câu trả lời PaCRAG"
        assert rag_msgs[0]["turn_id"] == turn_id

    def test_update_rag_does_not_affect_graphrag(self, repo):
        """Test 7: update_rag không tạo graphrag_content."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_rag(turn_id, "rag answer"))

        messages = run(repo.get_history_all("session-1"))
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]
        assert len(graph_msgs) == 0

    def test_update_rag_only_updates_correct_turn(self, repo):
        """Test 8: update_rag chỉ update đúng turn, không ảnh hưởng turn khác."""
        turn1 = run(repo.begin_turn("session-1", "câu hỏi 1"))
        turn2 = run(repo.begin_turn("session-1", "câu hỏi 2"))

        run(repo.update_rag(turn1, "rag answer 1"))

        messages = run(repo.get_history_all("session-1"))
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        assert len(rag_msgs) == 1
        assert rag_msgs[0]["turn_id"] == turn1
        assert rag_msgs[0]["content"] == "rag answer 1"


# ─── Tests: update_graphrag ──────────────────────────────────────────────────

class TestUpdateGraphrag:
    def test_update_graphrag_sets_graphrag_content(self, repo):
        """Test 9: update_graphrag cập nhật graphrag_content đúng turn."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_graphrag(turn_id, "câu trả lời GraphRAG"))

        messages = run(repo.get_history_all("session-1"))
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]
        assert len(graph_msgs) == 1
        assert graph_msgs[0]["content"] == "câu trả lời GraphRAG"
        assert graph_msgs[0]["turn_id"] == turn_id

    def test_update_graphrag_does_not_affect_rag(self, repo):
        """Test 10: update_graphrag không tạo rag_content."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_graphrag(turn_id, "graphrag answer"))

        messages = run(repo.get_history_all("session-1"))
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        assert len(rag_msgs) == 0

    def test_update_graphrag_only_updates_correct_turn(self, repo):
        """Test 11: update_graphrag chỉ update đúng turn."""
        turn1 = run(repo.begin_turn("session-1", "câu hỏi 1"))
        turn2 = run(repo.begin_turn("session-1", "câu hỏi 2"))

        run(repo.update_graphrag(turn2, "graphrag answer 2"))

        messages = run(repo.get_history_all("session-1"))
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]
        assert len(graph_msgs) == 1
        assert graph_msgs[0]["turn_id"] == turn2


# ─── Tests: async bất đồng bộ ────────────────────────────────────────────────

class TestAsyncIndependence:
    def test_rag_and_graphrag_update_same_turn_independently(self, repo):
        """Test 12: PaCRAG và GraphRAG cùng update 1 turn — không conflict."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))

        # Simulate bất đồng bộ: update cả 2 cùng turn_id
        async def both_updates():
            await asyncio.gather(
                repo.update_rag(turn_id, "rag answer"),
                repo.update_graphrag(turn_id, "graphrag answer"),
            )

        run(both_updates())

        messages = run(repo.get_history_all("session-1"))
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]

        assert len(rag_msgs) == 1
        assert len(graph_msgs) == 1
        assert rag_msgs[0]["content"] == "rag answer"
        assert graph_msgs[0]["content"] == "graphrag answer"

    def test_only_rag_updates_graphrag_null(self, repo):
        """Test 13: Chỉ PaCRAG update → graphrag_content vẫn NULL (không có message)."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_rag(turn_id, "rag only"))

        messages = run(repo.get_history_all("session-1"))
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]
        assert len(graph_msgs) == 0

    def test_only_graphrag_updates_rag_null(self, repo):
        """Test 14: Chỉ GraphRAG update → rag_content vẫn NULL (không có message)."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_graphrag(turn_id, "graphrag only"))

        messages = run(repo.get_history_all("session-1"))
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        assert len(rag_msgs) == 0

    def test_no_duplicate_user_message_across_turns(self, repo):
        """Test 15: Nhiều turn → user message không bị duplicate."""
        for i in range(5):
            turn_id = run(repo.begin_turn("session-1", f"câu hỏi {i}"))
            run(repo.update_rag(turn_id, f"rag {i}"))
            run(repo.update_graphrag(turn_id, f"graphrag {i}"))

        messages = run(repo.get_history_all("session-1"))
        user_msgs = [m for m in messages if m["role"] == "user"]
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]

        # Đúng 5 user, 5 rag, 5 graphrag — không duplicate
        assert len(user_msgs) == 5
        assert len(rag_msgs) == 5
        assert len(graph_msgs) == 5


# ─── Tests: role_filter ──────────────────────────────────────────────────────

class TestRoleFilter:
    def test_role_filter_assistant_rag_returns_user_and_rag(self, repo):
        """Test 16: role_filter='assistant_rag' → chỉ trả về turn có rag_content."""
        turn1 = run(repo.begin_turn("session-1", "câu hỏi 1"))
        run(repo.update_rag(turn1, "rag 1"))
        run(repo.update_graphrag(turn1, "graphrag 1"))

        turn2 = run(repo.begin_turn("session-1", "câu hỏi 2"))
        # turn2 chỉ có graphrag, không có rag
        run(repo.update_graphrag(turn2, "graphrag 2"))

        messages = run(repo.get_history_all("session-1", role_filter="assistant_rag"))

        # Chỉ turn1 có rag → chỉ turn1 xuất hiện
        user_msgs = [m for m in messages if m["role"] == "user"]
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]

        assert len(user_msgs) == 1
        assert len(rag_msgs) == 1
        assert len(graph_msgs) == 0

    def test_role_filter_assistant_graphrag_returns_user_and_graphrag(self, repo):
        """Test 17: role_filter='assistant_graphrag' → chỉ trả về turn có graphrag_content."""
        turn1 = run(repo.begin_turn("session-1", "câu hỏi 1"))
        run(repo.update_rag(turn1, "rag 1"))
        # turn1 không có graphrag

        turn2 = run(repo.begin_turn("session-1", "câu hỏi 2"))
        run(repo.update_rag(turn2, "rag 2"))
        run(repo.update_graphrag(turn2, "graphrag 2"))

        messages = run(repo.get_history_all("session-1", role_filter="assistant_graphrag"))

        user_msgs = [m for m in messages if m["role"] == "user"]
        rag_msgs = [m for m in messages if m["role"] == "assistant_rag"]
        graph_msgs = [m for m in messages if m["role"] == "assistant_graphrag"]

        assert len(user_msgs) == 1
        assert len(rag_msgs) == 0
        assert len(graph_msgs) == 1

    def test_no_role_filter_returns_all(self, repo):
        """Test 18: Không filter → trả về tất cả messages."""
        turn_id = run(repo.begin_turn("session-1", "câu hỏi"))
        run(repo.update_rag(turn_id, "rag"))
        run(repo.update_graphrag(turn_id, "graphrag"))

        messages = run(repo.get_history_all("session-1"))
        roles = {m["role"] for m in messages}
        assert "user" in roles
        assert "assistant_rag" in roles
        assert "assistant_graphrag" in roles


# ─── Tests: delete_session_history ───────────────────────────────────────────

class TestDeleteHistory:
    def test_delete_session_history_removes_all_turns(self, repo):
        """Test 19: Xóa session → không còn message nào."""
        for i in range(3):
            t = run(repo.begin_turn("session-1", f"câu hỏi {i}"))
            run(repo.update_rag(t, f"rag {i}"))

        deleted = run(repo.delete_session_history("session-1"))
        assert deleted == 3

        messages = run(repo.get_history_all("session-1"))
        assert len(messages) == 0

    def test_delete_session_does_not_affect_other_sessions(self, repo):
        """Test 20: Xóa session A không ảnh hưởng session B."""
        t_a = run(repo.begin_turn("session-A", "hỏi A"))
        run(repo.update_rag(t_a, "rag A"))

        t_b = run(repo.begin_turn("session-B", "hỏi B"))
        run(repo.update_rag(t_b, "rag B"))

        run(repo.delete_session_history("session-A"))

        msgs_a = run(repo.get_history_all("session-A"))
        msgs_b = run(repo.get_history_all("session-B"))

        assert len(msgs_a) == 0
        assert len([m for m in msgs_b if m["role"] == "user"]) == 1


# ─── Tests: get_recent_messages ──────────────────────────────────────────────

class TestGetRecentMessages:
    def test_get_recent_messages_respects_limit(self, repo):
        """Test 21: get_recent_messages trả về đúng số turn theo limit."""
        for i in range(5):
            t = run(repo.begin_turn("session-1", f"câu hỏi {i}"))
            run(repo.update_rag(t, f"rag {i}"))

        # limit=2 → 2 turn gần nhất → tối đa 4 messages (2 user + 2 rag)
        messages = run(repo.get_recent_messages("session-1", limit=2))
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_get_recent_messages_chronological_order(self, repo):
        """Test 22: get_recent_messages trả về theo thứ tự thời gian tăng dần."""
        t1 = run(repo.begin_turn("session-1", "câu hỏi 1"))
        t2 = run(repo.begin_turn("session-1", "câu hỏi 2"))

        messages = run(repo.get_recent_messages("session-1", limit=10))
        user_msgs = [m for m in messages if m["role"] == "user"]

        assert user_msgs[0]["content"] == "câu hỏi 1"
        assert user_msgs[1]["content"] == "câu hỏi 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
