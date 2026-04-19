"""
Unit tests cho MemoryRepository.
Dùng SQLite in-memory để không cần file thật.
Chạy: pytest src/tests/test_memory_repository.py -v
"""
import asyncio
import pytest
from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository


@pytest.fixture
def repo(tmp_path):
    """MemoryRepository dùng file SQLite tạm trong tmp_path."""
    db_file = tmp_path / "test_chat.db"
    return MemoryRepository(db_path=str(db_file))


# ─────────────────────────────────────────────
# begin_turn
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_begin_turn_returns_turn_id(repo):
    turn_id = await repo.begin_turn("session_1", "Xin chào")
    assert isinstance(turn_id, str)
    assert len(turn_id) > 0


@pytest.mark.asyncio
async def test_begin_turn_multiple_sessions(repo):
    id1 = await repo.begin_turn("session_A", "Câu hỏi 1")
    id2 = await repo.begin_turn("session_B", "Câu hỏi 2")
    assert id1 != id2


@pytest.mark.asyncio
async def test_begin_turn_same_session_different_ids(repo):
    id1 = await repo.begin_turn("session_1", "Câu hỏi 1")
    id2 = await repo.begin_turn("session_1", "Câu hỏi 2")
    assert id1 != id2


# ─────────────────────────────────────────────
# update_rag / update_graphrag
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_rag_stores_answer(repo):
    turn_id = await repo.begin_turn("session_1", "Hỏi về RAG")
    await repo.update_rag(turn_id, "Đây là câu trả lời RAG")

    msgs = await repo.get_history_all("session_1")
    rag_msgs = [m for m in msgs if m["role"] == "assistant_rag"]
    assert len(rag_msgs) == 1
    assert rag_msgs[0]["content"] == "Đây là câu trả lời RAG"


@pytest.mark.asyncio
async def test_update_graphrag_stores_answer(repo):
    turn_id = await repo.begin_turn("session_1", "Hỏi về GraphRAG")
    await repo.update_graphrag(turn_id, "Đây là câu trả lời GraphRAG")

    msgs = await repo.get_history_all("session_1")
    graph_msgs = [m for m in msgs if m["role"] == "assistant_graphrag"]
    assert len(graph_msgs) == 1
    assert graph_msgs[0]["content"] == "Đây là câu trả lời GraphRAG"


@pytest.mark.asyncio
async def test_update_rag_and_graphrag_independent(repo):
    """Cả hai update độc lập trên cùng 1 turn."""
    turn_id = await repo.begin_turn("session_1", "Câu hỏi")
    await repo.update_rag(turn_id, "RAG answer")
    await repo.update_graphrag(turn_id, "GraphRAG answer")

    msgs = await repo.get_history_all("session_1")
    roles = {m["role"] for m in msgs}
    assert "user" in roles
    assert "assistant_rag" in roles
    assert "assistant_graphrag" in roles


# ─────────────────────────────────────────────
# get_history_all
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_history_all_empty_session(repo):
    msgs = await repo.get_history_all("nonexistent_session")
    assert msgs == []


@pytest.mark.asyncio
async def test_get_history_all_returns_user_message(repo):
    await repo.begin_turn("session_1", "Câu hỏi của user")
    msgs = await repo.get_history_all("session_1")
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Câu hỏi của user"


@pytest.mark.asyncio
async def test_get_history_all_role_filter_rag(repo):
    """role_filter='assistant_rag' chỉ trả về turn có rag_content."""
    t1 = await repo.begin_turn("session_1", "Q1")
    await repo.update_rag(t1, "RAG 1")

    t2 = await repo.begin_turn("session_1", "Q2")
    await repo.update_graphrag(t2, "Graph 2")  # không có rag

    msgs = await repo.get_history_all("session_1", role_filter="assistant_rag")
    rag_msgs = [m for m in msgs if m["role"] == "assistant_rag"]
    assert len(rag_msgs) == 1
    assert rag_msgs[0]["content"] == "RAG 1"


@pytest.mark.asyncio
async def test_get_history_all_role_filter_graphrag(repo):
    """role_filter='assistant_graphrag' chỉ trả về turn có graphrag_content."""
    t1 = await repo.begin_turn("session_1", "Q1")
    await repo.update_graphrag(t1, "Graph 1")

    t2 = await repo.begin_turn("session_1", "Q2")
    await repo.update_rag(t2, "RAG 2")  # không có graphrag

    msgs = await repo.get_history_all("session_1", role_filter="assistant_graphrag")
    graph_msgs = [m for m in msgs if m["role"] == "assistant_graphrag"]
    assert len(graph_msgs) == 1
    assert graph_msgs[0]["content"] == "Graph 1"


@pytest.mark.asyncio
async def test_get_history_all_session_isolation(repo):
    """Lịch sử của session A không lẫn vào session B."""
    await repo.begin_turn("session_A", "Câu hỏi A")
    await repo.begin_turn("session_B", "Câu hỏi B")

    msgs_a = await repo.get_history_all("session_A")
    msgs_b = await repo.get_history_all("session_B")

    contents_a = {m["content"] for m in msgs_a}
    contents_b = {m["content"] for m in msgs_b}
    assert "Câu hỏi A" in contents_a
    assert "Câu hỏi B" not in contents_a
    assert "Câu hỏi B" in contents_b
    assert "Câu hỏi A" not in contents_b


# ─────────────────────────────────────────────
# get_recent_messages
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recent_messages_limit(repo):
    """Chỉ trả về N turn gần nhất."""
    for i in range(5):
        t = await repo.begin_turn("session_1", f"Q{i}")
        await repo.update_rag(t, f"A{i}")

    msgs = await repo.get_recent_messages("session_1", limit=2)
    # 2 turns × 2 messages (user + rag) = 4
    assert len(msgs) == 4


@pytest.mark.asyncio
async def test_get_recent_messages_oldest_first(repo):
    """Kết quả phải theo thứ tự oldest first (reversed từ DESC query)."""
    t1 = await repo.begin_turn("session_1", "First")
    await repo.update_rag(t1, "Answer 1")
    t2 = await repo.begin_turn("session_1", "Second")
    await repo.update_rag(t2, "Answer 2")

    msgs = await repo.get_recent_messages("session_1", limit=10)
    user_msgs = [m for m in msgs if m["role"] == "user"]
    # Cả 2 message phải có mặt
    contents = [m["content"] for m in user_msgs]
    assert "First" in contents
    assert "Second" in contents


@pytest.mark.asyncio
async def test_get_recent_messages_empty(repo):
    msgs = await repo.get_recent_messages("no_session", limit=5)
    assert msgs == []


# ─────────────────────────────────────────────
# delete_session_history
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_session_history_returns_count(repo):
    await repo.begin_turn("session_1", "Q1")
    await repo.begin_turn("session_1", "Q2")
    deleted = await repo.delete_session_history("session_1")
    assert deleted == 2


@pytest.mark.asyncio
async def test_delete_session_history_clears_data(repo):
    await repo.begin_turn("session_1", "Q1")
    await repo.delete_session_history("session_1")
    msgs = await repo.get_history_all("session_1")
    assert msgs == []


@pytest.mark.asyncio
async def test_delete_session_history_only_target_session(repo):
    """Xóa session A không ảnh hưởng session B."""
    await repo.begin_turn("session_A", "Q_A")
    await repo.begin_turn("session_B", "Q_B")
    await repo.delete_session_history("session_A")

    msgs_a = await repo.get_history_all("session_A")
    msgs_b = await repo.get_history_all("session_B")
    assert msgs_a == []
    assert len(msgs_b) > 0


@pytest.mark.asyncio
async def test_delete_nonexistent_session_returns_zero(repo):
    deleted = await repo.delete_session_history("ghost_session")
    assert deleted == 0
