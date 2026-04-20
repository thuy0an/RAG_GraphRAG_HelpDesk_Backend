"""
Unit tests cho CompareRepository — lưu trữ và truy vấn compare runs.
Dùng SQLite in-memory.
Chạy: pytest src/tests/test_compare_entities.py -v
"""
import pytest
from src.Features.LangChainAPI.persistence.CompareRepository import CompareRepository


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_compare.db"
    return CompareRepository(db_path=str(db_file))


# ─────────────────────────────────────────────
# create_run
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_run_returns_dict(repo):
    run = await repo.create_run(
        session_id="s1",
        file_name="test.pdf",
        file_type="application/pdf",
        file_size=1024,
        pac_ingest={"time_total_s": 1.2, "parent_chunks": 5, "child_chunks": 20},
        graphrag_ingest={"nodes": 10, "edges": 15},
    )
    assert isinstance(run, dict)
    assert "id" in run
    assert run["file_name"] == "test.pdf"
    assert run["session_id"] == "s1"


@pytest.mark.asyncio
async def test_create_run_unique_ids(repo):
    r1 = await repo.create_run("s1", "a.pdf", None, None, {}, {})
    r2 = await repo.create_run("s1", "b.pdf", None, None, {}, {})
    assert r1["id"] != r2["id"]


@pytest.mark.asyncio
async def test_create_run_with_errors(repo):
    run = await repo.create_run(
        session_id="s1",
        file_name="bad.pdf",
        file_type="application/pdf",
        file_size=0,
        pac_ingest={},
        graphrag_ingest={},
        errors={"graphrag": "Connection failed"},
    )
    assert run is not None


# ─────────────────────────────────────────────
# list_runs
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_runs_empty_session(repo):
    runs = await repo.list_runs("nonexistent")
    assert runs == []


@pytest.mark.asyncio
async def test_list_runs_returns_correct_session(repo):
    await repo.create_run("session_A", "a.pdf", None, None, {}, {})
    await repo.create_run("session_B", "b.pdf", None, None, {}, {})

    runs_a = await repo.list_runs("session_A")
    runs_b = await repo.list_runs("session_B")

    assert len(runs_a) == 1
    assert runs_a[0]["file_name"] == "a.pdf"
    assert len(runs_b) == 1
    assert runs_b[0]["file_name"] == "b.pdf"


@pytest.mark.asyncio
async def test_list_runs_multiple_files(repo):
    for i in range(3):
        await repo.create_run("session_1", f"file{i}.pdf", None, None, {}, {})
    runs = await repo.list_runs("session_1")
    assert len(runs) == 3


# ─────────────────────────────────────────────
# update_query_metrics
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_query_metrics(repo):
    run = await repo.create_run("s1", "test.pdf", None, None, {}, {})
    run_id = run["id"]

    updated = await repo.update_query_metrics(
        run_id,
        pac_query={
            "answer": "PAC answer",
            "time_total_s": 2.1,
            "latency_breakdown": {"hybrid_retrieval": 0.7, "llm_generation": 1.1},
            "metric_groups": {
                "retrieval_metrics": {"retrieved_chunk_count": 4, "retrieved_source_count": 2},
                "system_metrics": {"time_total_s": 2.1},
            },
        },
        graphrag_query={
            "answer": "Graph answer",
            "time_total_s": 3.5,
            "latency_breakdown": {"vector_search": 0.9, "llm_generation": 1.6},
            "metric_groups": {
                "graph_metrics": {"entity_count": 5, "graph_fact_count": 3},
                "generation_metrics": {"confidence_score": 0.8},
            },
        },
        query_text="Câu hỏi test",
    )
    assert updated is not None
    assert updated["query_text"] == "Câu hỏi test"
    assert updated["pac_query"]["latency_breakdown"]["hybrid_retrieval"] == 0.7
    assert updated["pac_query"]["metric_groups"]["retrieval_metrics"]["retrieved_chunk_count"] == 4
    assert updated["graphrag_query"]["metric_groups"]["graph_metrics"]["entity_count"] == 5


# ─────────────────────────────────────────────
# delete_run
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_run_removes_entry(repo):
    run = await repo.create_run("s1", "test.pdf", None, None, {}, {})
    run_id = run["id"]

    deleted = await repo.delete_run(run_id)
    # delete_run trả về rowcount (int) hoặc bool-like truthy
    assert deleted

    runs = await repo.list_runs("s1")
    assert all(r["id"] != run_id for r in runs)


@pytest.mark.asyncio
async def test_delete_nonexistent_run(repo):
    result = await repo.delete_run("nonexistent-id-000")
    # Không có row nào bị xóa → falsy (0 hoặc False)
    assert not result


@pytest.mark.asyncio
async def test_delete_run_only_removes_target(repo):
    r1 = await repo.create_run("s1", "a.pdf", None, None, {}, {})
    r2 = await repo.create_run("s1", "b.pdf", None, None, {}, {})

    await repo.delete_run(r1["id"])
    runs = await repo.list_runs("s1")
    assert len(runs) == 1
    assert runs[0]["id"] == r2["id"]
