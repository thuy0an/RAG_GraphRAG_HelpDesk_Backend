"""
Property-based tests cho LLMReranker.

**Validates: Requirements CP-2, CP-3, CP-6**

CP-2: ∀ score ∈ reranking_scores: 0.0 ≤ score ≤ 10.0
CP-3: len(reranked_docs) ≤ min(len(input_docs), top_k)
CP-6: Lỗi một passage không ảnh hưởng các passage khác (score = 0.0)
"""
import asyncio
import sys
import os
import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Thêm src vào path để import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from Features.LangChainAPI.RAG.LLMReranker import LLMReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reranker(llm_response_text: str) -> LLMReranker:
    """Create an LLMReranker with a mock provider returning the given text."""
    mock_response = MagicMock()
    mock_response.content = llm_response_text
    provider = MagicMock()
    provider.ainvoke = AsyncMock(return_value=mock_response)
    return LLMReranker(provider)


def make_reranker_with_error(error: Exception) -> LLMReranker:
    """Create an LLMReranker whose provider always raises an exception."""
    provider = MagicMock()
    provider.ainvoke = AsyncMock(side_effect=error)
    return LLMReranker(provider)


def make_docs(n: int) -> List[Dict[str, Any]]:
    return [{"content": f"passage {i}", "metadata": {}} for i in range(n)]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Arbitrary LLM output strings (may or may not contain a number)
llm_output_strategy = st.text(min_size=0, max_size=200)

# Docs list strategy
doc_strategy = st.fixed_dictionaries({
    "content": st.text(max_size=300),
    "metadata": st.fixed_dictionaries({"source": st.text(max_size=50)}),
})

docs_list_strategy = st.lists(doc_strategy, min_size=0, max_size=20)

# top_k strategy
top_k_strategy = st.integers(min_value=1, max_value=15)


# ---------------------------------------------------------------------------
# CP-2: Re-ranking scores are always in [0.0, 10.0]
# ---------------------------------------------------------------------------

@given(llm_output=llm_output_strategy, n_docs=st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_cp2_scores_within_bounds(llm_output: str, n_docs: int):
    """**Validates: Requirements CP-2**

    For any LLM output string, all reranking scores must be in [0.0, 10.0].
    """
    reranker = make_reranker(llm_output)
    docs = make_docs(n_docs)

    _, scores, _ = asyncio.run(reranker.rerank("test query", docs, top_k=n_docs))

    for score in scores:
        assert 0.0 <= score <= 10.0, (
            f"Score {score} is out of bounds [0.0, 10.0] for LLM output: {llm_output!r}"
        )


@given(raw_score=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_cp2_score_clamping(raw_score: float):
    """**Validates: Requirements CP-2**

    Even if the LLM returns a number outside [0, 10], the score must be clamped.
    """
    llm_output = str(raw_score)
    reranker = make_reranker(llm_output)
    docs = make_docs(1)

    _, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=1))

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 10.0, f"Score {scores[0]} not clamped for raw={raw_score}"


# ---------------------------------------------------------------------------
# CP-3: len(reranked_docs) ≤ min(len(input_docs), top_k)
# ---------------------------------------------------------------------------

@given(docs=docs_list_strategy, top_k=top_k_strategy)
@settings(max_examples=200)
def test_cp3_reranked_length_bounded(docs: List[Dict[str, Any]], top_k: int):
    """**Validates: Requirements CP-3**

    len(reranked_docs) must always be ≤ min(len(input_docs), top_k).
    """
    reranker = make_reranker("5")
    reranked, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=top_k))

    expected_max = min(len(docs), top_k)
    assert len(reranked) <= expected_max, (
        f"Got {len(reranked)} docs but expected at most {expected_max} "
        f"(input={len(docs)}, top_k={top_k})"
    )
    assert len(reranked) == len(scores), (
        "reranked_docs and scores must have the same length"
    )


@given(top_k=top_k_strategy)
@settings(max_examples=100)
def test_cp3_empty_input_returns_empty(top_k: int):
    """**Validates: Requirements CP-3**

    Empty input docs must always return empty results regardless of top_k.
    """
    reranker = make_reranker("7")
    reranked, scores, elapsed = asyncio.run(reranker.rerank("query", [], top_k=top_k))

    assert reranked == []
    assert scores == []
    assert elapsed == 0.0


# ---------------------------------------------------------------------------
# CP-6: Error on one passage does not affect others
# ---------------------------------------------------------------------------

def test_cp6_single_error_returns_zero_score():
    """**Validates: Requirements CP-6**

    When the LLM raises an exception for a passage, that passage gets score 0.0
    and other passages are still scored normally.
    """
    call_count = 0
    responses = ["8", "error_here", "6"]  # second call will raise

    async def mock_ainvoke(prompt):
        nonlocal call_count
        idx = call_count
        call_count += 1
        if idx == 1:
            raise RuntimeError("LLM timeout")
        mock_resp = MagicMock()
        mock_resp.content = responses[idx]
        return mock_resp

    provider = MagicMock()
    provider.ainvoke = mock_ainvoke
    reranker = LLMReranker(provider)

    docs = make_docs(3)
    reranked, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=3))

    # All 3 docs should be returned (top_k=3)
    assert len(reranked) == 3
    assert len(scores) == 3

    # The errored passage gets 0.0; others get their parsed scores
    all_scores_set = set(scores)
    assert 0.0 in all_scores_set, "Errored passage should have score 0.0"
    assert 8.0 in all_scores_set, "First passage should have score 8.0"
    assert 6.0 in all_scores_set, "Third passage should have score 6.0"


@given(
    n_docs=st.integers(min_value=2, max_value=10),
    error_index=st.integers(min_value=0, max_value=9),
)
@settings(max_examples=100)
def test_cp6_error_passage_gets_zero_others_unaffected(n_docs: int, error_index: int):
    """**Validates: Requirements CP-6**

    For any input size, an exception on one passage yields score 0.0 for that
    passage while all other passages still receive valid scores in [0.0, 10.0].
    """
    if error_index >= n_docs:
        error_index = n_docs - 1

    call_count_holder = [0]

    async def selective_error_ainvoke(prompt):
        idx = call_count_holder[0]
        call_count_holder[0] += 1
        if idx == error_index:
            raise ValueError("Simulated LLM error")
        mock_resp = MagicMock()
        mock_resp.content = "7"
        return mock_resp

    provider = MagicMock()
    provider.ainvoke = selective_error_ainvoke
    reranker = LLMReranker(provider)

    docs = make_docs(n_docs)
    reranked, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=n_docs))

    assert len(scores) == n_docs, "All docs should be scored"
    for score in scores:
        assert 0.0 <= score <= 10.0, f"Score {score} out of bounds"


# ---------------------------------------------------------------------------
# Unit tests – example-based sanity checks
# ---------------------------------------------------------------------------

def test_rerank_sorts_by_score_descending():
    """Docs should be sorted by score descending."""
    responses = ["3", "9", "1", "7"]
    call_idx = [0]

    async def mock_ainvoke(prompt):
        idx = call_idx[0]
        call_idx[0] += 1
        mock_resp = MagicMock()
        mock_resp.content = responses[idx]
        return mock_resp

    provider = MagicMock()
    provider.ainvoke = mock_ainvoke
    reranker = LLMReranker(provider)

    docs = [{"content": f"doc{i}", "metadata": {}} for i in range(4)]
    reranked, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=4))

    assert scores == sorted(scores, reverse=True), "Scores must be in descending order"
    assert scores[0] == 9.0
    assert scores[1] == 7.0
    assert scores[2] == 3.0
    assert scores[3] == 1.0


def test_rerank_top_k_limits_output():
    """top_k should limit the number of returned docs."""
    reranker = make_reranker("5")
    docs = make_docs(10)
    reranked, scores, _ = asyncio.run(reranker.rerank("query", docs, top_k=3))

    assert len(reranked) == 3
    assert len(scores) == 3


def test_rerank_elapsed_is_non_negative():
    """Elapsed time should always be >= 0."""
    reranker = make_reranker("5")
    docs = make_docs(3)
    _, _, elapsed = asyncio.run(reranker.rerank("query", docs, top_k=3))
    assert elapsed >= 0.0


def test_score_one_no_number_returns_zero():
    """When LLM returns text with no number, score should be 0.0."""
    reranker = make_reranker("không liên quan")
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 0.0


def test_score_one_valid_integer():
    """When LLM returns a valid integer string, score should match."""
    reranker = make_reranker("8")
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 8.0


def test_score_one_clamps_above_10():
    """Score above 10 should be clamped to 10.0."""
    reranker = make_reranker("15")
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 10.0


def test_score_one_no_parseable_number_returns_zero():
    """When LLM returns only non-numeric text, score should be 0.0."""
    reranker = make_reranker("không có số nào")
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 0.0


def test_score_one_large_number_clamped_to_10():
    """When LLM returns a number > 10, it should be clamped to 10.0."""
    reranker = make_reranker("100")
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 10.0


def test_score_one_exception_returns_zero():
    """Exception from provider should return 0.0."""
    reranker = make_reranker_with_error(RuntimeError("network error"))
    score = asyncio.run(reranker._score_one("query", "passage"))
    assert score == 0.0
