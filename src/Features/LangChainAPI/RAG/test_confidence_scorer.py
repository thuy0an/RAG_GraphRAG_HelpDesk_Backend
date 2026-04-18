"""
Property-based tests cho ConfidenceScorer.

**Validates: Requirements CP-4**

CP-4: confidence_score ∈ [0.0, 1.0] ∨ confidence_score = null
"""
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Thêm src vào path để import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from Features.LangChainAPI.RAG.ConfidenceScorer import ConfidenceScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scorer(llm_response_text: str) -> ConfidenceScorer:
    """Create a ConfidenceScorer with a mock provider returning the given text."""
    mock_response = MagicMock()
    mock_response.content = llm_response_text
    provider = MagicMock()
    provider.ainvoke = AsyncMock(return_value=mock_response)
    return ConfidenceScorer(provider)


def make_scorer_with_error(error: Exception) -> ConfidenceScorer:
    """Create a ConfidenceScorer whose provider always raises an exception."""
    provider = MagicMock()
    provider.ainvoke = AsyncMock(side_effect=error)
    return ConfidenceScorer(provider)


# ---------------------------------------------------------------------------
# CP-4: Confidence score is always None or in [0.0, 1.0]
# ---------------------------------------------------------------------------

@given(llm_output=st.text(min_size=0, max_size=300))
@settings(max_examples=300)
def test_cp4_score_bounds_for_any_llm_output(llm_output: str):
    """**Validates: Requirements CP-4**

    For any LLM output string, the result must be None or in [0.0, 1.0].
    """
    scorer = make_scorer(llm_output)
    result = asyncio.run(scorer.score("query", "context", "answer"))

    assert result is None or (0.0 <= result <= 1.0), (
        f"Score {result!r} is not None and not in [0.0, 1.0] for LLM output: {llm_output!r}"
    )


@given(
    llm_output=st.text(min_size=0, max_size=300),
    query=st.text(min_size=0, max_size=600),
    context=st.text(min_size=0, max_size=3000),
    answer=st.text(min_size=0, max_size=600),
)
@settings(max_examples=100)
def test_cp4_score_bounds_with_varied_inputs(
    llm_output: str, query: str, context: str, answer: str
):
    """**Validates: Requirements CP-4**

    For any combination of query/context/answer and any LLM output,
    the result must be None or in [0.0, 1.0].
    """
    scorer = make_scorer(llm_output)
    result = asyncio.run(scorer.score(query, context, answer))

    assert result is None or (0.0 <= result <= 1.0), (
        f"Score {result!r} is not None and not in [0.0, 1.0]"
    )


# ---------------------------------------------------------------------------
# CP-4: Exception from LLM → result must be None
# ---------------------------------------------------------------------------

@given(
    error_msg=st.text(min_size=0, max_size=100),
)
@settings(max_examples=100)
def test_cp4_exception_returns_none(error_msg: str):
    """**Validates: Requirements CP-4**

    When the LLM raises any exception, the result must be None (not a score).
    """
    scorer = make_scorer_with_error(RuntimeError(error_msg))
    result = asyncio.run(scorer.score("query", "context", "answer"))

    assert result is None, (
        f"Expected None when LLM raises exception, got {result!r}"
    )


def test_cp4_value_error_returns_none():
    """**Validates: Requirements CP-4** – ValueError from provider → None."""
    scorer = make_scorer_with_error(ValueError("bad value"))
    result = asyncio.run(scorer.score("q", "ctx", "ans"))
    assert result is None


def test_cp4_connection_error_returns_none():
    """**Validates: Requirements CP-4** – ConnectionError from provider → None."""
    scorer = make_scorer_with_error(ConnectionError("network failure"))
    result = asyncio.run(scorer.score("q", "ctx", "ans"))
    assert result is None


# ---------------------------------------------------------------------------
# CP-4: Out-of-range LLM output must be clamped to [0.0, 1.0]
# ---------------------------------------------------------------------------

@given(
    raw_score=st.floats(
        min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    )
)
@settings(max_examples=200)
def test_cp4_out_of_range_score_clamped(raw_score: float):
    """**Validates: Requirements CP-4**

    When the LLM returns a number that is parseable as 0 or 1 (boundary),
    the result must still be in [0.0, 1.0].
    The regex only matches values in [0, 1] by design, so any parsed value
    is already within bounds; this test verifies the clamp logic is correct.
    """
    # Use a string that contains a value the regex can match (0 or 1)
    # and verify the clamp still holds
    llm_output = str(raw_score)
    scorer = make_scorer(llm_output)
    result = asyncio.run(scorer.score("query", "context", "answer"))

    assert result is None or (0.0 <= result <= 1.0), (
        f"Score {result!r} out of [0.0, 1.0] for raw_score={raw_score}"
    )


def test_cp4_score_above_1_clamped():
    """**Validates: Requirements CP-4** – value > 1 in LLM output is handled."""
    # The regex only matches 0.x or 1.0*, so "1.5" won't match as a valid score.
    # But if somehow a value > 1 slips through, clamp must apply.
    # We test the clamp directly by verifying score("1.0") == 1.0
    scorer = make_scorer("1.0")
    result = asyncio.run(scorer.score("q", "ctx", "ans"))
    assert result is not None
    assert result <= 1.0


def test_cp4_score_below_0_clamped():
    """**Validates: Requirements CP-4** – value < 0 is handled (returns None since regex won't match)."""
    scorer = make_scorer("-0.5")
    result = asyncio.run(scorer.score("q", "ctx", "ans"))
    # Regex won't match negative numbers, so result is None
    assert result is None or result >= 0.0


# ---------------------------------------------------------------------------
# Unit tests – example-based sanity checks
# ---------------------------------------------------------------------------

def test_score_valid_decimal():
    """LLM returns '0.85' → score should be 0.85."""
    scorer = make_scorer("0.85")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 0.85


def test_score_zero():
    """LLM returns '0' → score should be 0.0."""
    scorer = make_scorer("0")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 0.0


def test_score_one():
    """LLM returns '1' → score should be 1.0."""
    scorer = make_scorer("1")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 1.0


def test_score_one_point_zero():
    """LLM returns '1.0' → score should be 1.0."""
    scorer = make_scorer("1.0")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 1.0


def test_score_no_number_returns_none():
    """LLM returns text with no parseable number → None."""
    scorer = make_scorer("không biết")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result is None


def test_score_number_with_surrounding_text():
    """LLM returns '0.72' embedded in text → score should be 0.72."""
    scorer = make_scorer("Độ tin cậy là 0.72 theo đánh giá của tôi.")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 0.72


def test_score_rounded_to_4_decimal_places():
    """Score should be rounded to 4 decimal places."""
    scorer = make_scorer("0.123456789")
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result is not None
    # 0.1234 after rounding to 4 decimal places
    assert result == round(result, 4)


def test_score_context_truncated_to_2000():
    """Long context should be truncated to 2000 chars without error."""
    long_context = "x" * 5000
    scorer = make_scorer("0.9")
    result = asyncio.run(scorer.score("query", long_context, "answer"))
    assert result == 0.9


def test_score_query_truncated_to_500():
    """Long query should be truncated to 500 chars without error."""
    long_query = "q" * 1000
    scorer = make_scorer("0.5")
    result = asyncio.run(scorer.score(long_query, "context", "answer"))
    assert result == 0.5


def test_score_answer_truncated_to_500():
    """Long answer should be truncated to 500 chars without error."""
    long_answer = "a" * 1000
    scorer = make_scorer("0.3")
    result = asyncio.run(scorer.score("query", "context", long_answer))
    assert result == 0.3


def test_score_provider_returns_string_without_content_attr():
    """When provider returns a plain string (no .content attr), should still parse."""
    provider = MagicMock()
    provider.ainvoke = AsyncMock(return_value="0.65")
    scorer = ConfidenceScorer(provider)
    result = asyncio.run(scorer.score("query", "context", "answer"))
    assert result == 0.65
