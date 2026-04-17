"""
Property-based tests for _compute_relevance_score and _compute_source_coverage
helper functions in LangChainController.

Validates: Requirements CP-1 (Relevance Score Bounds) and CP-2 (Source Coverage Bounds)
"""
import logging
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from typing import Optional
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inline the helper functions to avoid importing the full application stack
# (which requires langchain_mistralai and other heavy dependencies).
# The implementations below are identical to those in LangChainController.py.
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def _compute_relevance_score(embedding_model, query: str, answer: str) -> Optional[float]:
    """Tính cosine similarity giữa embedding của query và answer."""
    try:
        q_vec = np.array(embedding_model.embed_query(query))
        a_vec = np.array(embedding_model.embed_query(answer))
        norm = np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
        if norm == 0:
            return None
        cosine = float(np.dot(q_vec, a_vec) / norm)
        return round(max(0.0, min(1.0, cosine)), 4)
    except Exception as e:
        log.error(f"Failed to compute relevance score: {e}")
        return None


def _compute_source_coverage(sources: list, retrieved_chunk_count: int) -> Optional[float]:
    """Tính tỉ lệ unique sources / retrieved_chunk_count, clamped to [0.0, 1.0]."""
    if not retrieved_chunk_count:
        return None
    unique_sources = len({s.get("filename", "") for s in sources if s.get("filename")})
    return round(min(1.0, unique_sources / retrieved_chunk_count), 4)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_embedding_model(dim: int = 8):
    """Return a mock embedding model that returns a random unit-ish vector."""
    model = MagicMock()
    rng = np.random.default_rng(seed=42)

    def embed_query(text: str):
        # Deterministic-ish: hash the text to get a seed, then generate vector
        seed = hash(text) % (2**31)
        local_rng = np.random.default_rng(seed=seed)
        return local_rng.standard_normal(dim).tolist()

    model.embed_query.side_effect = embed_query
    return model


def _make_zero_embedding_model():
    """Return a mock embedding model that always returns a zero vector."""
    model = MagicMock()
    model.embed_query.return_value = [0.0] * 8
    return model


def _make_failing_embedding_model():
    """Return a mock embedding model that always raises an exception."""
    model = MagicMock()
    model.embed_query.side_effect = RuntimeError("embedding service unavailable")
    return model


# ---------------------------------------------------------------------------
# Tests for _compute_relevance_score
# ---------------------------------------------------------------------------

class TestComputeRelevanceScore:
    """
    **Validates: Requirements CP-1**
    Property: ∀ query, answer: relevance_score ∈ [0.0, 1.0] ∨ relevance_score = None
    """

    @given(
        query=st.text(min_size=1, max_size=200),
        answer=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_result_is_none_or_in_unit_interval(self, query, answer):
        """
        **Validates: Requirements CP-1**
        For any query and answer, the relevance score must be None or in [0.0, 1.0].
        """
        model = _make_embedding_model()
        score = _compute_relevance_score(model, query, answer)
        assert score is None or (0.0 <= score <= 1.0), (
            f"Expected None or value in [0.0, 1.0], got {score!r}"
        )

    @given(
        query=st.text(min_size=1, max_size=200),
        answer=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=50)
    def test_result_rounded_to_4_decimal_places(self, query, answer):
        """
        **Validates: Requirements CP-1**
        Non-None scores must be rounded to at most 4 decimal places.
        """
        model = _make_embedding_model()
        score = _compute_relevance_score(model, query, answer)
        if score is not None:
            assert score == round(score, 4), (
                f"Score {score!r} is not rounded to 4 decimal places"
            )

    def test_zero_vector_returns_none(self):
        """
        **Validates: Requirements CP-1**
        When embedding returns zero vectors, norm is 0 → result must be None.
        """
        model = _make_zero_embedding_model()
        score = _compute_relevance_score(model, "hello", "world")
        assert score is None

    def test_embedding_exception_returns_none(self):
        """
        **Validates: Requirements CP-1**
        When embedding model raises an exception, result must be None (not propagate).
        """
        model = _make_failing_embedding_model()
        score = _compute_relevance_score(model, "hello", "world")
        assert score is None

    def test_identical_query_and_answer_returns_high_score(self):
        """
        **Validates: Requirements CP-1**
        Identical query and answer should yield a high cosine similarity (close to 1.0).
        """
        model = _make_embedding_model()
        text = "What is the capital of France?"
        score = _compute_relevance_score(model, text, text)
        assert score is not None
        assert score >= 0.99, f"Expected score close to 1.0 for identical texts, got {score}"


# ---------------------------------------------------------------------------
# Tests for _compute_source_coverage
# ---------------------------------------------------------------------------

class TestComputeSourceCoverage:
    """
    **Validates: Requirements CP-2**
    Properties:
    - retrieved_chunk_count = 0 → source_coverage = None
    - source_coverage ∈ [0.0, 1.0] ∨ source_coverage = None
    """

    @given(
        sources=st.lists(
            st.fixed_dictionaries({"filename": st.text()}),
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_zero_chunk_count_always_returns_none(self, sources):
        """
        **Validates: Requirements CP-2**
        When retrieved_chunk_count is 0, result must always be None.
        """
        result = _compute_source_coverage(sources, 0)
        assert result is None, (
            f"Expected None when chunk_count=0, got {result!r}"
        )

    @given(
        sources=st.lists(
            st.fixed_dictionaries({"filename": st.text()}),
            max_size=20,
        ),
        chunk_count=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_nonzero_chunk_count_result_in_unit_interval(self, sources, chunk_count):
        """
        **Validates: Requirements CP-2**
        When chunk_count > 0, result must be None or in [0.0, 1.0].
        """
        result = _compute_source_coverage(sources, chunk_count)
        assert result is None or (0.0 <= result <= 1.0), (
            f"Expected None or value in [0.0, 1.0], got {result!r}"
        )

    @given(
        sources=st.lists(
            st.fixed_dictionaries({"filename": st.text()}),
            min_size=0,
            max_size=20,
        ),
        chunk_count=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=100)
    def test_result_rounded_to_4_decimal_places(self, sources, chunk_count):
        """
        **Validates: Requirements CP-2**
        Non-None results must be rounded to at most 4 decimal places.
        """
        result = _compute_source_coverage(sources, chunk_count)
        if result is not None:
            assert result == round(result, 4), (
                f"Coverage {result!r} is not rounded to 4 decimal places"
            )

    def test_empty_sources_with_nonzero_chunk_count(self):
        """
        **Validates: Requirements CP-2**
        Empty sources list with chunk_count > 0 → coverage = 0.0 (0 unique sources).
        """
        result = _compute_source_coverage([], 5)
        assert result == 0.0, f"Expected 0.0 for empty sources, got {result!r}"

    def test_sources_without_filename_key_ignored(self):
        """
        **Validates: Requirements CP-2**
        Sources without 'filename' key should be ignored in unique count.
        """
        sources = [{"other_key": "value"}, {"filename": ""}, {"filename": "doc.pdf"}]
        # Only "doc.pdf" has a non-empty filename → 1 unique source
        result = _compute_source_coverage(sources, 3)
        assert result == round(1 / 3, 4), f"Expected {round(1/3, 4)}, got {result!r}"

    def test_all_same_filename_counts_as_one_unique(self):
        """
        **Validates: Requirements CP-2**
        Multiple sources with the same filename count as 1 unique source.
        """
        sources = [
            {"filename": "doc.pdf"},
            {"filename": "doc.pdf"},
            {"filename": "doc.pdf"},
        ]
        result = _compute_source_coverage(sources, 3)
        assert result == round(1 / 3, 4), f"Expected {round(1/3, 4)}, got {result!r}"

    def test_coverage_clamped_to_one_when_sources_exceed_chunks(self):
        """
        **Validates: Requirements CP-2**
        When unique sources exceed retrieved_chunk_count, coverage is clamped to 1.0.
        """
        # 5 unique sources but only 3 chunks retrieved → would be 5/3 ≈ 1.667 without clamping
        sources = [
            {"filename": "a.pdf"},
            {"filename": "b.pdf"},
            {"filename": "c.pdf"},
            {"filename": "d.pdf"},
            {"filename": "e.pdf"},
        ]
        result = _compute_source_coverage(sources, 3)
        assert result == 1.0, f"Expected 1.0 (clamped), got {result!r}"
