"""
Property-based tests for CP-1: Source Filter Isolation

CP-1: ∀ doc ∈ results: source_filter ≠ null → doc.metadata.source = source_filter

**Validates: Requirements 1.1 (Metadata Filtering)**
"""

from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Pure function under test (mirrors logic in HybridRetriever)
# ---------------------------------------------------------------------------

def apply_source_filter(
    docs: List[Dict], source_filter: Optional[str]
) -> List[Dict]:
    """Filter docs by metadata.source.

    When source_filter is None, all docs are returned unchanged.
    When source_filter is provided, only docs whose metadata["source"]
    equals source_filter are returned.
    """
    if not source_filter:
        return docs
    return [
        doc for doc in docs
        if doc.get("metadata", {}).get("source") == source_filter
    ]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

source_text = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-."))

doc_strategy = st.fixed_dictionaries({
    "content": st.text(max_size=200),
    "metadata": st.fixed_dictionaries({
        "source": st.one_of(st.none(), source_text),
    }),
})

docs_list = st.lists(doc_strategy, min_size=0, max_size=30)


# ---------------------------------------------------------------------------
# Property 1 – CP-1: when source_filter is provided, all returned docs must
#              have metadata["source"] == source_filter
# ---------------------------------------------------------------------------

@given(docs=docs_list, source_filter=source_text)
@settings(max_examples=200)
def test_source_filter_returns_only_matching_docs(
    docs: List[Dict], source_filter: str
):
    """**Validates: Requirements 1.1**

    CP-1: When source_filter is provided, every doc in the result must have
    metadata.source equal to source_filter.
    """
    result = apply_source_filter(docs, source_filter)

    for doc in result:
        assert doc.get("metadata", {}).get("source") == source_filter, (
            f"Expected metadata.source == {source_filter!r}, "
            f"got {doc.get('metadata', {}).get('source')!r}"
        )


# ---------------------------------------------------------------------------
# Property 2 – when source_filter is None, all docs are returned unchanged
# ---------------------------------------------------------------------------

@given(docs=docs_list)
@settings(max_examples=200)
def test_no_filter_returns_all_docs(docs: List[Dict]):
    """**Validates: Requirements 1.1**

    When source_filter is None (or falsy), apply_source_filter must return
    the original list unchanged (same length, same order).
    """
    result = apply_source_filter(docs, None)

    assert result is docs, (
        "When source_filter is None, the original list should be returned as-is"
    )


# ---------------------------------------------------------------------------
# Property 3 – result is a subset of the input
# ---------------------------------------------------------------------------

@given(docs=docs_list, source_filter=st.one_of(st.none(), source_text))
@settings(max_examples=200)
def test_filter_result_is_subset_of_input(
    docs: List[Dict], source_filter: Optional[str]
):
    """**Validates: Requirements 1.1**

    The filtered result must always be a subset of the input docs
    (no new docs are introduced).
    """
    result = apply_source_filter(docs, source_filter)

    # Every doc in result must have been in docs
    for doc in result:
        assert doc in docs, "Result contains a doc not present in the input"

    # Result cannot be larger than input
    assert len(result) <= len(docs)


# ---------------------------------------------------------------------------
# Property 4 – idempotency: filtering twice with the same filter == filtering once
# ---------------------------------------------------------------------------

@given(docs=docs_list, source_filter=source_text)
@settings(max_examples=200)
def test_filter_is_idempotent(docs: List[Dict], source_filter: str):
    """**Validates: Requirements 1.1**

    Applying the same source_filter twice should yield the same result as
    applying it once (idempotency).
    """
    once = apply_source_filter(docs, source_filter)
    twice = apply_source_filter(once, source_filter)

    assert once == twice, (
        "Applying the same filter twice should be idempotent"
    )


# ---------------------------------------------------------------------------
# Example-based sanity checks
# ---------------------------------------------------------------------------

def test_filter_with_matching_docs():
    docs = [
        {"content": "a", "metadata": {"source": "file_a.pdf"}},
        {"content": "b", "metadata": {"source": "file_b.pdf"}},
        {"content": "c", "metadata": {"source": "file_a.pdf"}},
    ]
    result = apply_source_filter(docs, "file_a.pdf")
    assert len(result) == 2
    assert all(d["metadata"]["source"] == "file_a.pdf" for d in result)


def test_filter_with_no_matching_docs():
    docs = [
        {"content": "a", "metadata": {"source": "file_b.pdf"}},
    ]
    result = apply_source_filter(docs, "file_a.pdf")
    assert result == []


def test_filter_none_returns_all():
    docs = [
        {"content": "a", "metadata": {"source": "file_a.pdf"}},
        {"content": "b", "metadata": {"source": "file_b.pdf"}},
    ]
    result = apply_source_filter(docs, None)
    assert result is docs


def test_filter_empty_string_returns_all():
    """Empty string is falsy – treated the same as None."""
    docs = [
        {"content": "a", "metadata": {"source": "file_a.pdf"}},
    ]
    result = apply_source_filter(docs, "")
    assert result is docs


def test_filter_doc_without_metadata():
    docs = [
        {"content": "no metadata here"},
        {"content": "has metadata", "metadata": {"source": "x"}},
    ]
    result = apply_source_filter(docs, "x")
    assert len(result) == 1
    assert result[0]["metadata"]["source"] == "x"


def test_filter_doc_without_source_key():
    docs = [
        {"content": "a", "metadata": {}},
        {"content": "b", "metadata": {"source": "target"}},
    ]
    result = apply_source_filter(docs, "target")
    assert len(result) == 1
    assert result[0]["metadata"]["source"] == "target"
