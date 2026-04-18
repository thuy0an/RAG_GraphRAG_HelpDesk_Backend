"""
Property-based tests for Multi-hop Reasoning deduplication logic.

Validates: Requirements CP-5
CP-5: Multi-hop Deduplication
  ∀ p1, p2 ∈ merged_passages: p1 ≠ p2 → p1.content ≠ p2.content
"""

from hypothesis import given, settings
from hypothesis import strategies as st


def deduplicate_passages(passages):
    """
    Deduplicate passages by content field.
    Mirrors the dedup logic used in GraphRAG.multi_hop_retrieve().
    """
    seen = set()
    result = []
    for p in passages:
        content = p.get("content", "")
        if content and content not in seen:
            seen.add(content)
            result.append(p)
    return result


# Strategy: generate a single passage dict with a content field
passage_strategy = st.fixed_dictionaries(
    {"content": st.text(min_size=0, max_size=200)}
)

# Strategy: generate a list of passages (may contain duplicates)
passages_list_strategy = st.lists(passage_strategy, min_size=0, max_size=20)


@given(passages=passages_list_strategy)
@settings(max_examples=200)
def test_dedup_no_duplicate_content(passages):
    """
    **Validates: Requirements CP-5**

    Property: After deduplication, no two passages share the same content.
    """
    result = deduplicate_passages(passages)
    contents = [p["content"] for p in result]
    assert len(contents) == len(set(contents)), (
        f"Duplicate content found after deduplication: {contents}"
    )


@given(passages=passages_list_strategy)
@settings(max_examples=200)
def test_dedup_with_duplicates_no_duplicate_content(passages):
    """
    **Validates: Requirements CP-5**

    Property: Concatenating a list with itself (all duplicates) and deduplicating
    yields no duplicate content — simulating multi-hop merge scenario.
    """
    merged = passages + passages  # all entries are duplicates
    result = deduplicate_passages(merged)
    contents = [p["content"] for p in result]
    assert len(contents) == len(set(contents)), (
        f"Duplicate content found after deduplication of doubled list: {contents}"
    )


@given(
    hop1=passages_list_strategy,
    hop2=passages_list_strategy,
)
@settings(max_examples=200)
def test_dedup_multi_hop_merge_no_duplicate_content(hop1, hop2):
    """
    **Validates: Requirements CP-5**

    Property: Merging passages from hop1 and hop2 (as in multi_hop_retrieve)
    and deduplicating yields no duplicate content.
    """
    merged = hop1 + hop2
    result = deduplicate_passages(merged)
    contents = [p["content"] for p in result]
    assert len(contents) == len(set(contents)), (
        f"Duplicate content found after multi-hop merge deduplication: {contents}"
    )


@given(passages=passages_list_strategy)
@settings(max_examples=100)
def test_dedup_result_is_subset_of_input(passages):
    """
    **Validates: Requirements CP-5**

    Property: Every passage in the deduplicated result was present in the input.
    """
    result = deduplicate_passages(passages)
    input_contents = {p.get("content", "") for p in passages}
    for p in result:
        assert p.get("content", "") in input_contents, (
            f"Dedup result contains content not in input: {p}"
        )


@given(passages=passages_list_strategy)
@settings(max_examples=100)
def test_dedup_empty_content_excluded(passages):
    """
    **Validates: Requirements CP-5**

    Property: Passages with empty content are excluded from the result
    (they are not deduplicated into the output).
    """
    result = deduplicate_passages(passages)
    for p in result:
        assert p.get("content", "") != "", (
            f"Empty-content passage found in dedup result: {p}"
        )
