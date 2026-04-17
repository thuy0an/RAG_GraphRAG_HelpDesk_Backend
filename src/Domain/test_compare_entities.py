"""
Property-based tests for Domain entities in compare_entities.py.

**Validates: Requirements 1.2** (RetrievedPassage serialize/deserialize round-trip)
"""

import json
import pytest
from hypothesis import given, settings, strategies as st

from AI_HelpDesk_Backend.src.Domain.compare_entities import RetrievedPassage


@given(
    content=st.text(),
    filename=st.text(),
    pages=st.lists(st.integers(min_value=1, max_value=9999)),
)
@settings(max_examples=200)
def test_retrieved_passage_json_roundtrip(content: str, filename: str, pages: list[int]):
    """
    RetrievedPassage must survive a JSON serialize/deserialize round-trip
    without any data loss.

    **Validates: Requirements 1.2**
    """
    original = RetrievedPassage(content=content, filename=filename, pages=pages)

    # Serialize to JSON string and back
    json_str = original.model_dump_json()
    restored = RetrievedPassage.model_validate_json(json_str)

    assert restored.content == original.content
    assert restored.filename == original.filename
    assert restored.pages == original.pages


@given(
    content=st.text(),
    filename=st.text(),
    pages=st.lists(st.integers(min_value=1, max_value=9999)),
)
@settings(max_examples=200)
def test_retrieved_passage_dict_roundtrip(content: str, filename: str, pages: list[int]):
    """
    RetrievedPassage must survive a model_dump / model_validate round-trip
    (dict-based) without any data loss.

    **Validates: Requirements 1.2**
    """
    original = RetrievedPassage(content=content, filename=filename, pages=pages)

    data = original.model_dump()
    restored = RetrievedPassage(**data)

    assert restored.content == original.content
    assert restored.filename == original.filename
    assert restored.pages == original.pages
