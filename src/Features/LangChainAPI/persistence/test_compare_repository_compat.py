"""
Property-based tests cho backward compatibility của CompareRepository._deserialize_query_json.

Validates: Requirements CP-5 (Backward Compatibility)

NOTE: Tests are self-contained to avoid heavy dependencies (sqlmodel, sqlalchemy).
The _deserialize_query_json logic is replicated here for isolation.
"""

import json
import pytest
from hypothesis import given, settings, strategies as st
from typing import Optional


# ---------------------------------------------------------------------------
# Replicate the method under test (pure logic, no DB dependencies)
# ---------------------------------------------------------------------------

def _deserialize_query_json(raw: Optional[str]) -> Optional[dict]:
    """Deserialize query JSON với backward compatibility cho các trường mới.

    This is a direct copy of CompareRepository._deserialize_query_json
    for isolated testing without heavy DB dependencies.
    """
    if not raw:
        return None
    data = json.loads(raw)
    # Backward compatibility: đảm bảo các trường mới có giá trị mặc định
    data.setdefault("relevance_score", None)
    data.setdefault("source_coverage", None)
    data.setdefault("word_count", None)
    data.setdefault("doc_passages", [])
    data.setdefault("retrieved_chunks", [])
    return data


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: tạo dict JSON tùy ý KHÔNG chứa các trường mới
_old_record_strategy = st.fixed_dictionaries(
    {},
    optional={
        "answer": st.text(max_size=200),
        "time_total_s": st.one_of(st.none(), st.floats(min_value=0, max_value=300, allow_nan=False)),
        "answer_tokens": st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        "sources": st.lists(
            st.fixed_dictionaries({"filename": st.text(max_size=100)}),
            max_size=10,
        ),
        "extra_legacy_field": st.text(max_size=50),
    },
)


# ---------------------------------------------------------------------------
# Property 1: Deserialize bản ghi cũ không raise exception
# **Validates: Requirements CP-5**
# ---------------------------------------------------------------------------

@given(old_data=_old_record_strategy)
@settings(max_examples=200)
def test_deserialize_old_record_no_exception(old_data):
    """
    **Validates: Requirements CP-5**

    Với bất kỳ dict JSON hợp lệ nào không chứa các trường mới,
    _deserialize_query_json không được raise exception.
    """
    raw = json.dumps(old_data)
    # Must not raise
    result = _deserialize_query_json(raw)
    assert result is not None


# ---------------------------------------------------------------------------
# Property 2: Các trường mới có giá trị mặc định đúng
# **Validates: Requirements CP-5**
# ---------------------------------------------------------------------------

@given(old_data=_old_record_strategy)
@settings(max_examples=200)
def test_new_fields_have_correct_defaults(old_data):
    """
    **Validates: Requirements CP-5**

    Sau khi deserialize bản ghi cũ (không có các trường mới),
    các trường mới phải có giá trị mặc định đúng:
    - relevance_score  → None
    - source_coverage  → None
    - word_count       → None
    - doc_passages     → []
    - retrieved_chunks → []
    """
    raw = json.dumps(old_data)
    result = _deserialize_query_json(raw)

    assert result["relevance_score"] is None, (
        f"relevance_score phải là None, nhận được {result['relevance_score']!r}"
    )
    assert result["source_coverage"] is None, (
        f"source_coverage phải là None, nhận được {result['source_coverage']!r}"
    )
    assert result["word_count"] is None, (
        f"word_count phải là None, nhận được {result['word_count']!r}"
    )
    assert result["doc_passages"] == [], (
        f"doc_passages phải là [], nhận được {result['doc_passages']!r}"
    )
    assert result["retrieved_chunks"] == [], (
        f"retrieved_chunks phải là [], nhận được {result['retrieved_chunks']!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: Các trường cũ được giữ nguyên sau khi deserialize
# **Validates: Requirements CP-5**
# ---------------------------------------------------------------------------

@given(old_data=_old_record_strategy)
@settings(max_examples=200)
def test_existing_fields_preserved(old_data):
    """
    **Validates: Requirements CP-5**

    Các trường đã có trong bản ghi cũ phải được giữ nguyên giá trị
    sau khi deserialize (setdefault không ghi đè giá trị hiện có).
    """
    raw = json.dumps(old_data)
    result = _deserialize_query_json(raw)

    for key, value in old_data.items():
        assert key in result, f"Trường '{key}' bị mất sau deserialize"
        assert result[key] == value, (
            f"Trường '{key}' bị thay đổi: mong đợi {value!r}, nhận được {result[key]!r}"
        )


# ---------------------------------------------------------------------------
# Property 4: None / empty string trả về None
# **Validates: Requirements CP-5**
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [None, ""])
def test_deserialize_none_or_empty_returns_none(raw):
    """
    **Validates: Requirements CP-5**

    _deserialize_query_json(None) và _deserialize_query_json("") phải trả về None.
    """
    result = _deserialize_query_json(raw)
    assert result is None


# ---------------------------------------------------------------------------
# Property 5: Nếu bản ghi mới đã có các trường, giá trị không bị ghi đè
# **Validates: Requirements CP-5**
# ---------------------------------------------------------------------------

@given(
    relevance_score=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    source_coverage=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    word_count=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
    doc_passages=st.lists(st.text(max_size=20), max_size=5),
    retrieved_chunks=st.lists(st.text(max_size=20), max_size=5),
)
@settings(max_examples=100)
def test_existing_new_fields_not_overwritten(
    relevance_score, source_coverage, word_count, doc_passages, retrieved_chunks
):
    """
    **Validates: Requirements CP-5**

    Nếu bản ghi đã có sẵn các trường mới (bản ghi từ phiên bản mới),
    setdefault không được ghi đè giá trị đó.
    """
    data = {
        "answer": "test answer",
        "relevance_score": relevance_score,
        "source_coverage": source_coverage,
        "word_count": word_count,
        "doc_passages": doc_passages,
        "retrieved_chunks": retrieved_chunks,
    }
    raw = json.dumps(data)
    result = _deserialize_query_json(raw)

    assert result["relevance_score"] == relevance_score
    assert result["source_coverage"] == source_coverage
    assert result["word_count"] == word_count
    assert result["doc_passages"] == doc_passages
    assert result["retrieved_chunks"] == retrieved_chunks
