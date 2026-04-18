"""
Unit tests cho 2 vấn đề đã fix:

Vấn đề 1: uploadedFiles sync từ compareRuns tự set selected=True
  → Fix: compareRuns sync với selected=False (user phải chủ động tích)

Vấn đề 2: Logic activeSourceFilters có điều kiện sai
  → Fix: luôn filter theo file được tích, không phụ thuộc vào tổng số file
"""

from typing import Dict, List, Optional


# =============================================================================
# Mirror logic từ frontend
# =============================================================================

def sync_compare_runs_to_uploaded_files(
    compare_runs: List[Dict],
    existing_files: List[Dict],
    auto_select: bool = False,  # Fix: mặc định False
) -> List[Dict]:
    """
    Mirror logic từ useEffect sync compareRuns → uploadedFiles.
    Fix: file mới từ compareRuns được thêm với selected=False (không tự chọn).
    """
    existing = {f["name"]: f for f in existing_files}
    for run in compare_runs:
        name = run.get("file_name")
        if not name:
            continue
        if name not in existing:
            existing[name] = {"name": name, "selected": auto_select}
    return list(existing.values())


def compute_filter_params(uploaded_files: List[Dict]):
    """
    Mirror logic từ handleSendMessage (sau fix).
    Returns: (activeSourceFilter, activeSourceFilters)
    - 0 file tích → (None, None) → search tất cả
    - 1 file tích → (filename, None) → single filter
    - 2+ file tích → (None, [filenames]) → multi OR filter
    """
    selected = [f for f in uploaded_files if f.get("selected", False)]
    if len(selected) == 0:
        return None, None
    if len(selected) == 1:
        return selected[0]["name"], None
    return None, [f["name"] for f in selected]


def build_filter_expression(
    source_filter: Optional[str] = None,
    source_filters: Optional[List[str]] = None,
) -> Optional[str]:
    """Mirror logic từ HybridRetriever._build_filter_expression()"""
    sources = []
    if source_filters:
        sources.extend(source_filters)
    elif source_filter:
        sources.append(source_filter)

    if not sources:
        return None
    if len(sources) == 1:
        return f'@source:{{"{sources[0]}"}}'
    parts = " | ".join(f'@source:{{"{s}"}}' for s in sources)
    return f"({parts})"


# =============================================================================
# Tests cho Vấn đề 1: compareRuns sync không tự set selected=True
# =============================================================================

class TestCompareRunsSync:

    def test_new_file_from_compare_runs_not_auto_selected(self):
        """
        Fix vấn đề 1: File mới từ compareRuns phải có selected=False.
        User phải chủ động tích để filter.
        """
        compare_runs = [
            {"file_name": "TTHCM_Chuong2.pdf"},
            {"file_name": "TTHCM_Chuong3.pdf"},
        ]
        existing = []
        result = sync_compare_runs_to_uploaded_files(compare_runs, existing, auto_select=False)

        for f in result:
            assert f["selected"] is False, (
                f"File '{f['name']}' từ compareRuns không được tự chọn (selected phải là False)"
            )

    def test_existing_selected_file_not_overridden(self):
        """
        File đã có trong uploadedFiles với selected=True không bị thay đổi.
        """
        existing = [
            {"name": "Effective_Opinion.pdf", "selected": True},
        ]
        compare_runs = [
            {"file_name": "Effective_Opinion.pdf"},  # đã có
            {"file_name": "TTHCM_Chuong3.pdf"},      # mới
        ]
        result = sync_compare_runs_to_uploaded_files(compare_runs, existing, auto_select=False)
        result_map = {f["name"]: f for f in result}

        # File cũ giữ nguyên selected=True
        assert result_map["Effective_Opinion.pdf"]["selected"] is True
        # File mới có selected=False
        assert result_map["TTHCM_Chuong3.pdf"]["selected"] is False

    def test_scenario_user_uploads_3_files_only_2_selected(self):
        """
        Kịch bản từ screenshot:
        - 3 file trong compareRuns (từ lịch sử)
        - User chỉ tích 2 file (Effective_Opinion + TTHCM Chương III)
        - TTHCM Chương II không được tích
        """
        # Sau fix: compareRuns sync với selected=False
        compare_runs = [
            {"file_name": "Effective_Opinion_Words_Extraction_for_F.pdf"},
            {"file_name": "TTHCM - TOM TAT CHUONG III.pdf"},
            {"file_name": "TTHCM - Tom tat Chuong II.pdf"},
        ]
        existing = []
        files = sync_compare_runs_to_uploaded_files(compare_runs, existing, auto_select=False)

        # Tất cả đều selected=False ban đầu
        assert all(f["selected"] is False for f in files)

        # User tích 2 file
        for f in files:
            if f["name"] in [
                "Effective_Opinion_Words_Extraction_for_F.pdf",
                "TTHCM - TOM TAT CHUONG III.pdf",
            ]:
                f["selected"] = True

        selected = [f for f in files if f["selected"]]
        assert len(selected) == 2
        assert "TTHCM - Tom tat Chuong II.pdf" not in [f["name"] for f in selected]

    def test_old_behavior_was_broken(self):
        """
        Chứng minh behavior cũ (auto_select=True) gây ra vấn đề.
        """
        compare_runs = [
            {"file_name": "TTHCM_Chuong2.pdf"},
            {"file_name": "TTHCM_Chuong3.pdf"},
            {"file_name": "Effective_Opinion.pdf"},
        ]
        existing = []

        # Behavior cũ: tất cả đều selected=True
        old_result = sync_compare_runs_to_uploaded_files(compare_runs, existing, auto_select=True)
        assert all(f["selected"] is True for f in old_result), "Behavior cũ: tất cả selected=True"

        # Behavior mới: tất cả selected=False
        new_result = sync_compare_runs_to_uploaded_files(compare_runs, existing, auto_select=False)
        assert all(f["selected"] is False for f in new_result), "Behavior mới: tất cả selected=False"


# =============================================================================
# Tests cho Vấn đề 2: Logic filter không phụ thuộc vào tổng số file
# =============================================================================

class TestFilterLogicFix:

    def test_no_files_selected_no_filter(self):
        """0 file tích → không filter (search tất cả)."""
        files = [
            {"name": "A.pdf", "selected": False},
            {"name": "B.pdf", "selected": False},
            {"name": "C.pdf", "selected": False},
        ]
        sf, sfs = compute_filter_params(files)
        assert sf is None
        assert sfs is None

    def test_one_file_selected_single_filter(self):
        """1 file tích → single filter."""
        files = [
            {"name": "A.pdf", "selected": False},
            {"name": "B.pdf", "selected": True},
            {"name": "C.pdf", "selected": False},
        ]
        sf, sfs = compute_filter_params(files)
        assert sf == "B.pdf"
        assert sfs is None

    def test_two_files_selected_multi_filter(self):
        """2 file tích → multi OR filter."""
        files = [
            {"name": "A.pdf", "selected": True},
            {"name": "B.pdf", "selected": True},
            {"name": "C.pdf", "selected": False},
        ]
        sf, sfs = compute_filter_params(files)
        assert sf is None
        assert sfs == ["A.pdf", "B.pdf"]

    def test_all_files_selected_no_filter(self):
        """Tất cả file tích → multi filter (search trong tất cả = không cần filter)."""
        files = [
            {"name": "A.pdf", "selected": True},
            {"name": "B.pdf", "selected": True},
            {"name": "C.pdf", "selected": True},
        ]
        sf, sfs = compute_filter_params(files)
        # Tất cả tích → multi filter với tất cả file
        assert sf is None
        assert sfs == ["A.pdf", "B.pdf", "C.pdf"]

    def test_old_broken_logic(self):
        """
        Chứng minh logic cũ bị broken:
        Khi tất cả 3 file đều selected=True (do auto-sync),
        logic cũ: selectedFiles.length (3) < uploadedFiles.length (3) → False → activeSourceFilters = null
        → Không filter → search tất cả → PaCRAG không tìm thấy thông tin phù hợp
        """
        files = [
            {"name": "A.pdf", "selected": True},
            {"name": "B.pdf", "selected": True},
            {"name": "C.pdf", "selected": True},
        ]
        selected = [f for f in files if f["selected"]]
        total = len(files)

        # Logic cũ (broken)
        old_active_filters = selected if len(selected) > 1 and len(selected) < total else None
        assert old_active_filters is None, "Logic cũ: khi tất cả selected → không filter (BUG)"

        # Logic mới (fixed)
        _, new_active_filters = compute_filter_params(files)
        assert new_active_filters == ["A.pdf", "B.pdf", "C.pdf"], "Logic mới: luôn filter theo file được tích"

    def test_scenario_from_screenshot(self):
        """
        Kịch bản từ screenshot:
        - 3 file trong uploadedFiles (do sync từ compareRuns với selected=True cũ)
        - User tích 2 file: Effective_Opinion + TTHCM Chương III
        - TTHCM Chương II không tích
        - Kỳ vọng: filter theo 2 file được tích
        """
        files = [
            {"name": "Effective_Opinion_Words_Extraction_for_F.pdf", "selected": True},
            {"name": "TTHCM - TOM TAT CHUONG III.pdf", "selected": True},
            {"name": "TTHCM - Tom tat Chuong II.pdf", "selected": False},
        ]
        sf, sfs = compute_filter_params(files)

        assert sf is None  # Không phải single filter
        assert sfs is not None
        assert "Effective_Opinion_Words_Extraction_for_F.pdf" in sfs
        assert "TTHCM - TOM TAT CHUONG III.pdf" in sfs
        assert "TTHCM - Tom tat Chuong II.pdf" not in sfs


# =============================================================================
# Tests cho filter expression generation
# =============================================================================

class TestFilterExpression:

    def test_single_source_filter(self):
        """1 file → single filter expression."""
        expr = build_filter_expression(source_filter="doc_A.pdf")
        assert expr == '@source:{"doc_A.pdf"}'

    def test_multi_source_filters_or_expression(self):
        """Nhiều file → OR expression."""
        expr = build_filter_expression(source_filters=["A.pdf", "B.pdf"])
        assert expr == '(@source:{"A.pdf"} | @source:{"B.pdf"})'

    def test_three_source_filters(self):
        """3 file → OR expression với 3 phần."""
        expr = build_filter_expression(source_filters=["A.pdf", "B.pdf", "C.pdf"])
        assert expr == '(@source:{"A.pdf"} | @source:{"B.pdf"} | @source:{"C.pdf"})'

    def test_no_filter(self):
        """Không có filter → None."""
        expr = build_filter_expression()
        assert expr is None

    def test_source_filters_takes_priority_over_source_filter(self):
        """source_filters có ưu tiên hơn source_filter."""
        expr = build_filter_expression(
            source_filter="single.pdf",
            source_filters=["A.pdf", "B.pdf"]
        )
        assert "A.pdf" in expr
        assert "B.pdf" in expr
        assert "single.pdf" not in expr

    def test_scenario_2_tthcm_files_filter(self):
        """
        Kịch bản: user tích 2 file TTHCM → filter expression đúng.
        """
        files_selected = [
            "TTHCM - TOM TAT CHUONG III.pdf",
            "TTHCM - Tom tat Chuong II.pdf",
        ]
        expr = build_filter_expression(source_filters=files_selected)
        assert "TTHCM - TOM TAT CHUONG III.pdf" in expr
        assert "TTHCM - Tom tat Chuong II.pdf" in expr
        assert "Effective_Opinion" not in expr


# =============================================================================
# Integration: full flow test
# =============================================================================

class TestFullFlowIntegration:

    def test_full_flow_2_files_selected(self):
        """
        Full flow: user tích 2 file → filter expression đúng → không search file thứ 3.
        """
        # Step 1: compareRuns sync (sau fix: selected=False)
        compare_runs = [
            {"file_name": "Effective_Opinion.pdf"},
            {"file_name": "TTHCM_Chuong3.pdf"},
            {"file_name": "TTHCM_Chuong2.pdf"},
        ]
        files = sync_compare_runs_to_uploaded_files(compare_runs, [], auto_select=False)
        assert all(f["selected"] is False for f in files)

        # Step 2: User tích 2 file
        for f in files:
            if f["name"] in ["Effective_Opinion.pdf", "TTHCM_Chuong3.pdf"]:
                f["selected"] = True

        # Step 3: Compute filter params
        sf, sfs = compute_filter_params(files)
        assert sf is None
        assert set(sfs) == {"Effective_Opinion.pdf", "TTHCM_Chuong3.pdf"}

        # Step 4: Build filter expression
        expr = build_filter_expression(source_filter=sf, source_filters=sfs)
        assert "Effective_Opinion.pdf" in expr
        assert "TTHCM_Chuong3.pdf" in expr
        assert "TTHCM_Chuong2.pdf" not in expr

    def test_full_flow_no_selection_search_all(self):
        """
        Full flow: không tích file nào → search tất cả (không filter).
        """
        compare_runs = [{"file_name": "A.pdf"}, {"file_name": "B.pdf"}]
        files = sync_compare_runs_to_uploaded_files(compare_runs, [], auto_select=False)

        sf, sfs = compute_filter_params(files)
        assert sf is None
        assert sfs is None

        expr = build_filter_expression(source_filter=sf, source_filters=sfs)
        assert expr is None

    def test_full_flow_upload_new_file_auto_selected(self):
        """
        Khi user upload file mới (qua handleUploadDocs), file được set selected=True.
        Đây là behavior đúng – user vừa upload thì muốn dùng ngay.
        """
        # Simulate handleUploadDocs behavior
        existing = [{"name": "old_file.pdf", "selected": False}]
        new_file = "new_file.pdf"

        # handleUploadDocs sets selected=True for new files
        existing_map = {f["name"]: f for f in existing}
        if new_file not in existing_map:
            existing_map[new_file] = {"name": new_file, "selected": True}
        result = list(existing_map.values())

        result_map = {f["name"]: f for f in result}
        assert result_map["new_file.pdf"]["selected"] is True  # Mới upload → selected
        assert result_map["old_file.pdf"]["selected"] is False  # Cũ → không thay đổi
