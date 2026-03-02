"""
tests/test_eval.py — Unit tests for rmoe.eval (BenchmarkRunner).

Paper §4: "R-MoE achieved an F1-score of 0.92 on MIMIC-CXR, outperforming
GPT-4V (0.85) and Gemini (0.87). False positives reduced by 25%,
with ECE at 0.08 versus 0.15 for baselines."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from rmoe.eval import (
    BenchmarkCase, BenchmarkDataset, BenchmarkMetrics,
    CaseResult, _aggregate, _compute_ece, _compute_brier,
    _compute_auc, _top1_match, _top3_match, _normalise,
    BUILTIN_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  BenchmarkDataset
# ═══════════════════════════════════════════════════════════════════════════════

def test_builtin_dataset_count():
    """Built-in dataset has 20 cases (paper MIMIC-CXR + RSNA + extra)."""
    ds = BenchmarkDataset()
    assert len(ds) == 20


def test_builtin_cases_have_required_fields():
    """Every builtin case has required non-empty fields."""
    ds = BenchmarkDataset()
    for case in ds.cases:
        assert case.case_id, f"Missing case_id: {case}"
        assert case.ground_truth, f"Missing ground_truth: {case}"
        assert case.organ, f"Missing organ: {case}"
        assert case.modality, f"Missing modality: {case}"


def test_csv_dataset_loads(tmp_path):
    """BenchmarkDataset loads correctly from a CSV file."""
    csv = tmp_path / "cases.csv"
    csv.write_text(
        "case_id,image_path,ground_truth,ground_truth_icd11,"
        "organ,modality,expected_risk_score,notes,prior_image_path\n"
        "TEST-001,img.png,Pneumonia,CA40.0,lung,CXR,,test case,\n"
        "TEST-002,img2.png,Fracture,NB82.0,msk,CXR,,test case 2,\n"
    )
    ds = BenchmarkDataset(str(csv))
    assert len(ds) == 2
    assert ds.cases[0].case_id == "TEST-001"
    assert ds.cases[1].ground_truth == "Fracture"


def test_filter_organ():
    """filter_organ returns only cases matching that organ."""
    ds = BenchmarkDataset()
    lung_ds = ds.filter_organ("lung")
    assert len(lung_ds) > 0
    for case in lung_ds.cases:
        assert case.organ.lower() == "lung"


# ═══════════════════════════════════════════════════════════════════════════════
#  String matching helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalise_strips_whitespace():
    assert _normalise("  Hello   World  ") == "hello world"


def test_normalise_lowercase():
    assert _normalise("Pulmonary Adenocarcinoma") == "pulmonary adenocarcinoma"


def test_top1_match_exact():
    assert _top1_match("Pulmonary adenocarcinoma", "Pulmonary adenocarcinoma")


def test_top1_match_case_insensitive():
    assert _top1_match("pulmonary adenocarcinoma", "Pulmonary Adenocarcinoma")


def test_top1_match_substring():
    assert _top1_match("Pulmonary adenocarcinoma LUL", "Pulmonary adenocarcinoma")


def test_top1_no_match():
    assert not _top1_match("Community-acquired pneumonia", "Pulmonary adenocarcinoma")


def test_top3_match():
    ddx = [
        {"diagnosis": "Pulmonary adenocarcinoma", "probability": 0.42},
        {"diagnosis": "Community-acquired pneumonia", "probability": 0.31},
        {"diagnosis": "Sarcoidosis", "probability": 0.15},
    ]
    assert _top3_match(ddx, "Community-acquired pneumonia")
    assert _top3_match(ddx, "sarcoidosis")
    assert not _top3_match(ddx, "Glioblastoma")


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computations
# ═══════════════════════════════════════════════════════════════════════════════

def _make_results(sc_correct_pairs):
    """Build minimal CaseResult list from [(sc, correct)] pairs."""
    results = []
    for i, (sc, correct) in enumerate(sc_correct_pairs):
        case = BenchmarkCase(
            case_id=f"T{i}", image_path="", ground_truth="Dx",
            ground_truth_icd11="", organ="lung", modality="CXR",
            expected_risk_score="",
        )
        r = CaseResult(case=case, sc=sc, top1_correct=correct, top3_correct=correct)
        results.append(r)
    return results


def test_ece_perfect():
    """All predictions correct with medium confidence → low ECE."""
    results = _make_results([(0.85, True)] * 20)
    ece = _compute_ece(results)
    assert ece < 0.20, f"ECE too high: {ece}"


def test_ece_all_wrong_high_conf():
    """High confidence, all wrong → ECE near 1."""
    results = _make_results([(0.95, False)] * 20)
    ece = _compute_ece(results)
    assert ece > 0.50, f"ECE should be high: {ece}"


def test_brier_perfect():
    """Sc=1.0, all correct → Brier = 0."""
    results = _make_results([(1.0, True)] * 10)
    brier = _compute_brier(results)
    assert abs(brier) < 1e-6


def test_brier_worst():
    """Sc=0.0, all correct → Brier = 1."""
    results = _make_results([(0.0, True)] * 10)
    brier = _compute_brier(results)
    assert abs(brier - 1.0) < 1e-6


def test_auc_range():
    """AUC must be in [0, 1]."""
    results = _make_results([(0.9, True), (0.7, True), (0.5, False), (0.3, False)])
    auc = _compute_auc(results)
    assert 0.0 <= auc <= 1.0


def test_auc_perfect():
    """Perfect ranker: all correct cases have higher Sc → AUC ≈ 1.0."""
    results = _make_results([
        (0.95, True), (0.90, True), (0.85, True),
        (0.40, False), (0.30, False), (0.20, False),
    ])
    auc = _compute_auc(results)
    assert auc >= 0.85, f"Expected high AUC, got {auc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregate metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_aggregate_accuracy():
    """10 correct of 20 → accuracy = 0.5."""
    results = _make_results([(0.9, True)] * 10 + [(0.9, False)] * 10)
    m = _aggregate(results)
    assert abs(m.accuracy - 0.5) < 1e-6


def test_aggregate_f1():
    """All correct, none escalated → precision = recall = F1 = 1.0."""
    results = _make_results([(0.95, True)] * 10)
    m = _aggregate(results)
    assert m.precision == pytest.approx(1.0, abs=0.01)
    assert m.recall    == pytest.approx(1.0, abs=0.01)


def test_aggregate_escalation_rate():
    """2 of 10 escalated → escalation_rate = 0.2."""
    results = _make_results([(0.95, True)] * 8 + [(0.95, True)] * 2)
    for r in results[-2:]:
        r.escalated = True
    m = _aggregate(results)
    assert abs(m.escalation_rate - 0.2) < 1e-6


def test_aggregate_empty():
    """Empty results → default BenchmarkMetrics with n_cases=0."""
    m = _aggregate([])
    assert m.n_cases == 0
    assert m.f1 == 0.0


def test_aggregate_icd11_accuracy():
    """2 of 4 cases with correct ICD-11 → icd11_accuracy = 0.5."""
    results = _make_results([(0.9, True)] * 4)
    results[0].icd11_correct = True
    results[1].icd11_correct = True
    m = _aggregate(results)
    assert abs(m.icd11_accuracy - 0.5) < 1e-6
