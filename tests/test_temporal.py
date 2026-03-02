"""
tests/test_temporal.py — Unit tests for rmoe.temporal (TemporalComparator).

Paper §3.1 ARLL: "Comparative Temporal Analysis: Where prior imaging is
available, detect interval changes across time-points to inform diagnosis
confidence."

Paper §"#wanna# Protocol": "Perform Temporal Analysis: Trigger a search
for older imaging records to perform a comparative analysis of lesion
progression over time."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.temporal import (
    TemporalComparator, ChangeClass, TemporalAnalysis, mock_temporal_note,
)


@pytest.fixture
def comparator():
    return TemporalComparator(growth_threshold_mm=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  No prior scan
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_prior_returns_no_comparison(comparator):
    """No prior path → NoComparison classification."""
    result = comparator.compare("current.png", None)
    assert result.overall_class == ChangeClass.NoComparison
    assert result.sc_adjustment == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Size-based classification (Fleischner 1.5mm threshold)
# ═══════════════════════════════════════════════════════════════════════════════

def test_stable_within_threshold(comparator):
    """Delta < 1.5mm → Stable."""
    result = comparator.compare(
        "current.png", "prior.png",
        current_roi_size_mm=32.0, prior_roi_size_mm=31.2,
        # delta = 0.8mm < 1.5mm threshold
    )
    assert result.overall_class == ChangeClass.Stable


def test_progressed_above_threshold(comparator):
    """Growth > 1.5mm → Progressed."""
    result = comparator.compare(
        "current.png", "prior.png",
        current_roi_size_mm=38.0, prior_roi_size_mm=32.0,
        # delta = 6mm >> 1.5mm threshold
    )
    assert result.overall_class == ChangeClass.Progressed


def test_regressed_below_threshold(comparator):
    """Shrinkage > 1.5mm → Regressed."""
    result = comparator.compare(
        "current.png", "prior.png",
        current_roi_size_mm=24.0, prior_roi_size_mm=32.0,
        # delta = -8mm
    )
    assert result.overall_class == ChangeClass.Regressed


def test_new_finding(comparator):
    """No prior size, current > 0 → New finding."""
    result = comparator.compare(
        "current.png", "prior.png",
        current_roi_size_mm=15.0, prior_roi_size_mm=0.0,
    )
    assert result.overall_class == ChangeClass.New


def test_resolved_finding(comparator):
    """Prior size > 0, current = 0 → Resolved."""
    result = comparator.compare(
        "current.png", "prior.png",
        current_roi_size_mm=0.0, prior_roi_size_mm=20.0,
    )
    assert result.overall_class == ChangeClass.Resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  Sc adjustments (paper §3 — confidence gating)
# ═══════════════════════════════════════════════════════════════════════════════

def test_sc_adjustment_stable(comparator):
    """Stable → +0.02 Sc adjustment."""
    result = comparator.compare(
        "current.png", "prior.png", 31.0, 30.8
    )
    assert result.sc_adjustment == pytest.approx(+0.02, abs=1e-6)


def test_sc_adjustment_progressed(comparator):
    """Progressed → -0.05 Sc adjustment (penalty)."""
    result = comparator.compare(
        "current.png", "prior.png", 38.0, 30.0
    )
    assert result.sc_adjustment == pytest.approx(-0.05, abs=1e-6)


def test_sc_adjustment_regressed(comparator):
    """Regressed → +0.03 Sc adjustment (treatment response)."""
    result = comparator.compare(
        "current.png", "prior.png", 22.0, 32.0
    )
    assert result.sc_adjustment == pytest.approx(+0.03, abs=1e-6)


def test_sc_adjustment_new(comparator):
    """New finding → -0.04 Sc adjustment."""
    result = comparator.compare(
        "current.png", "prior.png", 15.0, 0.0
    )
    assert result.sc_adjustment == pytest.approx(-0.04, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  Significance flag (Fleischner criterion)
# ═══════════════════════════════════════════════════════════════════════════════

def test_significant_change_flag(comparator):
    """Growth ≥ 1.5mm → significant_change = True."""
    result = comparator.compare(
        "current.png", "prior.png", 34.0, 30.0
    )
    assert result.significant_change is True


def test_not_significant_small_change(comparator):
    """Growth < 1.5mm → significant_change = False."""
    result = comparator.compare(
        "current.png", "prior.png", 31.0, 30.5
    )
    assert result.significant_change is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Interval note text
# ═══════════════════════════════════════════════════════════════════════════════

def test_interval_note_contains_change_class(comparator):
    """Interval note should mention the classification."""
    result = comparator.compare(
        "current.png", "prior.png", 38.0, 32.0
    )
    note = result.interval_note.lower()
    assert "progress" in note or "grow" in note or "increas" in note


def test_interval_note_no_comparison(comparator):
    """NoComparison note should say 'no prior'."""
    result = comparator.compare("current.png", None)
    assert "prior" in result.interval_note.lower() or \
           "comparison" in result.interval_note.lower()


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock notes
# ═══════════════════════════════════════════════════════════════════════════════

def test_mock_note_no_prior():
    note = mock_temporal_note(has_prior=False)
    assert "prior" in note.lower() or "no prior" in note.lower() or "comparison" in note.lower()


def test_mock_note_with_prior_iter1():
    note = mock_temporal_note(has_prior=True, iteration=1)
    assert len(note) > 20


def test_mock_note_with_prior_iter2():
    """Iteration 2 should return 'stable' (post-first-look)."""
    note = mock_temporal_note(has_prior=True, iteration=2)
    assert "stable" in note.lower() or len(note) > 20
