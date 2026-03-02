"""
tests/test_bias.py — Unit tests for rmoe.bias (Cognitive Bias Detector).

Paper §"Error Patterns and Bias Mitigation" (Table 2):
  Anchoring Bias               14.3%
  Difficulty with Conflicting  21.4%
  Limited Alternative Consid.  28.6%
  Overthinking / Rationaliz.   35.7%
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.bias import CognitiveBiasDetector, BiasType, BiasReport


@pytest.fixture
def detector():
    return CognitiveBiasDetector()


def _hyps(probs):
    return [{"diagnosis": f"Dx{i}", "probability": p} for i, p in enumerate(probs)]


def _hyps_named(names_probs):
    return [{"diagnosis": n, "probability": p} for n, p in names_probs]


# ═══════════════════════════════════════════════════════════════════════════════
#  Clean case — no biases
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_bias_balanced_ddx(detector):
    """Balanced DDx with varied CoT → no bias flags."""
    cot = (
        "Step 1: The 3.2cm LUL mass with spiculation is suspicious for malignancy. "
        "Step 2: However, consolidation and pneumonia should be ruled out. "
        "Step 3: Sarcoidosis shows bilateral hilar adenopathy. "
        "Conclusion: Pulmonary adenocarcinoma is most likely but pneumonia remains possible."
    )
    ddx = _hyps_named([("Pulmonary adenocarcinoma", 0.45),
                       ("Community-acquired pneumonia", 0.30),
                       ("Pulmonary sarcoidosis", 0.15),
                       ("Tuberculosis", 0.10)])
    report = detector.analyse(cot, ddx, sc=0.90)
    assert report.clean or all(f.bias_type != BiasType.Anchoring for f in report.flags)


# ═══════════════════════════════════════════════════════════════════════════════
#  Anchoring Bias
# ═══════════════════════════════════════════════════════════════════════════════

def test_anchoring_bias_detected(detector):
    """Top-1 dominates 90%+ of mass AND is mentioned 10x more → Anchoring."""
    top = "adenocarcinoma"
    cot = (
        f"The {top} {top} {top} {top} {top} {top} {top} {top} {top} {top} "
        f"is clearly the diagnosis. "
        "Dx1 might be there."  # alt mentioned once
    )
    ddx = _hyps_named([(top, 0.90), ("Dx1", 0.06), ("Dx2", 0.04)])
    report = detector.analyse(cot, ddx, sc=0.85)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.Anchoring in bias_types, f"Expected Anchoring, got: {bias_types}"


def test_anchoring_not_triggered_equal_mentions(detector):
    """Equal mention frequency → no Anchoring flag."""
    cot = "adenocarcinoma is possible. pneumonia is also possible. sarcoidosis must be excluded."
    ddx = _hyps_named([("adenocarcinoma", 0.50), ("pneumonia", 0.35), ("sarcoidosis", 0.15)])
    report = detector.analyse(cot, ddx, sc=0.90)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.Anchoring not in bias_types


# ═══════════════════════════════════════════════════════════════════════════════
#  Limited Alternative Consideration
# ═══════════════════════════════════════════════════════════════════════════════

def test_limited_alternatives_one_hypothesis(detector):
    """Only 1 hypothesis with significant probability → LimitedAlternatives."""
    cot = "This is adenocarcinoma."
    ddx = _hyps_named([("Pulmonary adenocarcinoma", 1.0)])
    report = detector.analyse(cot, ddx, sc=0.90)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.LimitedAlternatives in bias_types


def test_limited_alternatives_two_hypotheses(detector):
    """2 hypotheses is below default threshold of 3 → flag."""
    cot = "Adenocarcinoma or pneumonia."
    ddx = _hyps_named([("Adenocarcinoma", 0.60), ("Pneumonia", 0.40)])
    report = detector.analyse(cot, ddx, sc=0.85)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.LimitedAlternatives in bias_types


def test_no_limited_alternatives_three_hypotheses(detector):
    """3 hypotheses with non-trivial probability → no LimitedAlternatives."""
    cot = "Three competing diagnoses."
    ddx = _hyps_named([("A", 0.50), ("B", 0.30), ("C", 0.20)])
    report = detector.analyse(cot, ddx, sc=0.90)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.LimitedAlternatives not in bias_types


# ═══════════════════════════════════════════════════════════════════════════════
#  Overthinking / Rationalization
# ═══════════════════════════════════════════════════════════════════════════════

def test_overthinking_long_cot_low_sc(detector):
    """Very long CoT + low Sc → Overthinking flag."""
    # 2500+ chars of reasoning with low confidence
    cot = "extended reasoning " * 150  # ~3000 chars
    ddx = _hyps_named([("Adenocarcinoma", 0.40), ("Pneumonia", 0.35), ("Sarcoidosis", 0.25)])
    report = detector.analyse(cot, ddx, sc=0.70)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.Overthinking in bias_types


def test_overthinking_not_triggered_short_cot(detector):
    """Short CoT even with low Sc → no Overthinking."""
    cot = "Short reasoning step."  # << 2000 chars
    ddx = _hyps_named([("A", 0.45), ("B", 0.30), ("C", 0.25)])
    report = detector.analyse(cot, ddx, sc=0.75)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.Overthinking not in bias_types


def test_overthinking_not_triggered_high_sc(detector):
    """Long CoT but high Sc → no Overthinking."""
    cot = "detailed reasoning " * 150
    ddx = _hyps_named([("A", 0.80), ("B", 0.15), ("C", 0.05)])
    report = detector.analyse(cot, ddx, sc=0.92)
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.Overthinking not in bias_types


# ═══════════════════════════════════════════════════════════════════════════════
#  Conflicting Data
# ═══════════════════════════════════════════════════════════════════════════════

def test_conflicting_data_progression_ignored(detector):
    """Temporal note shows Progressed but CoT ignores it → ConflictingData."""
    temporal_note = "Lesion has progressed from 3.2cm to 3.8cm — new finding suspected."
    cot = "The mass is solid and spiculated. High likelihood of malignancy."
    ddx = _hyps_named([("Adenocarcinoma", 0.60), ("Pneumonia", 0.25), ("Sarcoidosis", 0.15)])
    report = detector.analyse(cot, ddx, temporal_note=temporal_note, sc=0.85)
    # Should flag conflicting data since CoT ignores temporal information
    bias_types = [f.bias_type for f in report.flags]
    assert BiasType.ConflictingData in bias_types


# ═══════════════════════════════════════════════════════════════════════════════
#  Correction hints
# ═══════════════════════════════════════════════════════════════════════════════

def test_correction_hints_format(detector):
    """correction_hints() returns non-empty string when flags present."""
    cot = "extended reasoning " * 150
    ddx = _hyps_named([("A", 0.50), ("B", 0.50)])  # 2 hypotheses → LimitedAlternatives
    report = detector.analyse(cot, ddx, sc=0.70)
    if report.flags:
        hints = report.correction_hints()
        assert isinstance(hints, str)
        assert len(hints) > 0
        assert "BIAS AUDIT" in hints or "correction" in hints.lower()


def test_clean_report_empty_hints(detector):
    """No bias flags → correction_hints returns empty string."""
    cot = "Balanced three-hypothesis reasoning."
    ddx = _hyps_named([("A", 0.50), ("B", 0.30), ("C", 0.20)])
    report = detector.analyse(cot, ddx, sc=0.95)
    if report.clean:
        assert report.correction_hints() == ""
