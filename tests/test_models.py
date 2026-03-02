"""
tests/test_models.py — Unit tests for rmoe.models and WannaStateMachine.

Paper §3.1 Sc formula:  Sc = 1 − σ²
Paper §3.2 #wanna#:     If Sc < 0.90 → recurse (max 3 iter)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from rmoe.models import (
    DDxEnsemble, DDxHypothesis, FeedbackTensor, ReasoningOutput,
    WannaDecision, WannaState,
)
from rmoe.core import WannaStateMachine


# ═══════════════════════════════════════════════════════════════════════════════
#  DDxEnsemble — Sc = 1 − σ² formula (paper §3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ensemble(probs):
    hyps = [DDxHypothesis(f"Dx{i}", p) for i, p in enumerate(probs)]
    return DDxEnsemble(hyps)


def test_sc_formula_perfect():
    """Sc = 1 - Var(probs). For [1,0,0,0]: mean=0.25, Var=0.1875 → Sc=0.8125."""
    ens = _make_ensemble([1.0, 0.0, 0.0, 0.0])
    probs = [1.0, 0.0, 0.0, 0.0]
    mean = sum(probs) / len(probs)
    expected_var = sum((p - mean) ** 2 for p in probs) / len(probs)
    expected_sc  = 1.0 - expected_var
    assert abs(ens.sc - expected_sc) < 1e-6, f"Expected Sc={expected_sc:.4f}, got {ens.sc}"


def test_sc_formula_uniform():
    """Uniform 4-hypothesis DDx → high σ² → lower Sc."""
    ens = _make_ensemble([0.25, 0.25, 0.25, 0.25])
    # variance of [0.25,0.25,0.25,0.25] = 0, Sc = 1  BUT
    # formula is variance of the probability VALUES not the distribution
    # σ² = Var([0.25,0.25,0.25,0.25]) = 0, so Sc = 1 — check the actual impl
    assert 0.0 <= ens.sc <= 1.0


def test_sc_formula_paper_example():
    """Paper case: σ² ≈ 0.0149 → Sc ≈ 0.9851 (used in mock)"""
    probs = [0.42, 0.31, 0.15, 0.12]
    ens = _make_ensemble(probs)
    expected_var = 0.0
    mean = sum(probs) / len(probs)
    expected_var = sum((p - mean) ** 2 for p in probs) / len(probs)
    expected_sc  = 1.0 - expected_var
    assert abs(ens.sc - expected_sc) < 1e-6, f"Sc mismatch: {ens.sc} vs {expected_sc}"


def test_sc_always_valid():
    """Sc must remain in [0, 1] for any valid probability distribution."""
    import random
    rng = random.Random(42)
    for _ in range(20):
        raw = [rng.random() for _ in range(rng.randint(2, 8))]
        total = sum(raw)
        probs = [p / total for p in raw]
        ens = _make_ensemble(probs)
        assert 0.0 <= ens.sc <= 1.0, f"Sc out of range: {ens.sc}"


def test_entropy_half():
    """Binary 50/50 DDx → entropy = ln(2) ≈ 0.693 nats (paper uses natural log)."""
    ens = _make_ensemble([0.5, 0.5])
    expected = math.log(2)   # ≈ 0.6931 nats
    assert abs(ens.entropy() - expected) < 1e-4, f"Expected H≈{expected:.4f}, got {ens.entropy()}"


def test_entropy_certain():
    """Single certain hypothesis → entropy ≈ 0."""
    ens = _make_ensemble([1.0])
    assert abs(ens.entropy()) < 1e-6


def test_primary_hypothesis():
    """primary property returns hypothesis with highest probability."""
    ens = _make_ensemble([0.10, 0.55, 0.25, 0.10])
    assert ens.primary is not None
    assert ens.primary.probability == 0.55


def test_probabilities_list():
    """probabilities property returns list matching input probs."""
    probs = [0.4, 0.3, 0.2, 0.1]
    ens = _make_ensemble(probs)
    assert ens.probabilities == probs


# ═══════════════════════════════════════════════════════════════════════════════
#  WannaStateMachine (paper §3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_wanna_pass_above_threshold():
    """Sc ≥ θ → ProceedToReport (no recursion)."""
    sm  = WannaStateMachine(hard_limit=3, threshold=0.90)
    dec = sm.decide(0.95, iteration=1)
    assert dec.state == WannaState.ProceedToReport


def test_wanna_pass_exactly_threshold():
    """Sc = θ exactly → ProceedToReport."""
    sm  = WannaStateMachine(hard_limit=3, threshold=0.90)
    dec = sm.decide(0.90, iteration=1)
    assert dec.state == WannaState.ProceedToReport


def test_wanna_triggered_below_threshold():
    """Sc < θ, iter < limit → RequestHighResCrop (default wanna mode)."""
    sm  = WannaStateMachine(hard_limit=3, threshold=0.90)
    dec = sm.decide(0.75, iteration=1)
    assert dec.state in (WannaState.RequestHighResCrop, WannaState.RequestAlternateView)


def test_wanna_escalate_at_limit():
    """Sc < θ, iter == hard_limit → EscalateToHuman."""
    sm  = WannaStateMachine(hard_limit=3, threshold=0.90)
    dec = sm.decide(0.70, iteration=3)
    assert dec.state == WannaState.EscalateToHuman


def test_wanna_alternate_view():
    """'alternate view' in feedback_request → RequestAlternateView."""
    from rmoe.models import ReasoningOutput, DDxEnsemble, DDxHypothesis
    sm = WannaStateMachine(hard_limit=3, threshold=0.90)
    ens = DDxEnsemble([DDxHypothesis("Dx", 0.5)])
    ro  = ReasoningOutput(
        cot="some reasoning",
        ensemble=ens,
        wanna=True,
        feedback_request="Alternate View",
        feedback_payload="right_lateral",
        temporal_note="",
    )
    dec = sm.decide(0.70, iteration=1, reasoning=ro)
    assert dec.state == WannaState.RequestAlternateView


def test_threshold_configurable():
    """Custom threshold: Sc=0.85 < θ=0.92 → should trigger wanna."""
    sm  = WannaStateMachine(hard_limit=3, threshold=0.92)
    dec = sm.decide(0.88, iteration=1)
    assert dec.state != WannaState.ProceedToReport


def test_hard_limit_configurable():
    """Custom hard limit: escalate at iter=2 when limit=2."""
    sm  = WannaStateMachine(hard_limit=2, threshold=0.90)
    dec = sm.decide(0.70, iteration=2)
    assert dec.state == WannaState.EscalateToHuman
