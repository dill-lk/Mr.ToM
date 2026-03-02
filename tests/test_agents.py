"""
tests/test_agents.py — Unit tests for rmoe.agents parsing helpers.

Covers _is_clinical_hypothesis and _parse_arll_output to ensure garbage
text fragments produced by the model (when it outputs prose instead of JSON)
are rejected and the fallback ensemble is used instead.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from rmoe.agents import _is_clinical_hypothesis, _parse_arll_output


# ═══════════════════════════════════════════════════════════════════════════════
#  _is_clinical_hypothesis
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsClinicalHypothesis:
    """Garbage fragments from the problem statement must be rejected."""

    # ── Real medical diagnoses must PASS ──────────────────────────────────────

    def test_rib_fracture(self):
        assert _is_clinical_hypothesis("Rib fracture") is True

    def test_pulmonary_adenocarcinoma(self):
        assert _is_clinical_hypothesis("Pulmonary adenocarcinoma") is True

    def test_pneumothorax(self):
        assert _is_clinical_hypothesis("Pneumothorax") is True

    def test_community_acquired_pneumonia(self):
        assert _is_clinical_hypothesis("Community-acquired pneumonia") is True

    def test_wrist_fracture(self):
        assert _is_clinical_hypothesis("Wrist fracture") is True

    def test_tb_reactivation(self):
        assert _is_clinical_hypothesis("Tuberculosis reactivation") is True

    def test_pleural_effusion(self):
        assert _is_clinical_hypothesis("Pleural effusion") is True

    # ── Garbage fragments from broken-hand run must FAIL ─────────────────────

    def test_garbage_with_probability(self):
        """'with probability' starts with lowercase → must be rejected."""
        assert _is_clinical_hypothesis("with probability") is False

    def test_garbage_spatial_attention_map(self):
        """Lowercase start + attention map reference → must be rejected."""
        assert _is_clinical_hypothesis(
            "t the spatial attention map has a attn of"
        ) is False

    def test_garbage_trying_to_understand(self):
        """Lowercase start + ARLL pipeline reference → must be rejected."""
        assert _is_clinical_hypothesis(
            "m trying to understand how the ARLL Phase"
        ) is False

    def test_garbage_attention_map_uppercase(self):
        """'Spatial Attention Map' – contains 'attention map' → must be rejected."""
        assert _is_clinical_hypothesis(
            "e a Spatial Attention Map with an attn of"
        ) is False

    def test_garbage_approach_arll(self):
        """Partial sentence referencing ARLL → must be rejected."""
        assert _is_clinical_hypothesis(
            "igure out how to approach this ARLL Phase"
        ) is False

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_too_short(self):
        """Names shorter than 4 characters are rejected."""
        assert _is_clinical_hypothesis("Flu") is False

    def test_empty_string(self):
        assert _is_clinical_hypothesis("") is False

    def test_single_char(self):
        assert _is_clinical_hypothesis("X") is False

    def test_arll_in_name(self):
        """Any name containing 'arll' is treated as pipeline meta-text."""
        assert _is_clinical_hypothesis("ARLL phase output") is False

    def test_attention_map_in_name(self):
        """Any name containing 'attention map' is pipeline meta-text."""
        assert _is_clinical_hypothesis("Attention map analysis") is False


# ═══════════════════════════════════════════════════════════════════════════════
#  _parse_arll_output  —  fallback triggered when model outputs prose
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseArllOutput:
    """When ARLL outputs prose (no valid JSON), the fallback ensemble must be
    used rather than garbage hypothesis names."""

    def test_valid_json_parsed_correctly(self):
        """Clean JSON output is parsed without fallback."""
        raw = json.dumps({
            "cot": "Step 1 — fracture analysis …",
            "ddx": [
                {"diagnosis": "Rib fracture", "probability": 0.75, "evidence": "cortical break"},
                {"diagnosis": "Pneumothorax",  "probability": 0.15, "evidence": "no lung marking"},
                {"diagnosis": "Pulmonary contusion", "probability": 0.10, "evidence": ""},
            ],
            "sigma2": 0.07,
            "sc": 0.93,
            "wanna": False,
            "feedback_request": None,
            "feedback_payload": None,
            "rag_references": [],
            "temporal_note": None,
        })
        out = _parse_arll_output(raw)
        assert len(out.ensemble.hypotheses) == 3
        assert out.ensemble.hypotheses[0].diagnosis == "Rib fracture"
        assert out.ensemble.hypotheses[0].probability == 0.75

    def test_garbage_prose_yields_empty_ensemble(self):
        """Prose output (no JSON, no valid diagnosis:number pairs) gives empty
        ensemble, triggering the fallback in ReasoningExpert.execute()."""
        raw = (
            "I'm trying to understand how the ARLL Phase works. "
            "The spatial attention map has a attn of 0.100. "
            "m trying to parse with probability 0.950. "
            "arll phase approach 0.020."
        )
        out = _parse_arll_output(raw)
        # All regex-matched candidates are garbage (lowercase start or
        # containing non-clinical substrings) → ensemble must be empty
        assert out.ensemble.hypotheses == []

    def test_mixed_prose_valid_diagnosis(self):
        """When prose contains both garbage and a real diagnosis:prob pair,
        only the real diagnosis survives."""
        raw = (
            "Based on the analysis, with probability 0.90 we see findings. "
            "Rib fracture: 0.75 — cortical break visible. "
            "arll output: 0.10"
        )
        out = _parse_arll_output(raw)
        names = [h.diagnosis for h in out.ensemble.hypotheses]
        assert "Rib fracture" in names
        # Garbage entries must NOT be present
        for name in names:
            assert name[0].isupper(), f"Non-uppercase diagnosis slipped through: {name!r}"

    def test_json_with_garbage_diagnosis_field_filtered(self):
        """JSON block where the 'diagnosis' field contains meta-text is filtered."""
        raw = json.dumps({
            "cot": "some cot",
            "ddx": [
                {"diagnosis": "with probability", "probability": 0.95, "evidence": ""},
                {"diagnosis": "Rib fracture", "probability": 0.05, "evidence": "break"},
            ],
            "sigma2": 0.10,
            "sc": 0.90,
            "wanna": False,
            "feedback_request": None,
            "feedback_payload": None,
            "rag_references": [],
            "temporal_note": None,
        })
        out = _parse_arll_output(raw)
        names = [h.diagnosis for h in out.ensemble.hypotheses]
        assert "with probability" not in names
        assert "Rib fracture" in names
