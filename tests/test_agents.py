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


# ═══════════════════════════════════════════════════════════════════════════════
#  _parse_mpe_evidence  —  natural language fallback must not truncate
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseMpeEvidence:
    """_parse_mpe_evidence must preserve the full model output in feature_summary
    when the model outputs natural language instead of JSON."""

    def test_valid_json_parsed(self):
        """Clean JSON block is parsed into proper PerceptionEvidence fields."""
        from rmoe.agents import _parse_mpe_evidence
        raw = json.dumps({
            "rois": [{"label": "LUL opacity", "suspicion": "high"}],
            "feature_summary": "Spiculated left upper lobe opacity.",
            "confidence_level": "high",
            "saliency_crop": "120,60,380,280",
        })
        ev = _parse_mpe_evidence(raw)
        assert ev.feature_summary == "Spiculated left upper lobe opacity."
        assert ev.confidence_level == "high"
        assert ev.saliency_crop == "120,60,380,280"
        assert len(ev.rois) == 1

    def test_natural_language_fallback_preserves_full_text(self):
        """When the model outputs prose, feature_summary must equal the full
        raw string — the old code truncated at 300 chars which silently
        discarded most of what the vision model said."""
        from rmoe.agents import _parse_mpe_evidence
        # Generate a long natural-language description (> 300 chars, the old truncation limit)
        long_output = (
            "I can see a PA chest radiograph. There is an ill-defined opacity "
            "in the left upper lobe with irregular, spiculated margins. "
            "The lesion measures approximately 3.2 by 2.8 centimetres. "
            "The mediastinum is not widened. No pleural effusion is visible. "
            "The cardiac silhouette is within normal limits. "
            "The right lung fields appear clear. "
            "No pneumothorax is identified on this projection."
        )
        assert len(long_output) > 300, "test prerequisite: output must exceed the old 300-char truncation limit"
        ev = _parse_mpe_evidence(long_output)
        assert ev.feature_summary == long_output
        assert ev.raw_summary == long_output

    def test_natural_language_fallback_defaults(self):
        """Fallback PerceptionEvidence has empty rois and medium confidence."""
        from rmoe.agents import _parse_mpe_evidence
        ev = _parse_mpe_evidence("Some free-form text from the vision model.")
        assert ev.rois == []
        assert ev.confidence_level == "medium"

    def test_json_with_missing_feature_summary_falls_back_to_raw(self):
        """JSON block with no feature_summary field → feature_summary must equal
        the full raw text so the downstream reasoning expert is not starved."""
        from rmoe.agents import _parse_mpe_evidence
        raw = json.dumps({
            "rois": [{"label": "LUL opacity", "suspicion": "high"}],
            "confidence_level": "high",
            "saliency_crop": "120,60,380,280",
            # feature_summary intentionally absent
        })
        ev = _parse_mpe_evidence(raw)
        assert ev.feature_summary == raw, (
            "When feature_summary is absent from the JSON blob, "
            "feature_summary must fall back to the full raw text"
        )

    def test_json_with_empty_feature_summary_falls_back_to_raw(self):
        """JSON block with feature_summary='' → feature_summary must equal
        the full raw text (empty string is falsy, so raw is used)."""
        from rmoe.agents import _parse_mpe_evidence
        raw = json.dumps({
            "rois": [],
            "feature_summary": "",
            "confidence_level": "low",
            "saliency_crop": "",
        })
        ev = _parse_mpe_evidence(raw)
        assert ev.feature_summary == raw, (
            "When feature_summary is an empty string in the JSON blob, "
            "feature_summary must fall back to the full raw text"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MCVBuilder  —  feature_summary attribute fix
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCVBuilderAttributeFix:
    """MCVBuilder.build() must use evidence.feature_summary (not .summary)."""

    def test_evidence_text_populated_from_feature_summary(self):
        """MCVBuilder must read feature_summary from PerceptionEvidence and
        use it for region-feature extraction and token budget estimation."""
        from rmoe.mcv import MCVBuilder
        from rmoe.models import PerceptionEvidence

        evidence = PerceptionEvidence(
            feature_summary=(
                "Left upper lobe opacity with irregular margin. "
                "Mediastinum widened. Pleural effusion noted in right lower zone."
            ),
            confidence_level="high",
            saliency_crop="100,50,400,300",
        )
        mcv = MCVBuilder().build(evidence, modality="CXR")

        # evidence_text must be the feature_summary, not empty string
        assert mcv.evidence_text != "", (
            "MCVBuilder.evidence_text must be populated from feature_summary, not empty"
        )
        assert "opacity" in mcv.evidence_text.lower()

    def test_region_features_extracted_from_feature_summary(self):
        """Region keywords in feature_summary must produce spatial features."""
        from rmoe.mcv import MCVBuilder
        from rmoe.models import PerceptionEvidence

        evidence = PerceptionEvidence(
            feature_summary="Consolidation in the left upper lobe, mediastinum normal.",
        )
        mcv = MCVBuilder().build(evidence, modality="CXR")
        region_names = [sf.region for sf in mcv.spatial_features]
        assert "left upper lobe" in region_names, (
            "Expected 'left upper lobe' region feature from feature_summary text"
        )

    def test_intensity_profile_derived_from_feature_summary(self):
        """Density keywords in feature_summary must produce intensity descriptors."""
        from rmoe.mcv import MCVBuilder
        from rmoe.models import PerceptionEvidence

        evidence = PerceptionEvidence(
            feature_summary="Ground glass opacity in left upper lobe.",
        )
        mcv = MCVBuilder().build(evidence, modality="CXR")
        assert "ground glass" in mcv.intensity_profile, (
            "Expected 'ground glass' intensity profile entry from feature_summary"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  _parse_arll_output  —  CoT must not be truncated
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseArllOutputCotNotTruncated:
    """The CoT field must carry the full reasoning text to the CSR (MedGemma)."""

    def _long_cot(self) -> str:
        """Build a CoT string that is longer than the old 300/500-char truncation limits."""
        return (
            "Step 1 — Evidence review: MPE identified a 3.2×2.8 cm spiculated "
            "opacity in the posterior left upper lobe with irregular margin. "
            "Step 2 — Prior imaging not available; temporal comparison not possible. "
            "Step 3 — DDx construction: upper-lobe spiculated lesion carries broad "
            "differential including adenocarcinoma, community-acquired pneumonia, "
            "sarcoidosis, and TB reactivation. "
            "Step 4 — Ensemble analysis: adenocarcinoma (0.58), CAP (0.19), "
            "sarcoidosis (0.13), TB (0.10). "
            "Step 5 — σ² = 0.0312, Sc = 0.8587 — still below 0.90 threshold. "
            "Recommend lateral projection to confirm posterior pleural space."
        )

    def test_json_path_cot_not_truncated(self):
        """When ARLL outputs valid JSON with a 'cot' field, the full value is preserved."""
        long_cot = self._long_cot()
        assert len(long_cot) > 300, "test prerequisite: CoT must exceed old 300-char limit"
        raw = json.dumps({
            "cot": long_cot,
            "ddx": [
                {"diagnosis": "Pulmonary adenocarcinoma", "probability": 0.58, "evidence": "spiculated"},
            ],
            "sigma2": 0.03, "sc": 0.87,
            "wanna": True, "feedback_request": "Alternate View",
            "feedback_payload": "region=LUL;angle=lateral",
            "rag_references": [], "temporal_note": None,
        })
        out = _parse_arll_output(raw)
        assert out.cot == long_cot, "CoT must equal the full value from the JSON blob"

    def test_json_path_cot_fallback_uses_full_raw(self):
        """When ARLL JSON has no 'cot' field, the full raw string is used — not raw[:300]."""
        long_cot = self._long_cot()
        assert len(long_cot) > 300
        raw = json.dumps({
            # 'cot' key intentionally absent
            "ddx": [{"diagnosis": "Rib fracture", "probability": 0.80, "evidence": "break"}],
            "sigma2": 0.02, "sc": 0.98,
            "wanna": False, "feedback_request": None,
            "feedback_payload": None,
            "rag_references": [], "temporal_note": None,
        })
        out = _parse_arll_output(raw)
        assert out.cot == raw, (
            "When 'cot' is missing from ARLL JSON, cot must be the full raw output"
        )

    def test_regex_fallback_cot_uses_full_raw(self):
        """When ARLL outputs plain text (no JSON), the full text is stored as cot."""
        long_prose = self._long_cot()
        # Append a valid diagnosis:probability pair so the regex fallback finds something
        raw = long_prose + "\nRib fracture: 0.75 — cortical break."
        assert len(raw) > 500, "test prerequisite: must exceed old 500-char limit"
        out = _parse_arll_output(raw)
        assert out.cot == raw, (
            "Regex fallback must store the full raw text in cot, not raw[:500]"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ReportingExpert  —  cot-or-raw_output fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportingExpertCotFallback:
    """ReportingExpert must send reasoning.cot to CSR; when cot is empty it must
    fall back to reasoning.raw_output so MedGemma always has input."""

    def _run_and_capture(self, cot: str, raw_output: str) -> str:
        """Run ReportingExpert with mocked inference and return the user_input
        that was passed to infer_text()."""
        import json as _json
        from unittest.mock import patch
        from rmoe.agents import ReportingExpert, ExpertSwapper
        from rmoe.models import ReasoningOutput, DDxEnsemble, DDxHypothesis

        captured: dict = {}
        mock_response = _json.dumps({
            "standard": "ICD-11", "snomed_ct": "N/A",
            "risk_stratification": {"scale": "N/A", "score": "N/A",
                                    "interpretation": "", "action": ""},
            "narrative": "test", "summary": "test",
            "treatment_recommendations": "none",
            "hitl_review_required": False, "hitl_reason": "",
        })

        swapper = ExpertSwapper()

        def fake_infer_text(system_prompt, user_input, **kw):
            captured["user_input"] = user_input
            return mock_response

        reasoning = ReasoningOutput(
            cot=cot,
            raw_output=raw_output,
            ensemble=DDxEnsemble([DDxHypothesis("Rib fracture", 0.90)]),
        )
        rpt = ReportingExpert(swapper)
        # Patch both _HAS_LLAMA_CPP (so mock path is skipped) and infer_text
        with patch("rmoe.agents._HAS_LLAMA_CPP", True):
            with patch.object(swapper, "infer_text", side_effect=fake_infer_text):
                rpt.execute(reasoning, iterations_used=1)

        return captured.get("user_input", "")

    def test_cot_used_when_present(self):
        """When cot is populated it is included in the user_input passed to CSR."""
        cot_text = "Full chain-of-thought reasoning here."
        user_input = self._run_and_capture(cot=cot_text, raw_output="raw full output")
        assert cot_text in user_input, (
            "reasoning.cot must appear in the prompt sent to CSR"
        )

    def test_raw_output_fallback_when_cot_empty(self):
        """When reasoning.cot is empty, raw_output must be used instead."""
        raw_output = "Full ARLL raw output text — no cot field was parsed."
        user_input = self._run_and_capture(cot="", raw_output=raw_output)
        assert raw_output in user_input, (
            "When reasoning.cot is empty, reasoning.raw_output must be used in CSR prompt"
        )
