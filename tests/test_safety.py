"""
tests/test_safety.py — Unit tests for rmoe.safety (CSR Dual-Layer Validator).

Paper §"Synthesis Agent": "The first layer is a semantic parser that
extracts entities from the reasoning logic, while the second layer is a
deterministic rule-checker that evaluates the final report against
clinical safety protocols."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.safety import (
    CSRSafetyValidator, SafetyStatus, ViolationSeverity,
    SemanticParser, ClinicalRuleChecker,
)


@pytest.fixture
def validator():
    return CSRSafetyValidator()


# ═══════════════════════════════════════════════════════════════════════════════
#  Overall validator status
# ═══════════════════════════════════════════════════════════════════════════════

def test_pass_clean_report(validator):
    """Normal, complete report → PASS status."""
    report_text = (
        "Impression: 3.2cm left upper lobe spiculated mass. "
        "ICD-11: 2C25.0 Pulmonary adenocarcinoma. "
        "Risk Stratification: Lung-RADS 4X. "
        "PET-CT recommended within 1 month per ACR Lung-RADS 2022. "
        "MDT discussion recommended. Staging: T2N0M0."
    )
    report = validator.validate(report_text)
    assert report.status == SafetyStatus.PASS


def test_block_lungrads4x_without_pet(validator):
    """Lung-RADS 4X without PET-CT → CRITICAL → BLOCK."""
    report_text = (
        "Impression: Suspicious 3.2cm LUL mass. "
        "Risk Stratification: Lung-RADS 4X. "
        "Follow-up in 3 months."
        # No PET-CT mentioned
    )
    report = validator.validate(report_text)
    assert report.status == SafetyStatus.BLOCK
    rule_ids = [v.rule_id for v in report.critical_violations()]
    assert "LUNGRADS-4X" in rule_ids


def test_warn_malignancy_without_staging(validator):
    """Malignancy ICD-11 code without staging → WARN (or BLOCK if other rules fire)."""
    report_text = (
        "Impression: Consistent with adenocarcinoma. "
        "ICD-11: 2C25.0. "
        "CT chest follow-up recommended."
        # No staging, no MDT → should trigger staging warnings
    )
    report = validator.validate(report_text)
    # At minimum, violations should be present (staging or MDT)
    all_ids = [v.rule_id for v in report.violations]
    assert any("MALIGNANCY" in rid or "STAGING" in rid or "MDT" in rid
               for rid in all_ids), f"Expected malignancy rules, got: {all_ids}"


def test_block_nsaid_renal_failure(validator):
    """NSAIDs recommended with renal failure mention → CRITICAL → BLOCK."""
    report_text = (
        "Patient has acute renal failure (AKI). "
        "Recommend ibuprofen 400mg TDS for pain management."
    )
    report = validator.validate(report_text)
    critical_ids = [v.rule_id for v in report.critical_violations()]
    assert "NSAID-RENAL" in critical_ids
    assert report.status == SafetyStatus.BLOCK


def test_block_tirads5_without_fna(validator):
    """TR5 thyroid without FNA → CRITICAL → BLOCK."""
    report_text = (
        "Thyroid nodule: solid, hypoechoic, taller-than-wide. "
        "Risk Stratification: TR5. "
        "Recommend follow-up ultrasound in 6 months."
        # No FNA recommended
    )
    report = validator.validate(report_text)
    critical_ids = [v.rule_id for v in report.critical_violations()]
    assert "TIRADS-5" in critical_ids


def test_warn_birads5_without_biopsy(validator):
    """BI-RADS 5 without biopsy → WARNING."""
    report_text = (
        "Right breast 2.1cm spiculated mass. BI-RADS 5. "
        "Recommend clinical follow-up."
        # No biopsy mentioned
    )
    report = validator.validate(report_text)
    assert report.status in (SafetyStatus.WARN, SafetyStatus.BLOCK)


# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic parser (Layer 1)
# ═══════════════════════════════════════════════════════════════════════════════

def test_parser_extracts_icd11():
    """Parser finds ICD-11 codes in report text."""
    parser = SemanticParser()
    entities, _ = parser.parse("Diagnosis: ICD-11 2C25.0 — Pulmonary adenocarcinoma.")
    icd_codes = [e.value for e in entities if e.entity_type == "icd11"]
    assert "2C25.0" in icd_codes


def test_parser_extracts_risk_score():
    """Parser finds Lung-RADS and TIRADS scores."""
    parser = SemanticParser()
    text = "Lung-RADS 4X lesion in LUL. Thyroid: TR5 nodule."
    entities, _ = parser.parse(text)
    risk_scores = {e.value.upper() for e in entities if e.entity_type == "risk_score"}
    assert any("LUNG-RADS 4X" in r or "4X" in r for r in risk_scores)
    assert any("TR5" in r for r in risk_scores)


def test_parser_extracts_dose():
    """Parser finds dose mentions."""
    parser = SemanticParser()
    entities, _ = parser.parse("Prescribe ibuprofen 400 mg TDS.")
    doses = [e.value for e in entities if e.entity_type == "dose"]
    assert any("400" in d for d in doses)


def test_parser_empty_report():
    """Empty report text → no entities, no violations."""
    parser = SemanticParser()
    entities, violations = parser.parse("")
    assert entities == []
    assert violations == []


# ═══════════════════════════════════════════════════════════════════════════════
#  SafetyReport helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_safety_report_annotation_on_warn(validator):
    """WARN status adds non-empty annotation."""
    report_text = (
        "Malignancy: ICD-11 2C25.0. Impression: Adenocarcinoma. "
        # No staging, no MDT → WARN
        "CT chest follow-up recommended."
    )
    report = validator.validate(report_text)
    if report.status == SafetyStatus.WARN:
        assert isinstance(report.annotation, str)
        # annotation should contain rule IDs
        assert len(report.annotation) > 0


def test_paediatric_dose_warning(validator):
    """High dose flagged as WARNING for paediatric patient."""
    report_text = "Prescribe amoxicillin 500 mg TDS."
    report = validator.validate(report_text, patient_age_years=3.0)
    # Should flag DRUG-DOSE-PEDS warning
    all_rule_ids = [v.rule_id for v in report.violations]
    assert any("PEDS" in rid or "DOSE" in rid for rid in all_rule_ids)
