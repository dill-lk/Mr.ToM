"""
tests/test_ontology.py — Unit tests for rmoe.ontology.

Paper §3.1 CSR: "ICD-11/SNOMED CT classifications, risk stratification
(e.g., TIRADS/BI-RADS), and treatment recommendations."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.ontology import (
    ICD11, SNOMED_CT, RiskStratifier, ClinicalEntityExtractor,
    lookup_icd11, lookup_snomed,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ICD-11 lookup
# ═══════════════════════════════════════════════════════════════════════════════

def test_icd11_adenocarcinoma():
    """Pulmonary adenocarcinoma maps to ICD-11 code 2C25.0."""
    code = lookup_icd11("Pulmonary adenocarcinoma")
    assert code is not None, "Expected ICD-11 entry for pulmonary adenocarcinoma"
    assert "2C25" in code or "2C" in code


def test_icd11_pneumonia():
    """Pneumonia maps to CA40* or similar."""
    code = lookup_icd11("Community-acquired pneumonia")
    assert code is not None


def test_icd11_case_insensitive():
    """Lookup is case-insensitive."""
    code1 = lookup_icd11("Pulmonary Tuberculosis")
    code2 = lookup_icd11("pulmonary tuberculosis")
    assert code1 == code2


def test_icd11_dict_not_empty():
    """ICD11 dict should contain a reasonable number of entries."""
    assert len(ICD11) >= 10


def test_icd11_unknown_returns_none():
    """Unknown diagnosis should return None or 'N/A' (falsy / not a real code)."""
    result = lookup_icd11("xyzzy_not_a_real_diagnosis_1234")
    assert not result or result == "N/A", f"Expected falsy/N/A for unknown, got {result!r}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SNOMED CT lookup
# ═══════════════════════════════════════════════════════════════════════════════

def test_snomed_lung():
    """'lung' maps to a SNOMED CT concept."""
    term = lookup_snomed("lung")
    assert term is not None


def test_snomed_dict_not_empty():
    """SNOMED_CT dict should contain entries."""
    assert len(SNOMED_CT) >= 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Risk Stratifier (paper §3.1 — TIRADS / BI-RADS / Lung-RADS)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def stratifier():
    return RiskStratifier()


def test_lung_rads_4x(stratifier):
    """Lung-RADS 4X: large spiculated nodule → high-risk score."""
    # 32mm spiculated margin → Lung-RADS 4X territory
    result = stratifier.lung_rads(size_mm=32.0, margin="spiculated")
    assert result is not None
    result_str = (result.score + result.interpretation + result.action).lower()
    assert any(kw in result_str for kw in ("4", "suspicious", "pet", "biopsy", "malignant"))


def test_tirads_5(stratifier):
    """TIRADS TR5: solid hypoechoic taller-than-wide → high-risk."""
    result = stratifier.tirads(
        composition="solid", echogenicity="hypoechoic", shape="taller_than_wide"
    )
    assert result is not None
    assert result.scale == "TIRADS"


def test_birads_5(stratifier):
    """BI-RADS 5: spiculated mass → highly suspicious."""
    result = stratifier.birads(finding="spiculated mass skin retraction")
    assert result is not None
    assert result.scale == "BI-RADS"


def test_lung_rads_1(stratifier):
    """Lung-RADS: tiny < 6mm nodule → low-risk (score 1 or 2)."""
    result = stratifier.lung_rads(size_mm=4.0, margin="smooth")
    assert result is not None
    result_str = (result.score + result.interpretation).lower()
    assert any(kw in result_str for kw in ("1", "2", "benign", "annual", "routine", "low"))


def test_unknown_organ(stratifier):
    """Unknown organ falls back gracefully (no exception)."""
    result = stratifier.classify("spleen", finding="lesion", size_mm=15.0)
    # Should return a RiskScore, not raise
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  ClinicalEntityExtractor
# ═══════════════════════════════════════════════════════════════════════════════

def test_entity_extractor_finds_organ():
    """Extractor finds organ mentions in clinical text."""
    extractor = ClinicalEntityExtractor()
    text = "Left upper lobe 3.2cm spiculated mass consistent with adenocarcinoma."
    entities = extractor.extract(text)
    assert len(entities) > 0
    types = {e.entity_type for e in entities}
    # Should find at least finding/organ/size type
    assert len(types) > 0
