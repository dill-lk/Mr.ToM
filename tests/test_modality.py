"""
tests/test_modality.py — Unit tests for rmoe.modality (Modality Escalation Router).

Paper §"#wanna# Protocol" — third routing mode:
  "Request Additional Modalities: Identify that a primary X-ray is
   insufficient for the clinical question and suggest escalation to CT or MRI."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.modality import (
    ModalityEscalationRouter, Modality, EscalationUrgency,
)


@pytest.fixture
def router():
    return ModalityEscalationRouter()


# ═══════════════════════════════════════════════════════════════════════════════
#  CXR escalation pathways
# ═══════════════════════════════════════════════════════════════════════════════

def test_cxr_to_ct_nodule(router):
    """CXR + nodule mention → suggest CT chest (paper Lung-RADS pathway)."""
    suggestions = router.suggest(
        current_modality="CXR",
        evidence_text="3.2cm spiculated nodule left upper lobe",
        ddx_labels=["Pulmonary adenocarcinoma", "Community-acquired pneumonia"],
        sc=0.75,
    )
    assert len(suggestions) > 0
    modalities = [s.recommended_modality for s in suggestions]
    assert any(m in (Modality.CT, Modality.CT_IV) for m in modalities)


def test_cxr_to_ctpa_for_pe(router):
    """CXR + PE mention → suggest CT pulmonary angiography (CTPA)."""
    suggestions = router.suggest(
        current_modality="CXR",
        evidence_text="Hampton hump sign, Westermark sign",
        ddx_labels=["Pulmonary embolism", "Pleural effusion"],
        sc=0.65,
    )
    assert len(suggestions) > 0
    # PE escalation should be EMERGENCY urgency
    urgencies = [s.urgency for s in suggestions]
    assert any(u == EscalationUrgency.EMERGENCY for u in urgencies)


def test_cxr_to_ct_mediastinum(router):
    """Mediastinal mass on CXR → CT with IV contrast."""
    suggestions = router.suggest(
        current_modality="CXR",
        evidence_text="superior mediastinum widened",
        ddx_labels=["Mediastinal lymphoma", "Thymoma"],
        sc=0.72,
    )
    assert len(suggestions) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  CT escalation pathways
# ═══════════════════════════════════════════════════════════════════════════════

def test_ct_to_mri_brain(router):
    """CT + glioblastoma → suggest MRI brain."""
    suggestions = router.suggest(
        current_modality="CT",
        evidence_text="ring-enhancing mass, suspected glioblastoma",
        ddx_labels=["Glioblastoma", "Brain metastasis"],
        sc=0.78,
    )
    assert len(suggestions) > 0
    modalities = [s.recommended_modality for s in suggestions]
    assert any(m in (Modality.MRI, Modality.MRI_GD) for m in modalities)


def test_ct_to_petct_staging(router):
    """CT + adenocarcinoma → suggest PET-CT for staging."""
    suggestions = router.suggest(
        current_modality="CT",
        evidence_text="3.8cm adenocarcinoma LUL requiring staging",
        ddx_labels=["Pulmonary adenocarcinoma"],
        sc=0.80,
        risk_score="Lung-RADS 4X",
    )
    assert len(suggestions) > 0
    modalities = [s.recommended_modality for s in suggestions]
    assert Modality.PET_CT in modalities


# ═══════════════════════════════════════════════════════════════════════════════
#  No escalation when already on highest modality for context
# ═══════════════════════════════════════════════════════════════════════════════

def test_mri_no_ct_escalation(router):
    """MRI already done — no CT escalation for brain case."""
    suggestions = router.suggest(
        current_modality="MRI",
        evidence_text="ring-enhancing mass brain glioblastoma",
        ddx_labels=["Glioblastoma"],
        sc=0.85,
    )
    # MRI rules only apply if there are MRI → ??? rules defined
    # Currently only MRI → MRS is defined; no MRI → CT rule
    ct_suggestions = [s for s in suggestions
                      if s.recommended_modality in (Modality.CT, Modality.CT_IV)]
    assert len(ct_suggestions) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Modality.from_string
# ═══════════════════════════════════════════════════════════════════════════════

def test_modality_from_string_cxr():
    assert Modality.from_string("CXR") == Modality.CXR
    assert Modality.from_string("CHEST X-RAY") == Modality.CXR
    assert Modality.from_string("chest xray") == Modality.CXR


def test_modality_from_string_ct():
    assert Modality.from_string("CT") == Modality.CT
    assert Modality.from_string("ct chest") == Modality.CT


def test_modality_from_string_mri():
    assert Modality.from_string("MRI") == Modality.MRI
    assert Modality.from_string("mr") == Modality.MRI


def test_modality_from_string_petct():
    assert Modality.from_string("PET-CT") == Modality.PET_CT
    assert Modality.from_string("pet ct") == Modality.PET_CT


def test_modality_from_string_unknown():
    assert Modality.from_string("SPECT_BONE") == Modality.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
#  max_suggestions parameter
# ═══════════════════════════════════════════════════════════════════════════════

def test_max_suggestions_respected(router):
    """Never return more than max_suggestions."""
    suggestions = router.suggest(
        current_modality="CXR",
        evidence_text="nodule mass pulmonary embolism mediastinum",
        ddx_labels=["Adenocarcinoma", "PE", "Lymphoma"],
        sc=0.60,
        max_suggestions=1,
    )
    assert len(suggestions) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Wanna payload format
# ═══════════════════════════════════════════════════════════════════════════════

def test_wanna_payload_format(router):
    """format_wanna_payload returns string starting with #wanna#."""
    suggestions = router.suggest(
        current_modality="CXR",
        evidence_text="spiculated nodule suspicious for malignancy",
        ddx_labels=["Pulmonary adenocarcinoma"],
        sc=0.72,
    )
    if suggestions:
        payload = router.format_wanna_payload(suggestions)
        assert "#wanna#" in payload
        assert "RequestAdditionalModality" in payload
