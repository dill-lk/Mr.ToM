"""
rmoe/modality.py — Modality Escalation Router.

Implements paper §"The Recursive #wanna# Protocol":
  "Request Additional Modalities: Identify that a primary X-ray is
   insufficient for the clinical question and suggest escalation to
   CT or MRI."

This is the THIRD #wanna# routing mode (alongside High-Res Crop and
Alternate View). When the ARLL determines that current imaging is
fundamentally insufficient for a confident diagnosis, it can request
escalation to a higher-modality scan rather than simply re-cropping.

Clinical escalation pathways:
  CXR  → CT chest          (nodule characterisation, PE, aorta)
  CXR  → CT + contrast     (mediastinal mass, lymphoma staging)
  CT   → MRI               (soft tissue, spinal cord, brain)
  CT   → PET-CT            (staging, treatment response, Lung-RADS 4X)
  US   → CT / MRI          (liver characterisation, LI-RADS upgrade)
  MRI  → MRS               (brain spectroscopy, metabolite profiling)
  X-Ray → CT bone          (complex fracture, occult fracture)

EscalationRouter:
  • Takes current modality + diagnosis + confidence
  • Returns ModalityEscalation with recommended modality + clinical rationale
  • Integrates with WannaStateMachine as a third wanna action type
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class Modality(Enum):
    CXR    = "CXR"    # chest X-ray
    CT     = "CT"     # computed tomography (non-contrast)
    CT_IV  = "CT+IV"  # CT with IV contrast
    MRI    = "MRI"    # MRI (any sequence)
    MRI_GD = "MRI+Gd" # MRI with gadolinium
    PET_CT = "PET-CT"  # PET-CT (FDG unless specified)
    US     = "US"     # ultrasound
    NM     = "NM"     # nuclear medicine / SPECT
    XR     = "X-Ray"  # plain film (non-chest)
    FLUORO = "Fluoro" # fluoroscopy
    MRS    = "MRS"    # MR spectroscopy
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, s: str) -> "Modality":
        s = s.upper().strip()
        mapping = {
            "CXR": cls.CXR, "CHEST X-RAY": cls.CXR, "CHEST XRAY": cls.CXR,
            "CT": cls.CT, "CT CHEST": cls.CT, "CT ABDOMEN": cls.CT,
            "CT+IV": cls.CT_IV, "CT CONTRAST": cls.CT_IV, "CECT": cls.CT_IV,
            "MRI": cls.MRI, "MR": cls.MRI,
            "MRI+GD": cls.MRI_GD, "MRI CONTRAST": cls.MRI_GD,
            "PET": cls.PET_CT, "PET-CT": cls.PET_CT, "PET CT": cls.PET_CT,
            "US": cls.US, "ULTRASOUND": cls.US,
            "XRAY": cls.XR, "X-RAY": cls.XR, "PLAIN FILM": cls.XR,
        }
        return mapping.get(s, cls.UNKNOWN)


class EscalationUrgency(Enum):
    ROUTINE   = "Routine"      # within 4 weeks
    URGENT    = "Urgent"       # within 1 week
    EMERGENCY = "Emergency"    # same day / next working day


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityEscalation:
    """Recommended modality escalation."""
    current_modality:     Modality
    recommended_modality: Modality
    clinical_question:    str             # what the escalation will answer
    rationale:            str             # evidence-based reason
    guideline_reference:  str = ""        # e.g. "ACR Lung-RADS 2022"
    urgency:              EscalationUrgency = EscalationUrgency.ROUTINE
    wanna_payload:        str = ""        # formatted #wanna# feedback payload
    triggered_by:         str = ""        # "low_sc" | "risk_score" | "ddx_conflict"

    def to_wanna_payload(self) -> str:
        return (
            f"RequestAdditionalModality: modality={self.recommended_modality.value};"
            f"question={self.clinical_question};"
            f"urgency={self.urgency.value};"
            f"rationale={self.rationale}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Escalation rules table
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Rule:
    """One escalation rule."""
    rule_id:          str
    from_modality:    Modality
    to_modality:      Modality
    trigger_keywords: List[str]          # keywords in DDx / evidence text
    clinical_question: str
    rationale:        str
    guideline:        str
    urgency:          EscalationUrgency


_RULES: List[_Rule] = [
    # ── CXR → CT ──────────────────────────────────────────────────────────────
    _Rule("CXR-CT-NODULE", Modality.CXR, Modality.CT,
          ["nodule", "mass", "opacity", "lung-rads", "adenocarcinoma", "carcinoma"],
          "Characterise pulmonary nodule / mass density, morphology, and size",
          "CT provides superior characterisation of nodule density (solid/subsolid/GGO), "
          "spiculation, and pleural involvement vs CXR.",
          "ACR Lung-RADS 2022; Fleischner Society 2017",
          EscalationUrgency.URGENT),
    _Rule("CXR-CT-PE", Modality.CXR, Modality.CT_IV,
          ["pulmonary embolism", "pe", "hampton hump", "westermark"],
          "Confirm / exclude pulmonary embolism with CT pulmonary angiography",
          "CTPA is gold standard for PE diagnosis (sensitivity 83–100%). "
          "CXR findings (Hampton hump, Westermark) are non-specific.",
          "ESC Guidelines on PE 2019",
          EscalationUrgency.EMERGENCY),
    _Rule("CXR-CT-MEDIASTINUM", Modality.CXR, Modality.CT_IV,
          ["mediastinal", "mediastinum", "lymphoma", "thymoma"],
          "Characterise mediastinal mass — cross-sectional assessment",
          "CT with IV contrast delineates mediastinal mass compartment, "
          "vascular involvement, and lymph node stations for staging.",
          "ESMO Mediastinal Tumours Guidelines 2021",
          EscalationUrgency.URGENT),
    # ── CT → MRI ──────────────────────────────────────────────────────────────
    _Rule("CT-MRI-BRAIN", Modality.CT, Modality.MRI,
          ["glioblastoma", "glioma", "brain tumour", "cerebral", "intracranial"],
          "MRI brain for superior soft-tissue characterisation",
          "MRI provides superior soft-tissue contrast for brain tumour grading, "
          "edema extent, and leptomeningeal involvement vs CT.",
          "EANO Guidelines on Brain Tumours 2021",
          EscalationUrgency.URGENT),
    _Rule("CT-MRI-SPINE", Modality.CT, Modality.MRI,
          ["spinal cord", "myelopathy", "disc", "vertebral", "cord compression"],
          "MRI spine for cord and disc-space assessment",
          "MRI is mandatory for suspected cord compression, disc herniation, "
          "or infectious spondylodiscitis.",
          "NICE Spinal Assessment Guidelines",
          EscalationUrgency.URGENT),
    _Rule("CT-MRI-LIVER", Modality.CT, Modality.MRI_GD,
          ["hepatocellular", "hcc", "lirads", "lr-4", "lr-3", "liver lesion"],
          "MRI liver with hepatobiliary contrast for HCC characterisation",
          "Gadoxetate-enhanced MRI (LI-RADS algorithm) superior to CT for "
          "HCC detection in cirrhotic liver — hepatobiliary phase adds specificity.",
          "ACR LI-RADS 2023; EASL HCC Guidelines 2022",
          EscalationUrgency.URGENT),
    # ── CT → PET-CT ───────────────────────────────────────────────────────────
    _Rule("CT-PET-STAGING", Modality.CT, Modality.PET_CT,
          ["adenocarcinoma", "carcinoma", "malignancy", "lung-rads 4x", "lung-rads 4b",
           "staging", "metastasis"],
          "FDG PET-CT for metabolic staging and biopsy guidance",
          "FDG PET-CT provides whole-body staging, identifies hypermetabolic nodes, "
          "and guides biopsy to highest SUV lesion. Required for Lung-RADS 4X.",
          "ACR Lung-RADS 2022; NCCN NSCLC Guidelines 2024",
          EscalationUrgency.URGENT),
    # ── US → CT/MRI ───────────────────────────────────────────────────────────
    _Rule("US-CT-LIVER", Modality.US, Modality.CT_IV,
          ["liver", "hepatic", "focal lesion", "lirads"],
          "Multiphase CT liver for LI-RADS characterisation",
          "Multiphase CT with arterial / portal / delayed phases applies "
          "LI-RADS criteria for HCC characterisation.",
          "ACR LI-RADS 2023",
          EscalationUrgency.ROUTINE),
    # ── X-Ray → CT bone ───────────────────────────────────────────────────────
    _Rule("XR-CT-FRACTURE", Modality.XR, Modality.CT,
          ["fracture", "cortical", "scaphoid", "hip", "pelvis", "vertebral"],
          "CT for complex / occult fracture characterisation",
          "CT provides superior fracture pattern characterisation, "
          "particularly for scaphoid (occult), acetabular, and vertebral fractures.",
          "ACR Appropriateness Criteria — Suspected Fractures",
          EscalationUrgency.URGENT),
    # ── MRI → MRS ─────────────────────────────────────────────────────────────
    _Rule("MRI-MRS-BRAIN", Modality.MRI, Modality.MRS,
          ["glioma", "glioblastoma", "metabolite", "spectroscopy", "grade"],
          "MR spectroscopy for metabolite profiling and glioma grading",
          "MRS Cho/Cr and NAA ratios aid tumour grading and differentiation "
          "from radionecrosis / demyelination.",
          "EANO Brain Tumour MRI Protocol 2022",
          EscalationUrgency.ROUTINE),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  ModalityEscalationRouter
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityEscalationRouter:
    """
    Recommend modality escalation based on current modality + clinical context.

    Integration with WannaStateMachine:
        When wanna=True and current iteration is at limit-1, check if a
        modality escalation would resolve the ambiguity instead of
        escalating to human. If yes, emit escalation as the #wanna# payload.

    Usage:
        router = ModalityEscalationRouter()
        escalations = router.suggest(
            current_modality="CXR",
            evidence_text="3.2cm spiculated LUL mass, possible adenocarcinoma",
            ddx_labels=["Pulmonary adenocarcinoma", "Community-acquired pneumonia"],
            sc=0.72,
        )
    """

    def suggest(
        self,
        current_modality: str,
        evidence_text:    str,
        ddx_labels:       List[str],
        sc:               float = 1.0,
        risk_score:       str   = "",
        max_suggestions:  int   = 2,
    ) -> List[ModalityEscalation]:
        """
        Return up to max_suggestions ModalityEscalation objects.

        Rules are evaluated in definition order; the first matching rules
        (by keyword presence) are returned.
        """
        current = Modality.from_string(current_modality)
        combined_text = (
            " ".join(ddx_labels) + " " + evidence_text + " " + risk_score
        ).lower()

        suggestions: List[ModalityEscalation] = []

        for rule in _RULES:
            if rule.from_modality != current:
                continue
            if any(kw in combined_text for kw in rule.trigger_keywords):
                esc = ModalityEscalation(
                    current_modality=current,
                    recommended_modality=rule.to_modality,
                    clinical_question=rule.clinical_question,
                    rationale=rule.rationale,
                    guideline_reference=rule.guideline,
                    urgency=rule.urgency,
                    triggered_by="low_sc" if sc < 0.85 else "risk_score",
                )
                esc.wanna_payload = esc.to_wanna_payload()
                suggestions.append(esc)

            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def format_wanna_payload(self, escalations: List[ModalityEscalation]) -> str:
        """Format one or more escalations as a #wanna# feedback payload."""
        if not escalations:
            return ""
        e = escalations[0]
        return (
            f"#wanna# RequestAdditionalModality\n"
            f"  Current     : {e.current_modality.value}\n"
            f"  Recommended : {e.recommended_modality.value}\n"
            f"  Question    : {e.clinical_question}\n"
            f"  Rationale   : {e.rationale}\n"
            f"  Guideline   : {e.guideline_reference}\n"
            f"  Urgency     : {e.urgency.value}\n"
        )


def print_escalation_suggestion(esc: ModalityEscalation) -> None:
    """Print escalation suggestion to terminal."""
    try:
        from rmoe.ui import BOLD, CYAN, DIM, RESET, YELLOW, _rule
    except ImportError:
        BOLD = CYAN = DIM = RESET = YELLOW = ""
        def _rule(): print("─" * 72)

    urg_colour = {"Emergency": "\033[91m", "Urgent": YELLOW, "Routine": CYAN}
    uc = urg_colour.get(esc.urgency.value, "")

    print(f"\n  {BOLD}🔀 Modality Escalation Suggested{RESET}")
    print(f"  {esc.current_modality.value}  →  {BOLD}{esc.recommended_modality.value}{RESET}"
          f"  {uc}[{esc.urgency.value}]{RESET}")
    print(f"  {DIM}Clinical question:{RESET} {esc.clinical_question}")
    print(f"  {DIM}Rationale:{RESET}        {esc.rationale}")
    if esc.guideline_reference:
        print(f"  {DIM}Guideline:{RESET}        {esc.guideline_reference}")
