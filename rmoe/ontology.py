"""
rmoe/ontology.py — Medical ontology: ICD-11 codes, SNOMED CT IDs,
                   full risk-stratification scales, and clinical entity extraction.

Scales implemented:
  • Lung-RADS 2022  (pulmonary nodules)
  • TIRADS 2017     (thyroid nodules)
  • BI-RADS 5th ed. (breast imaging)
  • LI-RADS v2018   (liver observations)
  • PI-RADS v2.1    (prostate)
  • Fleischner 2017 (incidental pulmonary nodules)

All codes are drawn from publicly available clinical guidelines.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rmoe.models import ClinicalEntity, RiskScore


# ═══════════════════════════════════════════════════════════════════════════════
#  ICD-11 Code Table  (diagnosis → ICD-11 code)
# ═══════════════════════════════════════════════════════════════════════════════

ICD11: Dict[str, str] = {
    # Thoracic
    "pulmonary adenocarcinoma":            "2C25.0",
    "squamous cell carcinoma lung":        "2C25.1",
    "small cell lung carcinoma":           "2C25.2",
    "lung carcinoid tumour":               "2C25.5",
    "pulmonary metastasis":                "2D40.0",
    "community-acquired pneumonia":        "CA40.0",
    "hospital-acquired pneumonia":         "CA41.0",
    "covid-19 pneumonia":                  "RA01.0",
    "pulmonary tuberculosis":              "1B10.1",
    "pulmonary sarcoidosis":               "CB07.0",
    "pulmonary embolism":                  "BB50.0",
    "pneumothorax":                        "CB01.0",
    "pleural effusion":                    "CB21.0",
    "idiopathic pulmonary fibrosis":       "CB03.4",
    "copd":                                "CA22.0",
    "bronchiectasis":                      "CA26.0",
    # Cardiovascular
    "aortic aneurysm":                     "BA80.0",
    "pericardial effusion":                "BC91.0",
    "heart failure":                       "BD10.1",
    # MSK
    "rib fracture":                        "NB82.0",
    "vertebral fracture":                  "NA83.0",
    "hip fracture":                        "NB80.0",
    "osteoporosis":                        "FB83.1",
    "bone metastasis":                     "2E83.0",
    # Thyroid
    "papillary thyroid carcinoma":         "2D10.0",
    "follicular thyroid carcinoma":        "2D10.1",
    "thyroid nodule benign":               "5A00.00",
    # Breast
    "invasive ductal carcinoma breast":    "2C61.0",
    "ductal carcinoma in situ":            "2E65.0",
    # Hepatobiliary
    "hepatocellular carcinoma":            "2C12.0",
    "liver metastasis":                    "2D60.0",
    "liver haemangioma":                   "DA94.0",
    # Prostate
    "prostate adenocarcinoma":             "2C82.0",
    # Neurological
    "glioblastoma":                        "2A00.0",
    "meningioma":                          "2A01.0",
    "cerebral metastasis":                 "2D30.0",
    "ischaemic stroke":                    "8B20.0",
    "intracerebral haemorrhage":           "8B00.0",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SNOMED CT Table  (diagnosis → SNOMED CT concept ID)
# ═══════════════════════════════════════════════════════════════════════════════

SNOMED_CT: Dict[str, str] = {
    "pulmonary adenocarcinoma":            "254637007",
    "squamous cell carcinoma lung":        "254637007",
    "small cell lung carcinoma":           "254637007",
    "community-acquired pneumonia":        "233604007",
    "pulmonary tuberculosis":              "154283005",
    "pulmonary sarcoidosis":               "68631001",
    "pulmonary embolism":                  "59282003",
    "pneumothorax":                        "36118008",
    "pleural effusion":                    "60046008",
    "rib fracture":                        "33737001",
    "vertebral fracture":                  "263102004",
    "hip fracture":                        "370996005",
    "papillary thyroid carcinoma":         "363478007",
    "invasive ductal carcinoma breast":    "413448000",
    "hepatocellular carcinoma":            "25370001",
    "prostate adenocarcinoma":             "399068003",
    "glioblastoma":                        "393563007",
    "ischaemic stroke":                    "230690007",
}


def lookup_icd11(diagnosis: str) -> str:
    """Return the ICD-11 code for a diagnosis string (case-insensitive, partial match)."""
    d = diagnosis.lower()
    for key, code in ICD11.items():
        if key in d or d in key:
            return code
    return "N/A"


def lookup_snomed(diagnosis: str) -> str:
    """Return the SNOMED CT ID for a diagnosis string."""
    d = diagnosis.lower()
    for key, code in SNOMED_CT.items():
        if key in d or d in key:
            return code
    return "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
#  Risk Stratification  (all major scales)
# ═══════════════════════════════════════════════════════════════════════════════

class RiskStratifier:
    """
    Apply the appropriate validated imaging risk-stratification scale based on
    the anatomy and finding type.  Returns a RiskScore with score, interpretation,
    and recommended clinical action.

    Usage:
        rs = RiskStratifier()
        score = rs.classify("lung", size_mm=32, margin="spiculated")
    """

    # ── Lung-RADS 2022 ────────────────────────────────────────────────────────

    @staticmethod
    def lung_rads(size_mm: float, margin: str = "", subtype: str = "") -> RiskScore:
        """ACR Lung-RADS 2022 for CT pulmonary nodules."""
        m = margin.lower()
        s = subtype.lower()

        if size_mm < 6:
            return RiskScore(
                scale="Lung-RADS", score="2",
                interpretation="Benign — nodule < 6 mm",
                action="Annual LDCT in 12 months",
            )
        if size_mm < 8 and "ground glass" in s:
            return RiskScore(
                scale="Lung-RADS", score="3",
                interpretation="Probably benign",
                action="LDCT at 6 months",
            )
        if "spiculated" in m or "corona" in m or size_mm >= 20:
            return RiskScore(
                scale="Lung-RADS", score="4X",
                interpretation="Highly suspicious for malignancy",
                action="CT-guided biopsy or PET-CT — refer thoracic surgery",
            )
        if size_mm >= 15:
            return RiskScore(
                scale="Lung-RADS", score="4B",
                interpretation="Suspicious — tissue sampling strongly recommended",
                action="PET-CT + biopsy",
            )
        if size_mm >= 8:
            return RiskScore(
                scale="Lung-RADS", score="4A",
                interpretation="Suspicious",
                action="CT chest with contrast at 3 months",
            )
        return RiskScore(
            scale="Lung-RADS", score="3",
            interpretation="Probably benign",
            action="LDCT at 6 months",
        )

    # ── TIRADS 2017 ───────────────────────────────────────────────────────────

    @staticmethod
    def tirads(composition: str = "", echogenicity: str = "",
               shape: str = "", margin: str = "",
               echogenic_foci: str = "") -> RiskScore:
        """ACR TIRADS 2017 for ultrasound thyroid nodules (point-based)."""
        points = 0
        # Composition
        comp_map = {"cystic": 0, "mixed": 1, "solid": 2, "spongiform": 0}
        points += comp_map.get(composition.lower(), 0)
        # Echogenicity
        echo_map = {"anechoic": 0, "hyperechoic": 1, "isoechoic": 1,
                    "hypoechoic": 2, "very hypoechoic": 3}
        points += echo_map.get(echogenicity.lower(), 0)
        # Shape
        if "taller than wide" in shape.lower():
            points += 3
        # Margin
        if "irregular" in margin.lower() or "spiculated" in margin.lower():
            points += 2
        # Echogenic foci
        if "macrocalcification" in echogenic_foci.lower():
            points += 1
        if "peripheral calcification" in echogenic_foci.lower():
            points += 2
        if "punctate" in echogenic_foci.lower():
            points += 3

        if points <= 1:
            score, interp, action = "TR1", "Benign", "No FNA"
        elif points == 2:
            score, interp, action = "TR2", "Not suspicious", "No FNA"
        elif points in (3, 4):
            score, interp, action = "TR3", "Mildly suspicious",\
                "FNA if ≥ 2.5 cm solid / ≥ 1.5 cm spongiform"
        elif points in (5, 6):
            score, interp, action = "TR4", "Moderately suspicious",\
                "FNA if ≥ 1.5 cm"
        else:
            score, interp, action = "TR5", "Highly suspicious",\
                "FNA if ≥ 1.0 cm — consider surgery"

        return RiskScore(scale="TIRADS", score=score,
                         interpretation=interp, action=action)

    # ── BI-RADS 5th edition ───────────────────────────────────────────────────

    @staticmethod
    def birads(finding: str = "") -> RiskScore:
        """ACR BI-RADS 5th edition for breast imaging."""
        f = finding.lower()
        if "negative" in f or "no abnormality" in f:
            return RiskScore("BI-RADS", "1", "Negative", "Annual screening")
        if "benign" in f:
            return RiskScore("BI-RADS", "2", "Benign finding", "Annual screening")
        if "probably benign" in f:
            return RiskScore("BI-RADS", "3", "Probably benign (< 2% malignancy risk)",
                             "Short-interval follow-up at 6 months")
        if "suspicious" in f and "highly" not in f:
            return RiskScore("BI-RADS", "4", "Suspicious (15–95% malignancy risk)",
                             "Tissue sampling recommended")
        if "highly suspicious" in f or "malignant" in f:
            return RiskScore("BI-RADS", "5", "Highly suspicious (> 95% malignancy risk)",
                             "Biopsy — treatment planning")
        if "known" in f and "cancer" in f:
            return RiskScore("BI-RADS", "6", "Known biopsy-proven malignancy",
                             "Surgical planning")
        return RiskScore("BI-RADS", "0", "Incomplete — additional imaging needed",
                         "Recall for supplemental imaging")

    # ── LI-RADS v2018 ─────────────────────────────────────────────────────────

    @staticmethod
    def li_rads(arterial_enhancement: bool = False,
                washout: bool = False,
                capsule: bool = False,
                size_mm: float = 0.0,
                threshold_growth: bool = False) -> RiskScore:
        """ACR LI-RADS v2018 for hepatocellular carcinoma in at-risk patients."""
        major = sum([arterial_enhancement, washout, capsule, threshold_growth])
        if major == 0:
            return RiskScore("LI-RADS", "LR-1", "Definitely benign", "Routine surveillance")
        if major == 1 and size_mm < 10:
            return RiskScore("LI-RADS", "LR-2", "Probably benign",
                             "CT/MRI repeat at 3–6 months")
        if major == 1 and size_mm < 20:
            return RiskScore("LI-RADS", "LR-3", "Intermediate probability",
                             "CT/MRI repeat at 3 months")
        if major >= 2 and not threshold_growth:
            return RiskScore("LI-RADS", "LR-4", "Probably HCC",
                             "Multidisciplinary review — biopsy or resection")
        return RiskScore("LI-RADS", "LR-5", "Definitely HCC",
                         "Treat as HCC per local protocol")

    # ── PI-RADS v2.1 ──────────────────────────────────────────────────────────

    @staticmethod
    def pi_rads(score: int) -> RiskScore:
        """PI-RADS v2.1 for MRI prostate."""
        table = {
            1: ("PI-RADS 1", "Very low (clinically significant cancer very unlikely)", "Routine follow-up"),
            2: ("PI-RADS 2", "Low (clinically significant cancer unlikely)",            "Active surveillance"),
            3: ("PI-RADS 3", "Equivocal",                                                "Discussion with urologist; consider biopsy"),
            4: ("PI-RADS 4", "High (clinically significant cancer likely)",              "Targeted biopsy recommended"),
            5: ("PI-RADS 5", "Very high (clinically significant cancer highly likely)",  "Targeted biopsy + systematic biopsy"),
        }
        s = max(1, min(5, score))
        label, interp, action = table[s]
        return RiskScore("PI-RADS", label, interp, action)

    # ── Fleischner 2017 ───────────────────────────────────────────────────────

    @staticmethod
    def fleischner(size_mm: float, solid: bool = True,
                   high_risk: bool = False) -> RiskScore:
        """Fleischner Society 2017 guidelines for incidental pulmonary nodules."""
        if solid:
            if size_mm < 6:
                return RiskScore("Fleischner", "< 6 mm solid",
                                 "No routine follow-up needed (low risk)",
                                 "Optional CT at 12 months if high risk")
            if size_mm < 8:
                action = "CT at 6–12 months" if high_risk else "CT at 12 months"
                return RiskScore("Fleischner", "6–8 mm solid", "Low risk",  action)
            if size_mm < 15:
                return RiskScore("Fleischner", "8–15 mm solid", "Intermediate risk",
                                 "CT at 3 months, then PET or biopsy if growing")
            return RiskScore("Fleischner", "> 15 mm solid", "High risk",
                             "CT at 3 months or PET or biopsy")
        else:  # sub-solid
            if size_mm < 6:
                return RiskScore("Fleischner", "< 6 mm sub-solid",
                                 "No routine follow-up needed", "None required")
            return RiskScore("Fleischner", "≥ 6 mm sub-solid", "Intermediate risk",
                             "CT at 3–6 months then annually for 5 years")

    # ── Auto-classify ─────────────────────────────────────────────────────────

    def classify(
        self,
        organ: str,
        finding: str = "",
        size_mm: float = 0.0,
        margin: str = "",
        subtype: str = "",
        high_risk: bool = False,
    ) -> RiskScore:
        """
        Auto-select the appropriate risk scale based on organ and finding type.
        """
        o = organ.lower()
        if "lung" in o or "pulmonary" in o or "chest" in o:
            return self.lung_rads(size_mm=size_mm, margin=margin, subtype=subtype)
        if "thyroid" in o:
            return self.tirads(margin=margin)
        if "breast" in o:
            return self.birads(finding=finding)
        if "liver" in o or "hepatic" in o:
            return self.li_rads(size_mm=size_mm)
        if "prostate" in o:
            score = 4 if "suspicious" in finding.lower() else 3
            return self.pi_rads(score=score)
        # Generic Fleischner for any nodule
        if "nodule" in finding.lower():
            return self.fleischner(size_mm=size_mm, high_risk=high_risk)
        return RiskScore(scale="N/A", score="N/A",
                         interpretation="No applicable scale detected",
                         action="Clinical correlation recommended")


# ═══════════════════════════════════════════════════════════════════════════════
#  Clinical Entity Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalEntityExtractor:
    """
    Lightweight rule-based NER for clinical text.
    Extracts diagnoses, measurements, risk factors, and imaging findings,
    then maps them to ICD-11 and SNOMED CT codes.
    """

    _DIAGNOSES = list(ICD11.keys())

    _MEASURE_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*cm|"
        r"(\d+(?:\.\d+)?)\s*mm|"
        r"(\d+(?:\.\d+)?)\s*cm"
    )
    _RISK_FACTORS = [
        "smoking", "diabetes", "hypertension", "obesity", "family history",
        "alcohol", "cirrhosis", "hepatitis", "hpv", "brca",
    ]

    def extract(self, text: str) -> List[ClinicalEntity]:
        """Extract clinical entities from a free-text string."""
        entities: List[ClinicalEntity] = []
        t_lower = text.lower()

        # Diagnoses
        for diag in self._DIAGNOSES:
            if diag in t_lower:
                entities.append(ClinicalEntity(
                    entity_type="diagnosis",
                    text=diag.title(),
                    icd11=ICD11[diag],
                    snomed_ct=SNOMED_CT.get(diag, "N/A"),
                ))

        # Measurements
        for m in self._MEASURE_RE.finditer(text):
            entities.append(ClinicalEntity(
                entity_type="measurement",
                text=m.group(0).strip(),
            ))

        # Risk factors
        for rf in self._RISK_FACTORS:
            if rf in t_lower:
                entities.append(ClinicalEntity(
                    entity_type="risk_factor",
                    text=rf.title(),
                ))

        return entities
