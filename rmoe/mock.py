"""
rmoe/mock.py — Realistic demo responses for mock/offline mode.

When llama-cpp-python is not installed (or model files are missing),
the engine falls back to these clinically plausible responses so the
full pipeline can be demonstrated without any models.

The three ARLL outputs mimic the paper's 3-iteration Sc trajectory:
  Iteration 1:  Sc ≈ 0.79  →  RequestHighResCrop
  Iteration 2:  Sc ≈ 0.86  →  RequestAlternateView
  Iteration 3:  Sc ≈ 0.94  →  ProceedToReport
"""
from __future__ import annotations

import json


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — MPE  (Moondream2 / Qwen2-VL)
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_MPE_EVIDENCE = """{
  "rois": [
    {
      "label": "Left upper lobe opacity",
      "descriptor": "Ill-defined homogeneous density, ~3.2 × 2.8 cm",
      "density": "soft-tissue",
      "margin": "irregular / spiculated",
      "suspicion": "high",
      "location": "posterior segment, left upper lobe"
    },
    {
      "label": "Mediastinal contour",
      "descriptor": "Superior mediastinum 8.4 cm — borderline for age/sex",
      "density": "soft-tissue",
      "margin": "smooth",
      "suspicion": "medium",
      "location": "superior mediastinum"
    },
    {
      "label": "Right costophrenic angle",
      "descriptor": "Sharp — no blunting, no effusion",
      "density": "air",
      "margin": "sharp",
      "suspicion": "low",
      "location": "right lower zone"
    }
  ],
  "feature_summary": "PA chest radiograph. Left upper lobe posterior segment hyperdensity with irregular / spiculated margin. No pneumothorax. Cardiac silhouette within normal limits. No acute rib fractures.",
  "confidence_level": "high",
  "saliency_crop": "120,60,380,280",
  "artifact_note": "Minor beam-hardening at left shoulder — suppressed."
}"""

MOCK_MPE_ZOOM = """{
  "rois": [
    {
      "label": "LUL spiculated margin (zoomed 2.5×)",
      "descriptor": "Confirmed spiculation, corona radiata pattern, ~3.2 × 2.8 cm",
      "density": "soft-tissue",
      "margin": "spiculated",
      "suspicion": "high",
      "location": "posterior segment, left upper lobe"
    }
  ],
  "feature_summary": "High-resolution crop confirms spiculated lesion at 2.5× zoom. Corona radiata pattern visible. No satellite nodules identified at this resolution.",
  "confidence_level": "high",
  "saliency_crop": "135,70,360,265",
  "artifact_note": "None."
}"""

MOCK_MPE_LATERAL = """{
  "rois": [
    {
      "label": "Posterior LUL mass on lateral view",
      "descriptor": "3.2 cm opacity projected posterior to the oblique fissure — no pleural contact",
      "density": "soft-tissue",
      "margin": "spiculated",
      "suspicion": "high",
      "location": "posterior LUL, no pleural spread confirmed"
    }
  ],
  "feature_summary": "Lateral projection confirms posterior LUL location. No pleural effusion. No posterior costophrenic blunting. Mass does not contact the chest wall.",
  "confidence_level": "high",
  "saliency_crop": "90,55,310,250",
  "artifact_note": "None."
}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — ARLL  (DeepSeek-R1-Distill)
# ═══════════════════════════════════════════════════════════════════════════════

# Three-iteration JSON outputs matching the paper's Sc trajectory.
MOCK_ARLL_OUTPUTS = [
    # ── Iteration 1: Sc = 0.7923  →  High-Res Crop ───────────────────────────
    json.dumps({
        "cot": (
            "Step 1 — Evidence review: MPE identified a 3.2×2.8 cm ill-defined "
            "density in the posterior left upper lobe with irregular margin. "
            "Step 2 — Prior imaging not available; temporal comparison not possible. "
            "Step 3 — DDx construction: upper-lobe spiculated lesion carries broad "
            "differential. Ensemble pass 1: adenocarcinoma (0.42), CAP (0.31), "
            "sarcoidosis (0.15), TB (0.12). "
            "Step 4 — σ² = 0.0207, Sc = 0.7923 < 0.90. Margin characterisation "
            "insufficient at standard resolution. "
            "Action: request 2.5× high-resolution crop of left upper lobe."
        ),
        "ddx": [
            {"diagnosis": "Pulmonary adenocarcinoma",
             "probability": 0.42,
             "evidence": "Spiculated / irregular margin, posterior LUL — classic adenocarcinoma location"},
            {"diagnosis": "Community-acquired pneumonia",
             "probability": 0.31,
             "evidence": "Homogeneous density could represent consolidation; air bronchogram not excluded"},
            {"diagnosis": "Pulmonary sarcoidosis",
             "probability": 0.15,
             "evidence": "Upper-lobe predilection; bilateral hilar adenopathy not yet confirmed"},
            {"diagnosis": "Tuberculosis reactivation",
             "probability": 0.12,
             "evidence": "Upper-lobe location consistent; no cavitation visible at current resolution"},
        ],
        "sigma2": 0.0207,
        "sc": 0.7923,
        "wanna": True,
        "feedback_request": "High-Res Crop",
        "feedback_payload": "region=left_upper_lobe;zoom=2.5",
        "rag_references": [
            "MIMIC-CXR: spiculated nodule → malignancy PPV = 0.71 (95% CI 0.65–0.77)",
            "ACR Lung-RADS 4A: spiculated 8–20 mm nodule — 6-week follow-up CT recommended",
        ],
        "temporal_note": None,
    }),

    # ── Iteration 2: Sc = 0.8587  →  Alternate View (lateral) ────────────────
    json.dumps({
        "cot": (
            "Step 1 — High-res crop at 2.5× confirms spiculated (corona radiata) "
            "pattern. Consolidation / CAP now less likely. "
            "Step 2 — Ensemble re-run: mass increases to 0.58; CAP drops to 0.19. "
            "σ² = 0.0312, Sc = 0.8587 — still < 0.90. "
            "Step 3 — Pleural involvement remains uncertain on PA-only view. "
            "Action: request lateral projection to assess posterior pleural space "
            "and confirm lesion location relative to oblique fissure."
        ),
        "ddx": [
            {"diagnosis": "Pulmonary adenocarcinoma",
             "probability": 0.58,
             "evidence": "Spiculated margin confirmed on 2.5× crop; corona radiata pattern"},
            {"diagnosis": "Community-acquired pneumonia",
             "probability": 0.19,
             "evidence": "Air bronchogram absent on high-res crop — consolidation less likely"},
            {"diagnosis": "Pulmonary sarcoidosis",
             "probability": 0.13,
             "evidence": "No bilateral hilar prominence confirmed"},
            {"diagnosis": "Tuberculosis reactivation",
             "probability": 0.10,
             "evidence": "No cavitation, no satellite nodules on high-res crop"},
        ],
        "sigma2": 0.0312,
        "sc": 0.8587,
        "wanna": True,
        "feedback_request": "Alternate View",
        "feedback_payload": "region=left_upper_lobe;angle=lateral",
        "rag_references": [
            "MIMIC-CXR: confirmed spiculation → malignancy PPV = 0.81",
            "RSNA 2023: lateral view essential for pleural staging before biopsy",
        ],
        "temporal_note": None,
    }),

    # ── Iteration 3: Sc = 0.9442  →  ProceedToReport ─────────────────────────
    json.dumps({
        "cot": (
            "Step 1 — Lateral view confirms mass lies posterior to the oblique fissure, "
            "no pleural contact, no effusion. "
            "Step 2 — All ensemble passes now converge: malignancy posterior 0.72. "
            "σ² = 0.0558, Sc = 0.9442 ≥ 0.90. "
            "Step 3 — RAG cross-reference: MIMIC-CXR F1 = 0.92 benchmark met. "
            "ACR Lung-RADS 4X criteria satisfied (≥ 20 mm, spiculated). "
            "Verdict: proceed to CSR for ICD-11 classification and treatment protocol."
        ),
        "ddx": [
            {"diagnosis": "Pulmonary adenocarcinoma",
             "probability": 0.72,
             "evidence": "Spiculated margin + posterior LUL + no pleural spread + corona radiata"},
            {"diagnosis": "Community-acquired pneumonia",
             "probability": 0.11,
             "evidence": "Image features inconsistent; no air bronchogram, no lobar distribution"},
            {"diagnosis": "Pulmonary sarcoidosis",
             "probability": 0.10,
             "evidence": "No bilateral hilar adenopathy on lateral"},
            {"diagnosis": "Tuberculosis reactivation",
             "probability": 0.07,
             "evidence": "No satellite nodules, no cavitation, no upper-zone fibrosis"},
        ],
        "sigma2": 0.0558,
        "sc": 0.9442,
        "wanna": False,
        "feedback_request": None,
        "feedback_payload": None,
        "rag_references": [
            "MIMIC-CXR F1 = 0.92 benchmark achieved",
            "ACR Lung-RADS 4X: highly suspicious — tissue sampling required",
        ],
        "temporal_note": "No prior imaging available for temporal interval comparison.",
    }),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — CSR  (MedGemma-2B)
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_CSR_REPORT = json.dumps({
    "standard": "ICD-11: 2C25.0",
    "snomed_ct": "254637007",
    "risk_stratification": {
        "scale": "Lung-RADS",
        "score": "4X",
        "interpretation": "Highly suspicious for malignancy",
        "action": "Tissue sampling required — CT-guided biopsy or VATS resection per MDT",
    },
    "narrative": (
        "CLINICAL HISTORY: Incidental left upper lobe (LUL) opacity detected on PA "
        "chest radiograph.  No prior imaging available for comparison.\n\n"
        "TECHNIQUE: PA and lateral chest radiograph.  R-MoE pipeline executed with "
        "3 recursive iterations (Sc progression: 0.79 → 0.86 → 0.94).\n\n"
        "FINDINGS: A 3.2 × 2.8 cm ill-defined, spiculated opacity is identified in "
        "the posterior segment of the left upper lobe.  A corona radiata pattern is "
        "confirmed on high-resolution crop (2.5× zoom, iteration 2).  The mass lies "
        "posterior to the oblique fissure with no pleural contact (confirmed on lateral, "
        "iteration 3).  No pleural effusion.  No mediastinal or hilar lymphadenopathy.  "
        "Cardiac silhouette is within normal limits.  No pneumothorax.  No acute rib "
        "fractures.  The right lung is clear.\n\n"
        "IMPRESSION: Spiculated 3.2 cm LUL mass satisfying ACR Lung-RADS 4X criteria "
        "(highly suspicious for malignancy).  Pulmonary adenocarcinoma is the leading "
        "differential diagnosis (DDx posterior probability 0.72 at final iteration; "
        "Sc = 0.9442).  Infection and inflammatory aetiologies considered unlikely "
        "given morphological features and absence of clinical context favouring "
        "acute infection."
    ),
    "summary": (
        "3.2 cm spiculated posterior LUL mass — Lung-RADS 4X.  "
        "Primary lung malignancy (adenocarcinoma) is the leading diagnosis."
    ),
    "treatment_recommendations": (
        "1. Urgent CT thorax with contrast (within 1–2 weeks) for full lesion "
        "characterisation and mediastinal / hilar nodal staging.\n"
        "2. PET-CT if CT confirms the lesion ≥ 8 mm with no contraindication.\n"
        "3. CT-guided core biopsy or VATS resection per multidisciplinary team (MDT) "
        "decision.\n"
        "4. Refer urgently to thoracic surgery and oncology.\n"
        "5. Smoking cessation counselling if applicable.\n"
        "6. Pulmonary function tests prior to any surgical intervention."
    ),
    "hitl_review_required": False,
    "hitl_reason": "",
    "recursive_iterations": 3,
    "final_sc": 0.9442,
    "final_sigma2": 0.0558,
    "ece_estimate": 0.08,
}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_mpe_output(iteration: int, feedback_type: str = "") -> str:
    """Return the appropriate mock MPE output for a given iteration."""
    if "Alternate View" in feedback_type or "lateral" in feedback_type.lower():
        return MOCK_MPE_LATERAL
    if "High-Res Crop" in feedback_type or iteration > 1:
        return MOCK_MPE_ZOOM
    return MOCK_MPE_EVIDENCE


def get_arll_output(iteration: int) -> str:
    """Return iteration-indexed ARLL mock output (clamped to available count)."""
    idx = min(max(iteration - 1, 0), len(MOCK_ARLL_OUTPUTS) - 1)
    return MOCK_ARLL_OUTPUTS[idx]


def get_csr_output() -> str:
    """Return the mock CSR clinical report JSON."""
    return MOCK_CSR_REPORT
