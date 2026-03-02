"""
rmoe/models.py — All dataclasses, enums and typed structures for R-MoE v2.0.

Paper: "Recursive Multi-Agent Mixture-of-Experts (RMoE) for Autonomous
       Clinical Diagnostics"

Every field has a concise docstring explaining its role in the pipeline.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class WannaState(Enum):
    """Possible outcomes of the ARLL confidence gate."""
    ProceedToReport      = "ProceedToReport"       # Sc ≥ θ → go to CSR
    RequestHighResCrop   = "RequestHighResCrop"     # Sc < θ, iter < limit
    RequestAlternateView = "RequestAlternateView"   # Sc < θ, iter < limit
    EscalateToHuman      = "EscalateToHuman"        # iter == hard limit


class HITLMode(Enum):
    """Doctor-in-the-loop interaction mode."""
    Interactive = "interactive"   # always prompt doctor
    Auto        = "auto"          # prompt only in TTY sessions
    Disabled    = "disabled"      # fully autonomous


class ExpertTarget(Enum):
    """Which expert to query in post-diagnosis Q&A."""
    Reasoning = "reasoning"   # ARLL / DeepSeek-R1-Distill
    Clinical  = "clinical"    # CSR  / MedGemma-2B


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceParams:
    """
    Hyper-parameters forwarded to llama-cpp-python at load and sample time.
    Defaults are tuned for T4 (16 GB) with 2B-class quantised models.
    """
    n_ctx: int            = 2048   # KV-cache context window
    n_threads: int        = 4      # CPU decode threads
    n_threads_batch: int  = 4      # CPU prompt-eval threads
    max_new_tokens: int   = 512    # generation budget per inference call
    temperature: float    = 0.2    # paper §4.2: 0.2 for clinical precision
    top_k: int            = 40
    top_p: float          = 0.95
    repeat_penalty: float = 1.1
    penalty_last_n: int   = 64
    n_gpu_layers: int     = -1     # -1 = offload ALL layers (full GPU)


# ═══════════════════════════════════════════════════════════════════════════════
#  Differential Diagnosis  (DDx) types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DDxHypothesis:
    """Single candidate diagnosis with its probability mass and evidence string."""
    diagnosis: str    = ""
    probability: float = 0.0   # p ∈ [0, 1]
    evidence: str     = ""


@dataclass
class DDxEnsemble:
    """
    Collection of DDx hypotheses from the ARLL agent.

    Confidence score (paper §3.1):
        Sc = 1 − σ²
        σ² = Var(p₁ … pₙ)   over the DDx probability distribution.

    Rationale: when the model is highly confident, all probability mass
    sits on one diagnosis → σ² ≈ 0 → Sc ≈ 1.
    When mass is spread across many diagnoses → σ² large → Sc small.
    """
    hypotheses: List[DDxHypothesis] = field(default_factory=list)

    # ── Derived statistics ────────────────────────────────────────────────────

    @property
    def probabilities(self) -> List[float]:
        return [h.probability for h in self.hypotheses]

    @property
    def sigma2(self) -> float:
        """Variance σ² of the DDx probability distribution."""
        probs = self.probabilities
        if not probs:
            return 1.0
        mu = sum(probs) / len(probs)
        return sum((p - mu) ** 2 for p in probs) / len(probs)

    @property
    def sc(self) -> float:
        """Confidence score Sc = 1 − σ² ∈ [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.sigma2))

    @property
    def primary(self) -> Optional[DDxHypothesis]:
        """Hypothesis with highest probability mass."""
        return max(self.hypotheses, key=lambda h: h.probability) if self.hypotheses else None

    def is_confident(self, threshold: float = 0.90) -> bool:
        return self.sc >= threshold

    def entropy(self) -> float:
        """Shannon entropy H(P) of the DDx distribution (nats)."""
        e = 0.0
        for p in self.probabilities:
            if p > 0:
                e -= p * math.log(p)
        return e

    def to_dict(self) -> dict:
        return {
            "hypotheses": [
                {"diagnosis": h.diagnosis,
                 "probability": round(h.probability, 4),
                 "evidence":    h.evidence}
                for h in self.hypotheses
            ],
            "sigma2": round(self.sigma2, 6),
            "sc":     round(self.sc, 6),
            "entropy": round(self.entropy(), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-phase outputs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerceptionEvidence:
    """Structured output from MPE Phase 1 (Moondream2 / Qwen2-VL)."""
    rois: List[Dict]         = field(default_factory=list)   # regions of interest
    feature_summary: str     = ""
    confidence_level: str    = "medium"   # "low" | "medium" | "high"
    saliency_crop: str       = ""         # "x1,y1,x2,y2" bounding box string
    raw_summary: str         = ""         # full model output (for audit)


@dataclass
class ReasoningOutput:
    """Structured output from ARLL Phase 2 (DeepSeek-R1-Distill)."""
    cot: str                     = ""     # full chain-of-thought trace
    ensemble: DDxEnsemble        = field(default_factory=DDxEnsemble)
    wanna: bool                  = False  # True → #wanna# triggered
    feedback_request: str        = "none" # "High-Res Crop" | "Alternate View"
    feedback_payload: str        = ""     # "region=...;zoom=..." etc.
    rag_references: List[str]    = field(default_factory=list)
    temporal_note: str           = ""     # interval change vs prior scan
    raw_output: str              = ""     # full model output (for audit)


@dataclass
class FeedbackTensor:
    """Compact feedback returned to MPE by the #wanna# protocol."""
    request_type: str = "none"
    payload: str      = ""


@dataclass
class RiskScore:
    """Result from the RiskStratifier."""
    scale: str          = ""   # Lung-RADS | TIRADS | BI-RADS | LI-RADS | PI-RADS
    score: str          = ""   # e.g. "4X", "TR5", "6"
    interpretation: str = ""
    action: str         = ""   # recommended clinical action


@dataclass
class ClinicalEntity:
    """A named entity extracted from text (diagnosis, measurement, etc.)."""
    entity_type: str = ""   # diagnosis | measurement | risk_factor | finding
    text: str        = ""
    icd11: str       = ""
    snomed_ct: str   = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Uncertainty metrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintyMetrics:
    """All uncertainty quantities for a single iteration."""
    confidence: float         = 0.0   # Sc = 1 - σ²
    uncertainty: float        = 1.0   # 1 - Sc
    predictive_entropy: float = 0.0   # H(Sc) binary entropy
    ddx_variance: float       = 0.0   # σ²
    ddx_entropy: float        = 0.0   # Shannon entropy of DDx distribution


@dataclass
class CalibrationBin:
    """One bin of the ECE reliability diagram."""
    lower: float = 0.0
    upper: float = 0.1
    mean_conf: float  = 0.0
    mean_acc: float   = 0.0
    count: int        = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline trace & summary
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IterationTrace:
    """Full record of a single pipeline iteration (for audit & visualisation)."""
    iteration: int               = 1
    perception_summary: str      = ""
    reasoning_summary: str       = ""
    decision: str                = ""
    metrics: UncertaintyMetrics  = field(default_factory=UncertaintyMetrics)
    ddx_ensemble: Dict           = field(default_factory=dict)
    rag_references: List[str]    = field(default_factory=list)
    temporal_note: str           = ""
    doctor_feedback: str         = ""
    elapsed_s: float             = 0.0


@dataclass
class RunSummary:
    """
    Complete record of a diagnostic run.
    Includes the iteration trace, final report, calibration bins,
    and session metadata required for the audit trail.
    """
    session_id: str                        = field(default_factory=lambda: str(uuid.uuid4())[:8])
    success: bool                          = False
    escalated_to_human: bool               = False
    iterations_executed: int               = 0
    final_report_json: str                 = ""
    trace: List[IterationTrace]            = field(default_factory=list)
    total_elapsed_s: float                 = 0.0
    calibration_bins: List[Tuple[float, float, int]] = field(default_factory=list)
    image_path: str                        = ""
    prior_image_path: str                  = ""
    model_vision: str                      = ""
    model_reasoning: str                   = ""
    model_clinical: str                    = ""


@dataclass
class WannaDecision:
    """Decision emitted by the #wanna# state machine."""
    state: WannaState        = WannaState.ProceedToReport
    iteration: int           = 1
    feedback: FeedbackTensor = field(default_factory=FeedbackTensor)


# ═══════════════════════════════════════════════════════════════════════════════
#  Doctor-in-the-loop
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DoctorFeedback:
    """
    Input received from the clinician during HITL interaction.
    Parsed from free-text; zoom_region and is_zoom_command are extracted
    by HITLCoordinator.parse_zoom_command().
    """
    message: str          = ""
    zoom_region: str      = ""
    is_zoom_command: bool = False
    raw_input: str        = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Model configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelSettings:
    """
    Paths to all GGUF model files and inference hyper-parameters.
    T4-default: Moondream2 + DeepSeek-R1-Distill + MedGemma-2B.
    """
    vision_projection_model: str = "models/vision_proj.gguf"      # CLIP mmproj
    vision_text_model: str       = "models/vision_text.gguf"       # Moondream2
    reasoning_model: str         = "models/reasoning_expert.gguf"  # DeepSeek-R1-Distill
    clinical_model: str          = "models/clinical_expert.gguf"   # MedGemma-2B
    inference: InferenceParams   = field(default_factory=InferenceParams)
