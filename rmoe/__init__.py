"""
rmoe/__init__.py — Public API for the R-MoE v2.0 Python engine.

Importing this package exposes the primary classes and helpers needed by:
  • engine.py      (CLI entry-point)
  • colab_runner.py (Colab launcher)
  • any custom downstream scripts

Quick example:
    from rmoe import MrTom, WannaStateMachine, HITLMode, print_banner
    mr = MrTom(WannaStateMachine(hard_limit=3, threshold=0.90))
    mr.set_vision_model("models/vision_proj.gguf", "models/vision_text.gguf")
    mr.set_reasoning_model("models/reasoning_expert.gguf")
    mr.set_clinical_model("models/clinical_expert.gguf")
    summary = mr.process_patient_case("patient.png", audit_log_path="audit.json")
    mr.print_report()
    mr.print_charts()
    mr.run_qa_loop()
"""
from __future__ import annotations

from rmoe.models import (
    CalibrationBin, ClinicalEntity, DDxEnsemble, DDxHypothesis,
    DoctorFeedback, ExpertTarget, FeedbackTensor, HITLMode, InferenceParams,
    IterationTrace, ModelSettings, PerceptionEvidence, ReasoningOutput,
    RiskScore, RunSummary, UncertaintyMetrics, WannaDecision, WannaState,
)
from rmoe.core import (
    DiagnosticEngine, MrTom, MPEConfidenceGate, WannaStateMachine,
)
from rmoe.agents import ExpertSwapper, VisionExpert, ReasoningExpert, ReportingExpert
from rmoe.hitl import ExpertQueryRouter, HITLCoordinator
from rmoe.rag import VectorRAGEngine
from rmoe.ontology import (
    ClinicalEntityExtractor, ICD11, RiskStratifier, SNOMED_CT, lookup_icd11, lookup_snomed,
)
from rmoe.calibration import CalibrationTracker, compute_uncertainty
from rmoe.audit import AuditLogger, SessionReportGenerator
from rmoe.ui import print_banner, print_clinical_report, print_run_summary
from rmoe.charts import (
    benchmark_comparison, ddx_evolution_chart, reliability_diagram,
    sc_progression_chart, uncertainty_heatmap,
)
# ── New paper-aligned modules ─────────────────────────────────────────────────
from rmoe.bias import CognitiveBiasDetector, BiasType, BiasReport, print_bias_report
from rmoe.mcv import MCVBuilder, MCVInjector, MCVTensor
from rmoe.safety import (
    CSRSafetyValidator, SafetyStatus, SafetyReport, print_safety_report,
)
from rmoe.modality import (
    ModalityEscalationRouter, Modality, EscalationUrgency, ModalityEscalation,
)
from rmoe.temporal import (
    TemporalComparator, TemporalAnalysis, ChangeClass, mock_temporal_note,
)
from rmoe.saliency import SaliencyProcessor, CropCoordinates, AttentionMap
from rmoe.dicom import DICOMProcessor, WindowPreset, get_window, WINDOW_PRESETS
from rmoe.eval import BenchmarkRunner, BenchmarkDataset, BenchmarkMetrics, CaseResult
from rmoe.ensemble import MultiTemperatureEnsemble

__version__ = "2.0.0"
__author__  = "R-MoE Research Team"
__paper__   = "Recursive Multi-Agent Mixture-of-Experts (R-MoE) for Autonomous Clinical Diagnostics"

__all__ = [
    # Core engine
    "MrTom", "DiagnosticEngine", "WannaStateMachine", "MPEConfidenceGate",
    # Agents
    "ExpertSwapper", "VisionExpert", "ReasoningExpert", "ReportingExpert",
    # HITL
    "HITLCoordinator", "ExpertQueryRouter",
    # RAG
    "VectorRAGEngine",
    # Ontology
    "ICD11", "SNOMED_CT", "RiskStratifier", "ClinicalEntityExtractor",
    "lookup_icd11", "lookup_snomed",
    # Calibration
    "CalibrationTracker", "compute_uncertainty",
    # Audit
    "AuditLogger", "SessionReportGenerator",
    # Models / enums
    "WannaState", "WannaDecision", "HITLMode", "ExpertTarget",
    "DDxEnsemble", "DDxHypothesis", "PerceptionEvidence", "ReasoningOutput",
    "RunSummary", "IterationTrace", "UncertaintyMetrics", "ModelSettings",
    "InferenceParams", "DoctorFeedback", "RiskScore", "ClinicalEntity",
    "FeedbackTensor", "CalibrationBin",
    # UI helpers
    "print_banner", "print_clinical_report", "print_run_summary",
    # Charts
    "sc_progression_chart", "ddx_evolution_chart", "uncertainty_heatmap",
    "reliability_diagram", "benchmark_comparison",
    # Bias detection (paper §Error Patterns)
    "CognitiveBiasDetector", "BiasType", "BiasReport", "print_bias_report",
    # MCV inter-agent transfer (paper §MCV)
    "MCVBuilder", "MCVInjector", "MCVTensor",
    # CSR safety validator (paper §dual-layer)
    "CSRSafetyValidator", "SafetyStatus", "SafetyReport", "print_safety_report",
    # Modality escalation (paper §#wanna# mode 3)
    "ModalityEscalationRouter", "Modality", "EscalationUrgency", "ModalityEscalation",
    # Temporal analysis (paper §3.1 ARLL)
    "TemporalComparator", "TemporalAnalysis", "ChangeClass", "mock_temporal_note",
    # Saliency / crops (paper §MPE)
    "SaliencyProcessor", "CropCoordinates", "AttentionMap",
    # DICOM preprocessing
    "DICOMProcessor", "WindowPreset", "get_window", "WINDOW_PRESETS",
    # Benchmark / evaluation (paper §4)
    "BenchmarkRunner", "BenchmarkDataset", "BenchmarkMetrics", "CaseResult",
    # Multi-temperature ensemble (paper §3.1 Sc formula)
    "MultiTemperatureEnsemble",
    # Package metadata
    "__version__", "__author__", "__paper__",
]
