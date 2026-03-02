"""
rmoe/core.py — DiagnosticEngine, WannaStateMachine, MPEConfidenceGate, MrTom.

This module is the beating heart of R-MoE v2.0:
  1. MrTom              — clean public API (load settings, set paths, run)
  2. DiagnosticEngine   — full pipeline orchestration (3-phase + HITL)
  3. WannaStateMachine  — confidence-gated recursion decisions
  4. MPEConfidenceGate  — Phase-1-level confidence pre-filter

Paper §3.2 — #wanna# Protocol:
  If Sc < θ (0.90): emit #wanna# + feedback request → re-scan (max 3 iterations)
  If iterations exhausted: EscalateToHuman
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

from rmoe.agents import (ExpertSwapper, ReasoningExpert, ReportingExpert,
                          VisionExpert)
from rmoe.audit import AuditLogger, SessionReportGenerator
from rmoe.calibration import CalibrationTracker, compute_uncertainty
from rmoe.hitl import ExpertQueryRouter, HITLCoordinator
from rmoe.models import (DoctorFeedback, ExpertTarget, FeedbackTensor,
                          HITLMode, InferenceParams, IterationTrace,
                          ModelSettings, PerceptionEvidence, ReasoningOutput,
                          RunSummary, UncertaintyMetrics, WannaDecision,
                          WannaState)
from rmoe.ontology import ClinicalEntityExtractor, RiskStratifier
from rmoe.rag import VectorRAGEngine
from rmoe.ui import (GREEN, BOLD, CYAN, DIM, MAGENTA, RED, RESET, YELLOW,
                      print_arll_gate, print_arll_header, print_csr_header,
                      print_mpe_evidence, print_mpe_header,
                      print_run_summary, print_clinical_report,
                      print_iteration_header, print_wanna_prompt,
                      print_abstain, _rule, print_input_info)


# ═══════════════════════════════════════════════════════════════════════════════
#  WannaStateMachine
# ═══════════════════════════════════════════════════════════════════════════════

class WannaStateMachine:
    """
    Implements the #wanna# protocol (paper §3.2).

    decide(sc, iteration, reasoning_output) → WannaDecision

    State transitions:
      sc ≥ θ                     →  ProceedToReport
      sc < θ, iter ≤ limit, crop →  RequestHighResCrop
      sc < θ, iter ≤ limit, alt  →  RequestAlternateView
      sc < θ, iter > limit        →  EscalateToHuman
    """

    def __init__(self, hard_limit: int = 3, threshold: float = 0.90) -> None:
        self.hard_limit = hard_limit
        self.threshold  = threshold

    def decide(
        self,
        sc: float,
        iteration: int,
        reasoning: Optional[ReasoningOutput] = None,
    ) -> WannaDecision:
        if sc >= self.threshold:
            return WannaDecision(
                state=WannaState.ProceedToReport,
                iteration=iteration,
                feedback=FeedbackTensor("none", ""),
            )

        if iteration >= self.hard_limit:
            return WannaDecision(
                state=WannaState.EscalateToHuman,
                iteration=iteration,
                feedback=FeedbackTensor("none", ""),
            )

        if reasoning:
            req = reasoning.feedback_request.lower()
        else:
            req = ""

        if "alternate" in req or "view" in req:
            state = WannaState.RequestAlternateView
        else:
            state = WannaState.RequestHighResCrop

        payload  = reasoning.feedback_payload if reasoning else ""
        fb_type  = reasoning.feedback_request if reasoning else "High-Res Crop"

        return WannaDecision(
            state=state,
            iteration=iteration,
            feedback=FeedbackTensor(fb_type, payload),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MPE Confidence Gate
# ═══════════════════════════════════════════════════════════════════════════════

class MPEConfidenceGate:
    """
    Phase-1 pre-filter: if MPE reports low confidence and no usable ROIs,
    trigger an early #wanna# before wasting ARLL tokens.

    Returns True  → proceed to ARLL
    Returns False → trigger early #wanna#
    """

    def __init__(self, low_threshold: str = "low") -> None:
        self._low = low_threshold  # "low" level from MPE confidence_level field

    def passes(self, evidence: PerceptionEvidence) -> bool:
        lvl = evidence.confidence_level.lower()
        if lvl == "low" and not evidence.rois:
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  DiagnosticEngine
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticEngine:
    """
    Full R-MoE v2.0 pipeline orchestrator.

    Implements the exact flow from the paper and v2.0 architecture diagram:

      INPUT → MPE → [MPE Gate] → ARLL → [ARLL Gate] → CSR
                         ↑                    |
                         └──── #wanna# ←──────┘  (max 3 iter)
                                    ↓
                               [HITL prompt]

    All intermediate states are captured in RunSummary for HITL audit.
    """

    def __init__(
        self,
        settings:     ModelSettings,
        state_machine: WannaStateMachine,
        hitl_mode:    HITLMode = HITLMode.Auto,
        prompt_dir:   str = "prompts",
    ) -> None:
        self._settings    = settings
        self._sm          = state_machine
        self._hitl        = HITLCoordinator(mode=hitl_mode)
        self._mpe_gate    = MPEConfidenceGate()
        self._rag         = VectorRAGEngine()
        self._calibration = CalibrationTracker()
        self._extractor   = ClinicalEntityExtractor()
        self._stratifier  = RiskStratifier()
        self._swapper     = ExpertSwapper()
        self._prompt_dir  = prompt_dir

    # ── Main entry-point ─────────────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        prior_image: Optional[str] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> RunSummary:
        summary = RunSummary(
            image_path=prior_image and image_path or image_path,
            prior_image_path=prior_image or "",
            model_vision=self._settings.vision_text_model,
            model_reasoning=self._settings.reasoning_model,
            model_clinical=self._settings.clinical_model,
        )

        print_input_info(
            image_path, self._sm.threshold, self._sm.hard_limit,
            prior_image=prior_image, hitl_mode=self._hitl._mode,
        )

        run_start = time.time()
        last_reasoning: Optional[ReasoningOutput] = None
        mpe_context    = image_path
        prior_arll_ctx = ""
        doctor_hint:   Optional[DoctorFeedback] = None

        for iteration in range(1, self._sm.hard_limit + 1):
            iter_start = time.time()
            print_iteration_header(iteration, self._sm.hard_limit)

            # ── Phase 1: MPE ─────────────────────────────────────────────────
            print_mpe_header(
                self._settings.vision_projection_model,
                self._settings.vision_text_model,
            )
            ok = self._swapper.load_vision_model(
                self._settings.vision_text_model,
                self._settings.vision_projection_model,
                self._settings.inference,
            )
            vision_expert = VisionExpert(self._swapper, iteration)
            perception    = vision_expert.execute(
                mpe_context,
                prior_image=prior_image,
                doctor_feedback=doctor_hint,
                prompt_dir=self._prompt_dir,
            )
            mpe_gate_ok = self._mpe_gate.passes(perception)
            print_mpe_evidence(perception, mpe_gate_ok)

            # MPE early #wanna# (gate fail without consuming ARLL tokens)
            if not mpe_gate_ok and iteration < self._sm.hard_limit:
                trace = IterationTrace(
                    iteration=iteration,
                    perception_summary=perception.feature_summary[:100],
                    decision="MPEGateFail",
                    metrics=UncertaintyMetrics(confidence=0.5),
                    elapsed_s=time.time() - iter_start,
                )
                summary.trace.append(trace)
                summary.iterations_executed = iteration
                mpe_context = f"High-Res Crop|{iteration}|zoom=2.0"
                continue

            # ── Phase 2: ARLL ────────────────────────────────────────────────
            rag_refs = self._rag.get_references(
                perception.feature_summary, top_k=3
            )
            print_arll_header(os.path.basename(self._settings.reasoning_model))

            self._swapper.load_expert_model(
                self._settings.reasoning_model, self._settings.inference
            )
            reasoning_expert = ReasoningExpert(self._swapper, iteration)
            reasoning = reasoning_expert.execute(
                mpe_evidence=json.dumps({
                    "rois": perception.rois,
                    "feature_summary": perception.feature_summary,
                    "confidence_level": perception.confidence_level,
                }, ensure_ascii=False),
                prior_context=prior_arll_ctx,
                rag_refs=rag_refs,
                prompt_dir=self._prompt_dir,
            )

            ens    = reasoning.ensemble
            sc     = ens.sc
            sigma2 = ens.sigma2
            entropy = ens.entropy()

            from rmoe.ui import print_ddx_ensemble
            print_ddx_ensemble(ens)
            print_arll_gate(
                sc, sigma2, entropy,
                gate_passed=(sc >= self._sm.threshold),
                request=reasoning.feedback_request,
                payload=reasoning.feedback_payload,
                rag_refs=rag_refs,
            )

            # Update calibration tracker
            top_p  = ens.primary.probability if ens.primary else 0.5
            self._calibration.update(sc, sc >= self._sm.threshold)

            # Build uncertainty metrics
            metrics = compute_uncertainty(sc, ens.probabilities)

            trace = IterationTrace(
                iteration=iteration,
                perception_summary=perception.feature_summary[:120],
                reasoning_summary=reasoning.cot[:200],
                decision="",
                metrics=metrics,
                ddx_ensemble=ens.to_dict(),
                rag_references=rag_refs,
                temporal_note=reasoning.temporal_note,
                doctor_feedback=doctor_hint.message if doctor_hint else "",
                elapsed_s=time.time() - iter_start,
            )
            summary.trace.append(trace)
            summary.iterations_executed = iteration
            last_reasoning = reasoning

            # ── Wanna State Machine ──────────────────────────────────────────
            decision = self._sm.decide(sc, iteration, reasoning)
            trace.decision = decision.state.value

            if decision.state == WannaState.ProceedToReport:
                print(f"\n  {GREEN}{BOLD}✓ ARLL Gate passed — proceeding to CSR{RESET}")
                summary.success = True
                break

            if decision.state == WannaState.EscalateToHuman:
                reason = f"Sc={sc:.4f} < {self._sm.threshold:.2f} after {iteration} iterations."
                print_abstain(reason)
                summary.escalated_to_human = True
                if audit_logger:
                    audit_logger.log("escalation", {"reason": reason, "sc": sc})
                break

            # #wanna# — prompt HITL before re-scan
            print()
            req, payload = decision.feedback.request_type, decision.feedback.payload
            print_wanna_prompt(req, payload, iteration)
            if audit_logger:
                audit_logger.log("wanna_triggered", {
                    "iteration": iteration, "sc": sc,
                    "request": req, "payload": payload,
                })

            doctor_hint = self._hitl.prompt_wanna(req, payload, iteration)
            if doctor_hint and doctor_hint.is_zoom_command:
                mpe_context = (
                    f"High-Res Crop|{doctor_hint.zoom_region}|zoom=2.5"
                )
            else:
                mpe_context = f"{req}|{payload}"

            prior_arll_ctx = f"Iter {iteration} DDx: {json.dumps(ens.to_dict())}"

        # ── Phase 3: CSR ─────────────────────────────────────────────────────
        if last_reasoning:
            print_csr_header(self._settings.clinical_model)
            self._swapper.load_expert_model(
                self._settings.clinical_model, self._settings.inference
            )
            reporting = ReportingExpert(self._swapper)
            report_json = reporting.execute(
                last_reasoning,
                iterations_used=summary.iterations_executed,
                prompt_dir=self._prompt_dir,
            )
            summary.final_report_json = report_json
            self._swapper.unload()

        summary.total_elapsed_s = time.time() - run_start

        # Populate calibration bins for audit
        summary.calibration_bins = [
            (b.mean_conf, b.mean_acc, b.count)
            for b in self._calibration.reliability_bins()
        ]

        if audit_logger:
            audit_logger.flush(summary)

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  MrTom — clean public API
# ═══════════════════════════════════════════════════════════════════════════════

class MrTom:
    """
    Public-facing API for the R-MoE engine.

    Usage (programmatic):
        mr_tom = MrTom(WannaStateMachine())
        mr_tom.load_settings("settings/rmoe_settings.json")
        mr_tom.set_vision_model("models/vision_proj.gguf", "models/vision_text.gguf")
        mr_tom.set_reasoning_model("models/reasoning_expert.gguf")
        mr_tom.set_clinical_model("models/clinical_expert.gguf")
        summary = mr_tom.process_patient_case("image.png", audit_log_path="audit.json")

    Usage (Colab / CLI):
        python engine.py --image patient.png --audit-log audit.json
    """

    def __init__(
        self,
        state_machine: Optional[WannaStateMachine] = None,
        hitl_mode: HITLMode = HITLMode.Auto,
        prompt_dir: str = "prompts",
    ) -> None:
        self._sm        = state_machine or WannaStateMachine()
        self._settings  = ModelSettings()
        self._hitl_mode = hitl_mode
        self._prompt_dir = prompt_dir
        self._last_summary:     Optional[RunSummary]     = None
        self._last_reasoning:   Optional[ReasoningOutput] = None
        self._swapper:          Optional[ExpertSwapper]   = None

    # ── Configuration ─────────────────────────────────────────────────────────

    def load_settings(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            inf = cfg.get("inference", {})
            self._settings.inference = InferenceParams(
                n_ctx=inf.get("n_ctx",            self._settings.inference.n_ctx),
                n_threads=inf.get("n_threads",     self._settings.inference.n_threads),
                n_threads_batch=inf.get("n_threads_batch",
                                        self._settings.inference.n_threads_batch),
                max_new_tokens=inf.get("max_new_tokens",
                                       self._settings.inference.max_new_tokens),
                temperature=inf.get("temperature", self._settings.inference.temperature),
                top_k=inf.get("top_k",             self._settings.inference.top_k),
                top_p=inf.get("top_p",             self._settings.inference.top_p),
                repeat_penalty=inf.get("repeat_penalty",
                                       self._settings.inference.repeat_penalty),
                penalty_last_n=inf.get("penalty_last_n",
                                       self._settings.inference.penalty_last_n),
                n_gpu_layers=inf.get("n_gpu_layers",
                                     self._settings.inference.n_gpu_layers),
            )
            if "confidence_threshold" in cfg:
                self._sm.threshold = float(cfg["confidence_threshold"])
            if "max_iterations" in cfg:
                self._sm.hard_limit = int(cfg["max_iterations"])
            for key in ("vision_proj_model", "vision_projection_model"):
                if key in cfg:
                    self._settings.vision_projection_model = cfg[key]
            for key in ("vision_text_model",):
                if key in cfg:
                    self._settings.vision_text_model = cfg[key]
            if "reasoning_model" in cfg:
                self._settings.reasoning_model = cfg["reasoning_model"]
            if "clinical_model" in cfg:
                self._settings.clinical_model = cfg["clinical_model"]
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"  [config] Could not load {path}: {exc}")

    def set_vision_model(self, proj: str, text: str) -> None:
        self._settings.vision_projection_model = proj
        self._settings.vision_text_model       = text

    def set_reasoning_model(self, path: str) -> None:
        self._settings.reasoning_model = path

    def set_clinical_model(self, path: str) -> None:
        self._settings.clinical_model = path

    def set_temperature(self, t: float) -> None:
        self._settings.inference.temperature = t

    def set_max_tokens(self, n: int) -> None:
        self._settings.inference.max_new_tokens = n

    def set_gpu_layers(self, n: int) -> None:
        self._settings.inference.n_gpu_layers = n

    def set_hitl_mode(self, mode: HITLMode) -> None:
        self._hitl_mode = mode

    def set_prompt_dir(self, d: str) -> None:
        self._prompt_dir = d

    # ── Execution ─────────────────────────────────────────────────────────────

    def process_patient_case(
        self,
        image_path: str,
        prior_image: Optional[str] = None,
        audit_log_path: Optional[str] = None,
    ) -> RunSummary:
        logger = AuditLogger(audit_log_path) if audit_log_path else None
        engine = DiagnosticEngine(
            settings=self._settings,
            state_machine=self._sm,
            hitl_mode=self._hitl_mode,
            prompt_dir=self._prompt_dir,
        )
        summary = engine.run(image_path, prior_image=prior_image, audit_logger=logger)
        self._last_summary   = summary
        self._swapper        = engine._swapper
        return summary

    def ask_expert(self, question: str, target: Optional[ExpertTarget] = None) -> str:
        """
        Post-diagnosis Q&A.  Routes to ARLL (Reasoning) or CSR (Clinical) expert.
        ExpertQueryRouter auto-selects if target is None.
        """
        if not self._last_summary:
            return "No diagnostic session available. Run process_patient_case() first."

        if target is None:
            target = ExpertQueryRouter.route(question)

        if self._swapper is None:
            self._swapper = ExpertSwapper()

        context = ""
        if self._last_summary.trace:
            last = self._last_summary.trace[-1]
            context = json.dumps(last.ddx_ensemble, indent=2)

        if target == ExpertTarget.Clinical:
            model_path = self._settings.clinical_model
            system     = (
                "You are a senior radiologist. Answer the doctor's clinical "
                "question concisely, referencing the established findings."
            )
        else:
            model_path = self._settings.reasoning_model
            system     = (
                "You are ARLL, a clinical reasoning agent. Answer the doctor's "
                "question about diagnostic reasoning with evidence-based precision."
            )

        self._swapper.load_expert_model(model_path, self._settings.inference)
        user_input = (
            f"Context (last DDx):\n{context}\n\n"
            f"Doctor's question: {question}"
        )
        resp = self._swapper.infer_text(system, user_input, max_new_tokens=512)
        self._swapper.unload()
        return resp

    def run_qa_loop(self, default_target: Optional[ExpertTarget] = None) -> None:
        """Launch the interactive HITL Q&A loop."""
        hitl = HITLCoordinator(mode=HITLMode.Interactive)
        hitl.run_qa_loop(self.ask_expert, default_target=default_target)

    def print_summary(self) -> None:
        if self._last_summary:
            print_run_summary(self._last_summary, self._sm.hard_limit)

    def print_report(self) -> None:
        if self._last_summary and self._last_summary.final_report_json:
            print_clinical_report(self._last_summary.final_report_json)

    def print_charts(self) -> None:
        if not self._last_summary:
            return
        from rmoe.charts import (sc_progression_chart, ddx_evolution_chart,
                                   uncertainty_heatmap, benchmark_comparison,
                                   reliability_diagram, _paper_calibration_bins)
        sc_progression_chart(self._last_summary.trace, self._sm.threshold)
        ddx_evolution_chart(self._last_summary.trace)
        uncertainty_heatmap(self._last_summary.trace)
        reliability_diagram(_paper_calibration_bins(), 0.08)
        benchmark_comparison()

    def generate_session_report(self) -> str:
        if not self._last_summary:
            return "No session available."
        return SessionReportGenerator().generate(self._last_summary)
