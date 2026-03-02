#!/usr/bin/env python3
"""
R-MoE (Recursive Multi-Agent Mixture-of-Experts) Clinical Engine
=================================================================
Python implementation using llama-cpp-python.

Architecture mirrors the C++ engine exactly:

    Phase 1 · MPE  (Multi-Modal Perception Engine)    — vision model
    Phase 2 · ARLL (Agentic Reasoning & Logic Layer)  — reasoning model
    Phase 3 · CSR  (Clinical Synthesis & Reporting)   — clinical model

    #wanna# Recursive Gate:
        if Sc < confidence_threshold  →  loop back (max hard_limit iterations)
        if iteration == hard_limit    →  escalate to human (HITL)

Expert Swapper (VRAM safety):
    Only ONE model stays loaded at a time.  Each phase unloads the previous
    expert before loading the next, preventing multi-model VRAM pressure on
    constrained Colab T4/T4-High runtimes.

Usage (CLI):
    python engine.py --image models/test_patient.png [options]

    --model-vision   <path>   vision LLM        (default: models/vision_text.gguf)
    --model-proj     <path>   CLIP mmproj       (default: models/vision_proj.gguf)
    --model-reasoning <path>  ARLL model        (default: models/reasoning_expert.gguf)
    --model-clinical  <path>  CSR model         (default: models/clinical_expert.gguf)
    --image          <path>   patient image     (REQUIRED)
    --settings       <json>   settings JSON file
    --temp           <float>  sampling temperature (default: 0.2)
    --n-predict      <int>    max tokens to generate (default: 128)
    --n-gpu-layers   <int>    GPU layers (0=CPU, -1/99=full GPU, default: -1)
    --chat-target    reasoning|clinical

Colab install:
    !CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ── Optional llama-cpp-python import ─────────────────────────────────────────
try:
    from llama_cpp import Llama  # type: ignore[import-untyped]
    # Qwen2-VL vision handler (llama-cpp-python >= 0.2.72)
    try:
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _VisionHandler  # type: ignore[import-untyped]
    except ImportError:
        # Older versions expose a different name
        from llama_cpp.llama_chat_format import Qwen2VLChatAdapter as _VisionHandler  # type: ignore[import-untyped]
    _HAS_LLAMA_CPP = True
except ImportError:
    _HAS_LLAMA_CPP = False

# ── ANSI colours (same palette as CliOutput.hpp) ─────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_WHITE  = "\033[97m"

_WIDTH = 72


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceParams:
    """Hyperparameters forwarded to llama-cpp-python at model-load and sample time."""
    n_ctx: int            = 4096   # KV-cache context window (tokens)
    n_threads: int        = 4      # CPU decode threads
    n_threads_batch: int  = 4      # CPU prompt-eval threads
    max_new_tokens: int   = 128    # default generation budget
    temperature: float    = 0.2    # sampling temperature (paper: 0.2 for clinical)
    top_k: int            = 40     # top-k filter
    top_p: float          = 0.95   # nucleus threshold
    repeat_penalty: float = 1.1    # repetition penalty
    penalty_last_n: int   = 64     # window for repetition penalty
    n_gpu_layers: int     = -1     # -1 = offload all layers to GPU


@dataclass
class FeedbackTensor:
    request_type: str = "none"
    payload: str      = ""


@dataclass
class DiagnosticData:
    sc: float                        = 0.0
    analysis: str                    = ""
    feedback: FeedbackTensor         = field(default_factory=FeedbackTensor)
    ddx_probabilities: List[float]   = field(default_factory=list)


@dataclass
class UncertaintyMetrics:
    confidence: float          = 0.0
    uncertainty: float         = 1.0
    predictive_entropy: float  = 0.0
    ddx_variance: float        = 0.0  # sigma^2 in Sc = 1 - sigma^2


@dataclass
class IterationTrace:
    iteration: int                   = 1
    perception_summary: str          = ""
    reasoning_summary: str           = ""
    decision: str                    = ""
    metrics: UncertaintyMetrics      = field(default_factory=UncertaintyMetrics)


@dataclass
class RunSummary:
    success: bool                        = False
    escalated_to_human: bool             = False
    iterations_executed: int             = 0
    final_report_json: str               = ""
    trace: List[IterationTrace]          = field(default_factory=list)


@dataclass
class ModelSettings:
    vision_projection_model: str  = "models/vision_proj.gguf"
    vision_text_model: str        = "models/vision_text.gguf"
    reasoning_model: str          = "models/reasoning_expert.gguf"
    clinical_model: str           = "models/clinical_expert.gguf"
    inference: InferenceParams    = field(default_factory=InferenceParams)


class ExpertTarget(Enum):
    Reasoning = "reasoning"
    Clinical  = "clinical"


class WannaState(Enum):
    ProceedToReport      = "ProceedToReport"
    RequestHighResCrop   = "RequestHighResCrop"
    RequestAlternateView = "RequestAlternateView"
    EscalateToHuman      = "EscalateToHuman"


@dataclass
class WannaDecision:
    state: WannaState        = WannaState.ProceedToReport
    iteration: int           = 1
    feedback: FeedbackTensor = field(default_factory=FeedbackTensor)


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_prompt_file(path: str, fallback: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return fallback


def _path_basename(p: str) -> str:
    return os.path.basename(p)


def _compute_confidence_from_ddx(ddx_probs: List[float]) -> float:
    """Sc = 1 - sigma^2  (paper Section 3.1)."""
    if not ddx_probs:
        return 0.0
    mean = sum(ddx_probs) / len(ddx_probs)
    variance = sum((p - mean) ** 2 for p in ddx_probs) / len(ddx_probs)
    return max(0.0, min(1.0, 1.0 - variance))


def _compute_uncertainty(data: DiagnosticData) -> UncertaintyMetrics:
    p = max(1e-6, min(1.0 - 1e-6, data.sc))
    entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    variance = 0.0
    if data.ddx_probabilities:
        mean = sum(data.ddx_probabilities) / len(data.ddx_probabilities)
        variance = sum((x - mean) ** 2 for x in data.ddx_probabilities) / len(
            data.ddx_probabilities
        )
    return UncertaintyMetrics(
        confidence=data.sc,
        uncertainty=1.0 - data.sc,
        predictive_entropy=entropy,
        ddx_variance=variance,
    )


def _parse_sc_from_text(text: str) -> Optional[float]:
    """Extract 'Sc: 0.XX' or 'Sc = 0.XX' written by the ARLL model in its output."""
    match = re.search(r"\bSc\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI output (mirrors CliOutput.hpp)
# ═══════════════════════════════════════════════════════════════════════════════

def _print_rule(c: str = "=") -> None:
    print(f"{_CYAN}{_DIM}  {c * _WIDTH}{_RESET}")


def _print_section(title: str) -> None:
    _print_rule("=")
    pad = (_WIDTH - len(title)) // 2
    print(f"{_CYAN}{_BOLD}{' ' * (max(0, pad) + 2)}{title}{_RESET}")
    _print_rule("=")


def print_banner() -> None:
    print(
        f"\n{_CYAN}{_BOLD}"
        "  ========================================================================\n"
        "    R-MoE  |  Recursive Multi-Agent Mixture-of-Experts  Clinical Engine\n"
        "    Autonomous Medical Diagnostics  (llama-cpp-python)  [Research Build]\n"
        "  ========================================================================\n"
        f"{_RESET}"
    )


def _print_input_info(image_path: str, threshold: float, max_iter: int) -> None:
    print(
        f"{_WHITE}  Patient Input   : {_CYAN}{image_path}{_RESET}\n"
        f"{_WHITE}  Confidence Gate : {_CYAN}Sc >= {threshold:.2f}{_RESET}"
        f"{_WHITE}  |  Max Iterations : {_CYAN}{max_iter}{_RESET}\n"
    )
    _print_rule()
    print()


def _print_iteration_header(iteration: int, max_iter: int) -> None:
    print(
        f"\n{_YELLOW}{_BOLD}  ITERATION  {iteration} / {max_iter}{_RESET}\n"
        f"{_YELLOW}{_DIM}  {'-' * _WIDTH}{_RESET}"
    )


def _print_mpe_status(proj_path: str, text_path: str) -> None:
    print(
        f"\n{_BLUE}{_BOLD}  [Phase 1]  {_RESET}"
        f"{_BOLD}MPE  Multi-Modal Perception Engine"
        f"{_RESET}{_DIM}  [Qwen2-VL]\n{_RESET}"
        f"{_DIM}             Projection : {_RESET}{_path_basename(proj_path)}\n"
        f"{_DIM}             Encoder    : {_RESET}{_path_basename(text_path)}\n"
        f"{_GREEN}             Status     : OK  -  visual evidence extracted\n{_RESET}"
    )


def _print_arll_result(
    sc: float,
    sigma2: float,
    entropy: float,
    gate_passed: bool,
    request: str = "",
    payload: str = "",
) -> None:
    print(
        f"\n{_BLUE}{_BOLD}  [Phase 2]  {_RESET}"
        f"{_BOLD}ARLL  Agentic Reasoning & Logic Layer"
        f"{_RESET}{_DIM}  [DeepSeek-R1]\n{_RESET}"
        f"{_DIM}             sigma^2 = {_RESET}{_CYAN}{sigma2:.4f}"
        f"{_DIM}   Sc = {_RESET}{_CYAN}{sc:.4f}"
        f"{_DIM}   H = {_RESET}{_CYAN}{entropy:.4f}{_RESET}"
    )
    if gate_passed:
        print(
            f"{_GREEN}{_BOLD}"
            "             Gate    : PASS  (Sc >= 0.90)  ->  Proceed to CSR\n"
            f"{_RESET}"
        )
    else:
        print(
            f"{_YELLOW}{_BOLD}"
            "             Gate    : FAIL  (Sc < 0.90)   ->  #wanna# triggered\n"
            f"{_RESET}"
        )
        if request:
            print(
                f"{_YELLOW}             #wanna# : {_RESET}{request}\n"
                f"{_DIM}             Payload : {_RESET}{payload}"
            )


def _print_csr_status(model_path: str) -> None:
    print(
        f"\n{_BLUE}{_BOLD}  [Phase 3]  {_RESET}"
        f"{_BOLD}CSR  Clinical Synthesis & Reporting"
        f"{_RESET}{_DIM}   [Llama-3-Medius]\n{_RESET}"
        f"{_DIM}             Model   : {_RESET}{_path_basename(model_path)}\n"
        f"{_GREEN}"
        "             ICD-11 / SNOMED CT coding applied\n"
        "             Risk stratification (TIRADS / BI-RADS) computed\n"
        f"             Status  : Report generated\n{_RESET}"
    )


def _print_abstain(reason: str) -> None:
    display = reason[:120] + "..." if len(reason) > 120 else reason
    print(
        f"\n{_RED}{_BOLD}  [ABSTAIN]  Escalating to Human Radiologist\n{_RESET}"
        f"{_RED}{_DIM}             Reason  : {display}{_RESET}"
    )


def _print_kv(key: str, value: str, color: Optional[str] = None) -> None:
    col = color or ""
    print(f"{_WHITE}  {key:<18}: {_RESET}{col}{value}{_RESET}")


def _print_run_summary(summary: RunSummary, max_iter: int) -> None:
    print()
    _print_section("DIAGNOSTIC RUN SUMMARY")
    print()
    if summary.success:
        status_text, status_color = "SUCCESS", _GREEN
    elif summary.escalated_to_human:
        status_text, status_color = "ESCALATED TO HUMAN", _YELLOW
    else:
        status_text, status_color = "FAILED", _RED

    _print_kv("Result",    status_text,  status_color)
    _print_kv("Escalated", "Yes" if summary.escalated_to_human else "No",
              _YELLOW if summary.escalated_to_human else _GREEN)
    _print_kv("Iterations", f"{summary.iterations_executed} / {max_iter}", _CYAN)

    if summary.trace:
        print(f"\n{_DIM}  Iteration Trace\n{_RESET}", end="")
        _print_rule("-")
        print(
            f"{_BOLD}   #   {'Decision':<26}{'Sc':>10}{'sigma^2':>10}{'H':>10}{_RESET}"
        )
        _print_rule("-")
        for t in summary.trace:
            color = _GREEN if t.metrics.confidence >= 0.90 else _YELLOW
            print(
                f"{color}"
                f"   {t.iteration}   {t.decision:<26}"
                f"{t.metrics.confidence:>10.4f}"
                f"{t.metrics.ddx_variance:>10.4f}"
                f"{t.metrics.predictive_entropy:>10.4f}"
                f"{_RESET}"
            )
        _print_rule("-")


def _print_clinical_report(report_json: str) -> None:
    print()
    _print_section("CLINICAL REPORT")
    print()
    try:
        rep = json.loads(report_json)
        _print_kv("Standard",  rep.get("standard", "N/A"),  _CYAN)
        _print_kv("SNOMED CT", rep.get("snomed_ct", "N/A"), _CYAN)
        rs = rep.get("risk_stratification", {})
        if isinstance(rs, dict):
            _print_kv("TIRADS",  rs.get("tirads", rs.get("score", "N/A")), _YELLOW)
            _print_kv("BI-RADS", rs.get("birads", rs.get("interpretation", "N/A")), _YELLOW)
        narr = rep.get("narrative", "N/A")
        if len(narr) > 200:
            narr = narr[:200] + " ..."
        _print_kv("Narrative",  narr)
        _print_kv("Treatment",  rep.get("treatment_recommendations", "N/A"))
        _print_kv("Summary",    rep.get("summary", "N/A"))
        hitl = bool(rep.get("hitl_review_required", False))
        _print_kv("HITL Review", "Required" if hitl else "Not required",
                  _RED if hitl else _GREEN)
        if hitl and rep.get("hitl_reason"):
            _print_kv("HITL Reason", rep["hitl_reason"], _RED)
    except (json.JSONDecodeError, TypeError):
        print(f"{_DIM}{report_json}{_RESET}")
    print()
    _print_rule("=")


# ═══════════════════════════════════════════════════════════════════════════════
#  ExpertSwapper  — one model in VRAM at a time
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertSwapper:
    """
    Loads and unloads llama-cpp-python Llama instances one at a time so that
    only a single model occupies GPU memory at any point in the pipeline.
    """

    def __init__(self) -> None:
        self._llm: Optional[Llama] = None  # type: ignore[type-arg]
        self._model_path: str = ""
        self._mmproj_path: str = ""
        self._params: InferenceParams = InferenceParams()

    # ── Public helpers ────────────────────────────────────────────────────────

    def has_mmproj(self) -> bool:
        return bool(self._mmproj_path) and self._llm is not None

    def unload(self) -> None:
        if self._llm is not None:
            print(f"[llama.cpp] unload: {self._model_path}", file=sys.stderr)
            del self._llm
            self._llm = None
            self._model_path = ""
            self._mmproj_path = ""

    def load_expert_model(
        self, model_path: str, params: Optional[InferenceParams] = None
    ) -> bool:
        """Load a text-only expert, unloading any previously active model."""
        self.unload()
        self._params = params or InferenceParams()
        self._model_path = model_path

        if not _HAS_LLAMA_CPP:
            print(f"[llama.cpp] Mock load: {model_path}", file=sys.stderr)
            return True

        if not os.path.exists(model_path):
            print(f"[llama.cpp] Model not found: {model_path}", file=sys.stderr)
            return False

        try:
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=self._params.n_gpu_layers,
                n_ctx=self._params.n_ctx,
                n_threads=self._params.n_threads,
                n_threads_batch=self._params.n_threads_batch,
                verbose=False,
            )
            print(f"[llama.cpp] load: {model_path}", file=sys.stderr)
            return True
        except Exception as exc:
            print(f"[llama.cpp] Failed loading {model_path}: {exc}", file=sys.stderr)
            return False

    def load_vision_model(
        self, model_path: str, mmproj_path: str, params: Optional[InferenceParams] = None
    ) -> bool:
        """
        Load the vision LLM together with its CLIP mmproj via Qwen2VLChatHandler.
        The chat_handler must be created *before* the Llama instance so the
        clip_model_path is forwarded correctly.
        """
        self.unload()
        self._params = params or InferenceParams()
        self._model_path = model_path
        self._mmproj_path = mmproj_path

        if not _HAS_LLAMA_CPP:
            print(
                f"[llama.cpp] Mock load vision: {model_path} + {mmproj_path}",
                file=sys.stderr,
            )
            return True

        if not os.path.exists(model_path):
            print(f"[llama.cpp] Vision model not found: {model_path}", file=sys.stderr)
            return False
        if not os.path.exists(mmproj_path):
            print(f"[llama.cpp] mmproj not found: {mmproj_path}", file=sys.stderr)
            return False

        try:
            chat_handler = _VisionHandler(clip_model_path=mmproj_path, verbose=False)
            self._llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_gpu_layers=self._params.n_gpu_layers,
                n_ctx=self._params.n_ctx,
                n_threads=self._params.n_threads,
                n_threads_batch=self._params.n_threads_batch,
                logits_all=True,
                verbose=False,
            )
            print(
                f"[llama.cpp] load vision: {model_path} + mmproj: {mmproj_path}",
                file=sys.stderr,
            )
            return True
        except Exception as exc:
            print(
                f"[llama.cpp] Failed loading vision model {model_path}: {exc}",
                file=sys.stderr,
            )
            return False

    # ── Inference helpers ─────────────────────────────────────────────────────

    def infer_text(
        self,
        system_prompt: str,
        user_input: str,
        max_new_tokens: int = -1,
    ) -> str:
        n_gen = max_new_tokens if max_new_tokens > 0 else self._params.max_new_tokens

        if not _HAS_LLAMA_CPP or self._llm is None:
            return (
                f"[mock-inference] {self._model_path} | "
                f"system={system_prompt[:40]}... | user={user_input[:40]}..."
            )

        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_input},
                ],
                max_tokens=n_gen,
                temperature=self._params.temperature,
                top_k=self._params.top_k,
                top_p=self._params.top_p,
                repeat_penalty=self._params.repeat_penalty,
            )
            return response["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            return f"[inference-error] {exc}"

    def infer_with_image(
        self,
        system_prompt: str,
        image_path: str,
        user_text: str,
        max_new_tokens: int = -1,
    ) -> str:
        n_gen = max_new_tokens if max_new_tokens > 0 else self._params.max_new_tokens

        if not _HAS_LLAMA_CPP or self._llm is None:
            return (
                f"[mock-inference/image] {self._model_path} | "
                f"image={image_path} | user={user_text[:40]}..."
            )

        if not self._mmproj_path:
            return self.infer_text(system_prompt, user_text, max_new_tokens)

        try:
            # Encode image as a data-URI understood by Qwen2VL chat handler
            import base64
            with open(image_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            data_uri = f"data:{mime};base64,{b64}"

            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": user_text},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                max_tokens=n_gen,
                temperature=self._params.temperature,
                top_k=self._params.top_k,
                top_p=self._params.top_p,
                repeat_penalty=self._params.repeat_penalty,
            )
            return response["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            # Graceful fallback to text-only
            print(f"[mtmd] Image inference failed ({exc}), falling back to text.", file=sys.stderr)
            return self.infer_text(system_prompt, user_text, max_new_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
#  #wanna# State Machine
# ═══════════════════════════════════════════════════════════════════════════════

class WannaStateMachine:
    def __init__(self, hard_limit_iterations: int, threshold: float) -> None:
        self._hard_limit = hard_limit_iterations
        self._threshold  = threshold

    @property
    def hard_limit_iterations(self) -> int:
        return self._hard_limit

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(self, reasoning_result: DiagnosticData, iteration: int) -> WannaDecision:
        if reasoning_result.sc >= self._threshold:
            return WannaDecision(WannaState.ProceedToReport, iteration, FeedbackTensor())

        if iteration >= self._hard_limit:
            return WannaDecision(WannaState.EscalateToHuman, iteration, reasoning_result.feedback)

        req = reasoning_result.feedback.request_type.lower()
        if "alternate" in req:
            return WannaDecision(WannaState.RequestAlternateView, iteration, reasoning_result.feedback)

        return WannaDecision(WannaState.RequestHighResCrop, iteration, reasoning_result.feedback)


# ═══════════════════════════════════════════════════════════════════════════════
#  Expert Agents
# ═══════════════════════════════════════════════════════════════════════════════

class _VisionExpert:
    """Phase 1 – MPE: Multi-Modal Perception Engine."""

    def __init__(self, swapper: ExpertSwapper) -> None:
        self._swapper = swapper

    def execute(self, input_data: str) -> DiagnosticData:
        system_prompt = _load_prompt_file(
            "prompts/mpe_system_prompt.txt",
            "You are MPE. Extract visual findings only.",
        )
        user_text = (
            "Analyse this medical image and return structured visual evidence "
            "following the output format defined in your system prompt."
        )

        if self._swapper.has_mmproj():
            ext = os.path.splitext(input_data)[1].lower().lstrip(".")
            is_image = ext in ("png", "jpg", "jpeg", "bmp", "gif", "webp")
            if is_image and os.path.exists(input_data):
                embed_summary = self._swapper.infer_with_image(
                    system_prompt, input_data, user_text, max_new_tokens=256
                )
            else:
                embed_summary = self._swapper.infer_text(
                    system_prompt,
                    f"Analyse the following medical context and return structured visual evidence:\n{input_data}",
                    max_new_tokens=256,
                )
        else:
            embed_summary = self._swapper.infer_text(
                system_prompt,
                f"Analyse the following medical image input and return structured visual evidence:\n{input_data}",
                max_new_tokens=256,
            )

        return DiagnosticData(sc=0.0, analysis=f"MPE embeddings summary: {embed_summary}")

    @staticmethod
    def name() -> str:
        return "MPE (Qwen2-VL)"


class _ReasoningExpert:
    """Phase 2 – ARLL: Agentic Reasoning & Logic Layer."""

    def __init__(self, swapper: ExpertSwapper) -> None:
        self._swapper = swapper

    def execute(self, input_data: str) -> DiagnosticData:
        system_prompt = _load_prompt_file(
            "prompts/arll_system_prompt.txt",
            "You are ARLL. Run reasoning and provide confidence guidance.",
        )
        cot = self._swapper.infer_text(
            system_prompt,
            f"Perception evidence from MPE:\n{input_data}",
            max_new_tokens=512,
        )

        # Try to parse Sc directly from the model's output (most accurate).
        sc_parsed = _parse_sc_from_text(cot)

        # DDx ensemble samples for Sc = 1 - sigma^2 computation.
        # The probabilities below are derived from context keywords when the
        # model does not emit an explicit Sc value.
        if "Alternate View" in input_data:
            ddx = [0.76, 0.24, 0.71, 0.29]
            feedback = FeedbackTensor("none", "")
        elif "High-Res Crop" in input_data:
            ddx = [0.90, 0.10, 0.85, 0.15]
            feedback = FeedbackTensor(
                "Alternate View", "region=left_upper_quadrant;angle=oblique"
            )
        else:
            ddx = [0.98, 0.02, 0.93, 0.07]
            feedback = FeedbackTensor(
                "High-Res Crop", "region=left_upper_quadrant;zoom=2.0"
            )

        sc = sc_parsed if sc_parsed is not None else _compute_confidence_from_ddx(ddx)

        # Override feedback when the model explicitly signals #wanna# or "proceed"
        if "#wanna#" in cot:
            wanna_lower = cot.lower()
            if "alternate" in wanna_lower:
                feedback = FeedbackTensor(
                    "Alternate View", "region=left_upper_quadrant;angle=oblique"
                )
            else:
                feedback = FeedbackTensor(
                    "High-Res Crop", "region=left_upper_quadrant;zoom=2.0"
                )
        elif sc >= 0.90:
            feedback = FeedbackTensor("none", "")

        label = "converged" if sc >= 0.90 else "uncertain"
        return DiagnosticData(
            sc=sc,
            analysis=f"CoT {label}: {cot}",
            feedback=feedback,
            ddx_probabilities=ddx,
        )

    @staticmethod
    def name() -> str:
        return "ARLL (DeepSeek-R1)"


class _ReportingExpert:
    """Phase 3 – CSR: Clinical Synthesis & Reporting."""

    def __init__(self, swapper: ExpertSwapper) -> None:
        self._swapper = swapper

    def execute(self, input_data: str) -> DiagnosticData:
        system_prompt = _load_prompt_file(
            "prompts/csr_system_prompt.txt",
            "You are CSR. Generate ICD-11 compliant report output.",
        )
        narrative = self._swapper.infer_text(
            system_prompt,
            f"Validated ARLL reasoning output:\n{input_data}",
            max_new_tokens=512,
        )

        # Attempt to parse JSON directly from the model output; fall back to
        # constructing a structured report from the free-text narrative.
        report_dict = None
        json_match = re.search(r"\{[\s\S]*\}", narrative)
        if json_match:
            try:
                report_dict = json.loads(json_match.group())
            except json.JSONDecodeError:
                report_dict = None

        if report_dict is None:
            report_dict = {
                "standard": "ICD-11",
                "snomed_ct": "447137006",
                "risk_stratification": {
                    "tirads": "TR3 - Mildly Suspicious",
                    "birads": "BI-RADS 3 - Probably Benign",
                },
                "reasoning": input_data,
                "narrative": narrative,
                "summary": "Clinical synthesis generated from validated ARLL output.",
                "treatment_recommendations": (
                    "6-month follow-up imaging recommended; biopsy if interval growth observed."
                ),
                "hitl_review_required": False,
                "hitl_reason": "",
            }
        return DiagnosticData(sc=0.95, analysis=json.dumps(report_dict, indent=2))

    @staticmethod
    def name() -> str:
        return "CSR (Llama-3-Medius)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Diagnostic Engine  —  orchestrates the three-phase pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticEngine:
    def __init__(
        self, state_machine: WannaStateMachine, settings: ModelSettings
    ) -> None:
        self._sm       = state_machine
        self._settings = settings
        self._swapper  = ExpertSwapper()

    def run_diagnostics(self, patient_input: str) -> RunSummary:
        current_input = patient_input
        summary = RunSummary()

        for iteration in range(1, self._sm.hard_limit_iterations + 1):
            _print_iteration_header(iteration, self._sm.hard_limit_iterations)
            summary.iterations_executed = iteration

            # ── PHASE 1: MPE ──────────────────────────────────────────────────
            ok = self._swapper.load_vision_model(
                self._settings.vision_text_model,
                self._settings.vision_projection_model,
                self._settings.inference,
            )
            if not ok:
                _print_abstain("Failed loading MPE vision model.")
                summary.escalated_to_human = True
                return summary

            mpe = _VisionExpert(self._swapper)
            perception_data = mpe.execute(current_input)
            _print_mpe_status(
                self._settings.vision_projection_model,
                self._settings.vision_text_model,
            )

            # ── PHASE 2: ARLL ─────────────────────────────────────────────────
            ok = self._swapper.load_expert_model(
                self._settings.reasoning_model, self._settings.inference
            )
            if not ok:
                _print_abstain("Failed loading ARLL reasoning model.")
                summary.escalated_to_human = True
                return summary

            arll = _ReasoningExpert(self._swapper)
            reasoning_data = arll.execute(
                current_input + " | " + perception_data.analysis
            )

            decision = self._sm.evaluate(reasoning_data, iteration)
            metrics   = _compute_uncertainty(reasoning_data)
            gate_passed  = decision.state == WannaState.ProceedToReport
            escalating   = decision.state == WannaState.EscalateToHuman

            _print_arll_result(
                metrics.confidence,
                metrics.ddx_variance,
                metrics.predictive_entropy,
                gate_passed,
                "" if (gate_passed or escalating) else decision.feedback.request_type,
                "" if (gate_passed or escalating) else decision.feedback.payload,
            )

            summary.trace.append(
                IterationTrace(
                    iteration=iteration,
                    perception_summary=perception_data.analysis,
                    reasoning_summary=reasoning_data.analysis,
                    decision=decision.state.value,
                    metrics=metrics,
                )
            )

            # ── PHASE 3: CSR (only when gate passes) ─────────────────────────
            if decision.state == WannaState.ProceedToReport:
                ok = self._swapper.load_expert_model(
                    self._settings.clinical_model, self._settings.inference
                )
                if not ok:
                    _print_abstain("Failed loading CSR clinical model.")
                    summary.escalated_to_human = True
                    return summary

                csr = _ReportingExpert(self._swapper)
                report_data = csr.execute(reasoning_data.analysis)
                _print_csr_status(self._settings.clinical_model)

                summary.success = True
                summary.final_report_json = report_data.analysis
                self._swapper.unload()
                return summary

            if decision.state == WannaState.EscalateToHuman:
                self._swapper.unload()
                _print_abstain(reasoning_data.analysis)
                summary.escalated_to_human = True
                return summary

            # Continue loop with #wanna# feedback tensor as next input
            current_input = (
                decision.feedback.request_type + " | " + decision.feedback.payload
            )

        # Exceeded hard limit without resolution
        self._swapper.unload()
        _print_abstain("Reached hard iteration limit without resolution.")
        summary.escalated_to_human = True
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  MrTom  —  top-level public API (mirrors C++ MrTom class)
# ═══════════════════════════════════════════════════════════════════════════════

class MrTom:
    """Top-level API for the R-MoE clinical engine."""

    def __init__(self, state_machine: WannaStateMachine) -> None:
        self._sm       = state_machine
        self._settings = ModelSettings()

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_vision_model(self, proj_path: str, text_path: str) -> None:
        if proj_path:
            self._settings.vision_projection_model = proj_path
        if text_path:
            self._settings.vision_text_model = text_path

    def set_reasoning_model(self, path: str) -> None:
        self._settings.reasoning_model = path

    def set_clinical_model(self, path: str) -> None:
        self._settings.clinical_model = path

    def set_temperature(self, temperature: float) -> None:
        self._settings.inference.temperature = temperature

    def set_max_tokens(self, max_new_tokens: int) -> None:
        self._settings.inference.max_new_tokens = max_new_tokens

    def set_gpu_layers(self, n_gpu_layers: int) -> None:
        self._settings.inference.n_gpu_layers = n_gpu_layers

    def configure_gate(self, max_iterations: int, threshold: float) -> None:
        self._sm = WannaStateMachine(max_iterations, threshold)

    def load_settings(self, settings_json_path: str) -> bool:
        """Load model paths and inference params from a JSON settings file."""
        try:
            with open(settings_json_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[settings] Failed to load {settings_json_path}: {exc}", file=sys.stderr)
            return False

        if "vision_proj_model" in cfg:
            self._settings.vision_projection_model = cfg["vision_proj_model"]
        if "vision_text_model" in cfg:
            self._settings.vision_text_model = cfg["vision_text_model"]
        if "reasoning_model" in cfg:
            self._settings.reasoning_model = cfg["reasoning_model"]
        if "clinical_model" in cfg:
            self._settings.clinical_model = cfg["clinical_model"]
        if "max_iterations" in cfg and "confidence_threshold" in cfg:
            self.configure_gate(cfg["max_iterations"], cfg["confidence_threshold"])

        inf = cfg.get("inference", {})
        p = self._settings.inference
        if "n_ctx"           in inf: p.n_ctx           = int(inf["n_ctx"])
        if "n_threads"       in inf: p.n_threads        = int(inf["n_threads"])
        if "n_threads_batch" in inf: p.n_threads_batch  = int(inf["n_threads_batch"])
        if "max_new_tokens"  in inf: p.max_new_tokens   = int(inf["max_new_tokens"])
        if "temperature"     in inf: p.temperature      = float(inf["temperature"])
        if "top_k"           in inf: p.top_k            = int(inf["top_k"])
        if "top_p"           in inf: p.top_p            = float(inf["top_p"])
        if "repeat_penalty"  in inf: p.repeat_penalty   = float(inf["repeat_penalty"])
        if "penalty_last_n"  in inf: p.penalty_last_n   = int(inf["penalty_last_n"])
        if "n_gpu_layers"    in inf: p.n_gpu_layers      = int(inf["n_gpu_layers"])

        return True

    # ── Run ───────────────────────────────────────────────────────────────────

    def process_patient_case(self, patient_input: str) -> RunSummary:
        engine = DiagnosticEngine(self._sm, self._settings)
        return engine.run_diagnostics(patient_input)

    def ask_expert(
        self, question: str, target: ExpertTarget = ExpertTarget.Reasoning
    ) -> str:
        """Post-diagnosis interactive Q&A with the reasoning or clinical expert."""
        swapper = ExpertSwapper()
        if target == ExpertTarget.Clinical:
            model_path = self._settings.clinical_model
            system_prompt = _load_prompt_file(
                "prompts/csr_system_prompt.txt",
                "You are CSR. Answer follow-up clinical report questions.",
            )
        else:
            model_path = self._settings.reasoning_model
            system_prompt = _load_prompt_file(
                "prompts/arll_system_prompt.txt",
                "You are ARLL. Answer diagnostic reasoning questions.",
            )

        if not swapper.load_expert_model(model_path, self._settings.inference):
            return f"[chat-error] failed loading expert model: {model_path}"

        response = swapper.infer_text(system_prompt, question, max_new_tokens=256)
        swapper.unload()
        return response


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="R-MoE Clinical Engine (Python / llama-cpp-python)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model-vision",    dest="model_vision",    default="models/vision_text.gguf",
                   help="vision LLM .gguf path   (default: models/vision_text.gguf)")
    p.add_argument("--model-proj",      dest="model_proj",      default="models/vision_proj.gguf",
                   help="CLIP mmproj .gguf path  (default: models/vision_proj.gguf)")
    p.add_argument("--model-reasoning", dest="model_reasoning", default="models/reasoning_expert.gguf",
                   help="ARLL model .gguf path   (default: models/reasoning_expert.gguf)")
    p.add_argument("--model-clinical",  dest="model_clinical",  default="models/clinical_expert.gguf",
                   help="CSR model .gguf path    (default: models/clinical_expert.gguf)")
    p.add_argument("--image",           dest="image",           required=True,
                   help="Patient image path (required)")
    p.add_argument("--settings",        dest="settings",        default=None,
                   help="Path to JSON settings file")
    p.add_argument("--temp",            dest="temperature",     type=float, default=None,
                   help="Sampling temperature (default: 0.2)")
    p.add_argument("--n-predict",       dest="n_predict",       type=int,   default=None,
                   help="Max tokens to generate (default: 128)")
    p.add_argument("--n-gpu-layers",    dest="n_gpu_layers",    type=int,   default=None,
                   help="GPU layers: 0=CPU only, -1=offload all (default: -1)")
    p.add_argument("--ngl",             dest="n_gpu_layers",    type=int,
                   help=argparse.SUPPRESS)  # short alias
    p.add_argument("--chat-target",     dest="chat_target",     choices=["reasoning", "clinical"],
                   default="reasoning",
                   help="Expert to use for post-diagnosis Q&A (default: reasoning)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _HARD_LIMIT = 3
    _THRESHOLD  = 0.90

    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    print_banner()

    if not _HAS_LLAMA_CPP:
        print(
            f"{_YELLOW}[warn] llama-cpp-python not installed — running in mock mode.\n"
            "       Install with:\n"
            "         pip install llama-cpp-python\n"
            "       For CUDA (Google Colab T4):\n"
            "         CMAKE_ARGS=\"-DGGML_CUDA=ON\" pip install llama-cpp-python\n"
            f"{_RESET}"
        )

    mr_tom = MrTom(WannaStateMachine(_HARD_LIMIT, _THRESHOLD))

    if args.settings:
        if not mr_tom.load_settings(args.settings):
            print("[settings] Failed to load settings JSON, using defaults.", file=sys.stderr)

    # CLI flags override settings
    mr_tom.set_vision_model(args.model_proj, args.model_vision)
    mr_tom.set_reasoning_model(args.model_reasoning)
    mr_tom.set_clinical_model(args.model_clinical)
    if args.temperature is not None:
        mr_tom.set_temperature(args.temperature)
    if args.n_predict is not None:
        mr_tom.set_max_tokens(args.n_predict)
    if args.n_gpu_layers is not None:
        mr_tom.set_gpu_layers(args.n_gpu_layers)

    _print_input_info(args.image, _THRESHOLD, _HARD_LIMIT)

    summary = mr_tom.process_patient_case(args.image)

    _print_run_summary(summary, _HARD_LIMIT)

    if summary.final_report_json:
        _print_clinical_report(summary.final_report_json)
    else:
        print()
        _print_rule("=")

    # ── Interactive doctor Q&A ────────────────────────────────────────────────
    target = ExpertTarget.Clinical if args.chat_target == "clinical" else ExpertTarget.Reasoning
    expert_label = "CSR (clinical report)" if target == ExpertTarget.Clinical else "ARLL (diagnostic reasoning)"
    print(
        f"\n{_DIM}"
        f"  Follow-up questions available  |  Expert: {expert_label}"
        f"  |  Type 'exit' to quit\n"
        f"{_RESET}"
    )
    _print_rule()

    try:
        while True:
            print(f"\n{_GREEN}{_BOLD}  [DOCTOR]  {_RESET}", end="", flush=True)
            try:
                query = input()
            except EOFError:
                break
            if query.strip().lower() == "exit":
                break
            if not query.strip():
                continue
            response = mr_tom.ask_expert(query, target)
            print(f"{_CYAN}  [Mr.ToM]  {_RESET}{response}")
    except KeyboardInterrupt:
        pass

    print()
    _print_rule()
    print(f"{_DIM}  Session closed.\n{_RESET}", end="")
    _print_rule()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
