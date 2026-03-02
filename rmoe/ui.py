"""
rmoe/ui.py — Terminal UI: ANSI colours, box-drawing helpers, phase headers,
             DDx probability bars, gate indicators, run-summary table,
             and the clinical report formatter.

All print functions are pure side-effects (print to stdout).
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from rmoe.models import (DDxEnsemble, HITLMode, IterationTrace,
                          PerceptionEvidence, RunSummary, UncertaintyMetrics,
                          WannaState)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANSI palette  (gracefully disabled when not in a TTY)
# ═══════════════════════════════════════════════════════════════════════════════

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR", "")

def _c(code: str) -> str:
    return "" if _NO_COLOR else code

RESET   = _c("\033[0m");  BOLD    = _c("\033[1m");  DIM     = _c("\033[2m")
CYAN    = _c("\033[36m"); GREEN   = _c("\033[32m"); YELLOW  = _c("\033[33m")
RED     = _c("\033[31m"); BLUE    = _c("\033[34m"); WHITE   = _c("\033[97m")
MAGENTA = _c("\033[35m"); ITALIC  = _c("\033[3m")

# Box-drawing character sets
_H = "═"; _V = "║"; _TL = "╔"; _TR = "╗"; _BL = "╚"; _BR = "╝"
_ML = "╠"; _MR = "╣"; _TM = "╦"; _BM = "╩"; _CR = "╬"
_h = "─"; _v = "│"; _tl = "┌"; _tr = "┐"; _bl = "└"; _br = "┘"
_ml = "├"; _mr = "┤"
WIDTH = 72


# ═══════════════════════════════════════════════════════════════════════════════
#  Primitive helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rule(char: str = _H, color: str = "") -> None:
    print(f"{color}{DIM}  {char * WIDTH}{RESET}")


def _box_top(color: str = CYAN) -> None:
    print(f"{color}  {_TL}{_H * WIDTH}{_TR}{RESET}")


def _box_mid(color: str = CYAN) -> None:
    print(f"{color}  {_ML}{_H * WIDTH}{_MR}{RESET}")


def _box_bot(color: str = CYAN) -> None:
    print(f"{color}  {_BL}{_H * WIDTH}{_BR}{RESET}")


def _box_row(text: str, color: str = CYAN, inner: str = "") -> None:
    inner = inner or WHITE
    padded = text.ljust(WIDTH - 2)
    print(f"{color}  {_V} {inner}{padded}{color} {_V}{RESET}")


def _section(title: str, color: str = CYAN) -> None:
    _box_top(color)
    pad = (WIDTH - len(title) - 2) // 2
    label = " " * pad + title + " " * (WIDTH - len(title) - 2 - pad)
    print(f"{color}  {_V} {BOLD}{WHITE}{label}{color} {_V}{RESET}")
    _box_bot(color)


def _kv(key: str, value: str, kw: int = 22,
         kc: str = "", vc: str = "") -> None:
    kc = kc or DIM
    vc = vc or WHITE
    print(f"  {kc}{key:<{kw}}{RESET}: {vc}{value}{RESET}")


def _pb(val: float, width: int = 20, full: str = "█", empty: str = "░") -> str:
    """Horizontal ASCII progress bar."""
    filled = max(0, min(width, round(val * width)))
    return full * filled + empty * (width - filled)


# ═══════════════════════════════════════════════════════════════════════════════
#  Banner
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    print()
    _box_top(CYAN)
    lines = [
        f"  R-MoE v2.0  ·  Recursive Multi-Agent Mixture-of-Experts",
        f"  \"Hybrid Autonomous-Human Medical Reasoning\"",
        f"  Powered by llama-cpp-python  ·  No C++ recompilation required",
        f"",
        f"  Paper: 'RMoE for Autonomous Clinical Diagnostics'",
        f"  Benchmarks (MIMIC-CXR):  F1=0.92  ECE=0.08  TypeI=5.2%",
        f"  Models: Moondream2  ·  DeepSeek-R1-Distill  ·  MedGemma-2B",
    ]
    for ln in lines:
        padded = ln.ljust(WIDTH - 2)
        print(f"{CYAN}  {_V} {BOLD if 'R-MoE' in ln else ''}{WHITE}{padded}{CYAN} {_V}{RESET}")
    _box_bot(CYAN)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Input info
# ═══════════════════════════════════════════════════════════════════════════════

def print_input_info(
    image_path: str,
    threshold: float,
    max_iter: int,
    prior_image: Optional[str] = None,
    hitl_mode: HITLMode = HITLMode.Auto,
) -> None:
    _section("DIAGNOSTIC SESSION", BLUE)
    print()
    _kv("Patient image",   image_path,          vc=CYAN)
    if prior_image:
        _kv("Prior scan",  prior_image,          vc=CYAN)
    _kv("Confidence gate", f"Sc ≥ {threshold:.2f}", vc=GREEN)
    _kv("Max iterations",  str(max_iter),        vc=CYAN)
    _kv("HITL mode",       hitl_mode.value,      vc=YELLOW)
    print()
    _rule()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Iteration header
# ═══════════════════════════════════════════════════════════════════════════════

def print_iteration_header(iteration: int, max_iter: int) -> None:
    dots = "●" * iteration + "○" * (max_iter - iteration)
    print(
        f"\n{YELLOW}{BOLD}  ┌─ ITERATION {iteration}/{max_iter}  {dots} "
        f"{'─' * (WIDTH - 22 - max_iter * 2)}{RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — MPE
# ═══════════════════════════════════════════════════════════════════════════════

def print_mpe_header(proj: str, text: str) -> None:
    print(
        f"\n{BLUE}{BOLD}  ╔═ [Phase 1] MPE — Multi-Modal Perception Engine {DIM}[Moondream2]{RESET}\n"
        f"{BLUE}  ║{RESET}{DIM}  proj  : {os.path.basename(proj)}\n"
        f"{BLUE}  ║{RESET}{DIM}  model : {os.path.basename(text)}{RESET}"
    )


def print_mpe_evidence(evidence: Optional[PerceptionEvidence],
                        gate_passed: bool) -> None:
    if evidence:
        lvl = evidence.confidence_level
        lc  = GREEN if lvl == "high" else (YELLOW if lvl == "medium" else RED)
        print(f"{BLUE}  ║{RESET}  Perception confidence : {BOLD}{lc}{lvl.upper()}{RESET}")
        for roi in evidence.rois[:3]:
            sc = roi.get("suspicion", "")
            rc = RED if sc == "high" else (YELLOW if sc == "medium" else DIM)
            print(
                f"{BLUE}  ║{RESET}  {rc}▶ {roi.get('label','ROI')}: "
                f"{roi.get('descriptor','')}{RESET}"
            )
        if evidence.saliency_crop:
            print(f"{BLUE}  ║{RESET}{DIM}  Saliency crop : {evidence.saliency_crop}{RESET}")
    if gate_passed:
        print(f"{BLUE}  ║{RESET}{GREEN}  MPE Gate ✓  evidence forwarded to ARLL{RESET}")
    else:
        print(
            f"{BLUE}  ║{RESET}{YELLOW}  MPE Gate ✗  low perception confidence "
            f"→ #wanna# early trigger{RESET}"
        )
    print(f"{BLUE}  ╚{'═' * (WIDTH - 3)}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — ARLL
# ═══════════════════════════════════════════════════════════════════════════════

def print_arll_header(model_name: str) -> None:
    print(
        f"\n{MAGENTA}{BOLD}  ╔═ [Phase 2] ARLL — Agentic Reasoning & Logic Layer "
        f"{DIM}[{model_name}]{RESET}\n"
    )


def print_ddx_ensemble(ensemble: DDxEnsemble) -> None:
    if not ensemble.hypotheses:
        return
    print(f"{MAGENTA}  ║{RESET}  DDx Ensemble ({len(ensemble.hypotheses)} hypotheses):")
    for h in sorted(ensemble.hypotheses, key=lambda x: x.probability, reverse=True):
        bar = _pb(h.probability)
        c   = GREEN if h.probability >= 0.5 else (YELLOW if h.probability >= 0.2 else DIM)
        print(
            f"{MAGENTA}  ║{RESET}  {c}{bar}{RESET} "
            f"{h.probability:.3f}  {h.diagnosis}"
        )


def print_arll_gate(
    sc: float, sigma2: float, entropy: float,
    gate_passed: bool,
    request: str = "", payload: str = "",
    rag_refs: Optional[List[str]] = None,
) -> None:
    sc_c  = GREEN if sc >= 0.90 else (YELLOW if sc >= 0.70 else RED)
    print()
    print(
        f"{MAGENTA}  ║{RESET}  σ² = {CYAN}{sigma2:.4f}{RESET}"
        f"   Sc = {BOLD}{sc_c}{sc:.4f}{RESET}"
        f"   H = {CYAN}{entropy:.4f}{RESET}"
        f"   {_pb(sc, width=16)} "
    )
    if gate_passed:
        print(f"{MAGENTA}  ║{RESET}{GREEN}{BOLD}  Gate ✓  Sc ≥ 0.90  →  Proceed to CSR{RESET}")
    else:
        print(f"{MAGENTA}  ║{RESET}{YELLOW}{BOLD}  Gate ✗  Sc < 0.90  →  #wanna# triggered{RESET}")
        if request and request != "none":
            print(
                f"{MAGENTA}  ║{RESET}{YELLOW}  #wanna# : {BOLD}{request}{RESET}\n"
                f"{MAGENTA}  ║{RESET}{DIM}  Payload  : {payload}{RESET}"
            )
    if rag_refs:
        for ref in rag_refs[:2]:
            print(f"{MAGENTA}  ║{RESET}{DIM}  RAG ▶ {ref}{RESET}")
    print(f"{MAGENTA}  ╚{'═' * (WIDTH - 3)}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — CSR
# ═══════════════════════════════════════════════════════════════════════════════

def print_csr_header(model_path: str) -> None:
    print(
        f"\n{GREEN}{BOLD}  ╔═ [Phase 3] CSR — Clinical Synthesis & Reporting "
        f"{DIM}[{os.path.basename(model_path)}]{RESET}\n"
        f"{GREEN}  ║{RESET}{DIM}  ICD-11 / SNOMED CT coding applied\n"
        f"{GREEN}  ║{RESET}{DIM}  Risk stratification: Lung-RADS / TIRADS / BI-RADS\n"
        f"{GREEN}  ╚{'═' * (WIDTH - 3)}{RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HITL / abstain
# ═══════════════════════════════════════════════════════════════════════════════

def print_wanna_prompt(request: str, payload: str, iteration: int) -> None:
    print(
        f"\n{YELLOW}{BOLD}"
        f"  ╔═ R-MoE → DOCTOR  (Iteration {iteration} · Low confidence) ══════╗\n"
        f"  ║  Requesting: {request:<50}║\n"
        f"  ║  Target    : {payload:<50}║\n"
        f"  ║  Your hint helps me focus.  Examples:                          ║\n"
        f"  ║    'Show me the fracture site'                                 ║\n"
        f"  ║    'Focus on T4–T6 vertebrae'                                  ║\n"
        f"  ║    (press Enter to let me auto-refocus)                        ║\n"
        f"  ╚══════════════════════════════════════════════════════════════════╝{RESET}"
    )


def print_abstain(reason: str) -> None:
    display = (reason[:110] + "…") if len(reason) > 110 else reason
    print(
        f"\n{RED}{BOLD}  ╔═ ABSTAIN — Escalating to Human Radiologist ══════╗\n"
        f"  ║  {display:<68}║\n"
        f"  ╚══════════════════════════════════════════════════════╝{RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Run summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_run_summary(summary: RunSummary, max_iter: int) -> None:
    print()
    _section("DIAGNOSTIC RUN SUMMARY", BLUE)
    print()

    if summary.success:
        status, sc = "SUCCESS", GREEN
    elif summary.escalated_to_human:
        status, sc = "ESCALATED TO HUMAN", YELLOW
    else:
        status, sc = "FAILED", RED

    _kv("Result",     status,                sc)
    _kv("Escalated",  "Yes" if summary.escalated_to_human else "No",
                      YELLOW if summary.escalated_to_human else GREEN)
    _kv("Iterations", f"{summary.iterations_executed} / {max_iter}", CYAN)
    _kv("Elapsed",    f"{summary.total_elapsed_s:.1f} s", DIM)
    _kv("Session ID", summary.session_id, DIM)

    if summary.trace:
        print()
        _rule(_h)
        header = (f"  {'#':>3}  {'Decision':<26} {'Sc':>8} {'σ²':>8}"
                  f" {'H':>7} {'t(s)':>6}  {'Doctor hint':<16}")
        print(f"{BOLD}{header}{RESET}")
        _rule(_h)
        for t in summary.trace:
            c  = GREEN if t.metrics.confidence >= 0.90 else YELLOW
            hint = ("✓ " + t.doctor_feedback[:12]) if t.doctor_feedback else "—"
            print(
                f"{c}  {t.iteration:>3}  {t.decision:<26}"
                f" {t.metrics.confidence:>8.4f}"
                f" {t.metrics.ddx_variance:>8.4f}"
                f" {t.metrics.predictive_entropy:>7.4f}"
                f" {t.elapsed_s:>6.1f}"
                f"  {hint:<16}{RESET}"
            )
        _rule(_h)


# ═══════════════════════════════════════════════════════════════════════════════
#  Clinical report
# ═══════════════════════════════════════════════════════════════════════════════

def print_clinical_report(report_json: str) -> None:
    print()
    _section("CLINICAL REPORT", GREEN)
    print()
    try:
        rep = json.loads(report_json)

        _kv("ICD-11",    rep.get("standard",  "N/A"), vc=CYAN)
        _kv("SNOMED CT", rep.get("snomed_ct", "N/A"), vc=CYAN)

        rs = rep.get("risk_stratification", {})
        if isinstance(rs, dict):
            scale  = rs.get("scale",  "")
            score  = rs.get("score",  "N/A")
            interp = rs.get("interpretation", "")
            action = rs.get("action", "")
            _kv(scale or "Risk Scale", f"{score}  —  {interp}", vc=YELLOW)
            if action:
                _kv("Action", action, vc=YELLOW)

        print()
        narr = rep.get("narrative", "")
        if narr:
            print(f"  {BOLD}Narrative:{RESET}")
            for para in narr.split("\n\n"):
                if para.strip():
                    lines = para.strip().split("\n")
                    for ln in lines:
                        print(f"  {DIM}│{RESET}  {ln}")
                    print()

        rec = rep.get("treatment_recommendations", "")
        if rec:
            print(f"  {BOLD}Recommendations:{RESET}")
            for line in rec.strip().split("\n"):
                print(f"  {GREEN}▶{RESET}  {line.strip()}")
            print()

        _kv("Summary", rep.get("summary", "N/A"), vc=WHITE)

        hitl = bool(rep.get("hitl_review_required", False))
        _kv("HITL review", "REQUIRED" if hitl else "Not required",
            vc=RED if hitl else GREEN)
        if hitl and rep.get("hitl_reason"):
            _kv("HITL reason", rep["hitl_reason"], vc=RED)

        if rep.get("final_sc") is not None:
            _kv("Final Sc",  f"{rep['final_sc']:.4f}", vc=CYAN)
        if rep.get("final_sigma2") is not None:
            _kv("Final σ²",  f"{rep['final_sigma2']:.6f}", vc=DIM)
        if rep.get("ece_estimate") is not None:
            _kv("ECE",       f"{rep['ece_estimate']:.4f}  (paper target ≤ 0.08)",
                vc=GREEN if rep["ece_estimate"] <= 0.10 else YELLOW)

    except (json.JSONDecodeError, TypeError, AttributeError):
        print(f"{DIM}{report_json[:800]}{RESET}")
    print()
    _rule()


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive Q&A header
# ═══════════════════════════════════════════════════════════════════════════════

def print_qa_header(expert_label: str) -> None:
    print()
    _rule()
    print(
        f"\n{DIM}  Post-diagnosis Q&A  ·  Expert: {expert_label}"
        f"  ·  Type 'exit' or Ctrl-C to quit\n"
        f"  Commands: 'zoom <region>'  ·  'switch clinical'  ·  'switch reasoning'{RESET}"
    )
    _rule()
