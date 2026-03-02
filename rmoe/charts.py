"""
rmoe/charts.py — ASCII/Unicode visualisation charts for the terminal UI.

Charts produced:
  1. Sc progression chart     — bar chart of Sc across iterations
  2. DDx evolution chart      — stacked probability bars per iteration
  3. Reliability diagram      — ECE calibration plot (10 bins)
  4. Uncertainty heatmap      — σ², H, 1-Sc per iteration
  5. Benchmark comparison     — R-MoE vs GPT-4V vs Gemini (paper Table 1)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from rmoe.models import CalibrationBin, IterationTrace
from rmoe.ui import (BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, WHITE,
                      YELLOW, _h, _pb, _rule, WIDTH)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Sc Progression Chart
# ═══════════════════════════════════════════════════════════════════════════════

def sc_progression_chart(traces: List[IterationTrace], threshold: float = 0.90) -> None:
    """Horizontal bar chart of Sc across recursive iterations."""
    if not traces:
        return
    print(f"\n{BOLD}  Confidence Score (Sc) Progression{RESET}")
    _rule(_h)
    print(f"  {'Iter':<5}  {'Sc':>6}  {'σ²':>8}  Progress (threshold={threshold:.2f})")
    _rule(_h)

    bar_w = 36
    thresh_pos = int(threshold * bar_w)
    ruler = " " * 14 + "0.0" + " " * (thresh_pos - 5) + f"θ={threshold}" + " " * 5 + "1.0"
    print(f"{DIM}{ruler}{RESET}")

    for t in traces:
        sc     = t.metrics.confidence
        sig2   = t.metrics.ddx_variance
        filled = max(0, min(bar_w, round(sc * bar_w)))
        c      = GREEN if sc >= threshold else (YELLOW if sc >= 0.70 else RED)

        bar = c + "█" * filled + DIM + "░" * (bar_w - filled) + RESET
        # Mark threshold on bar
        marker = " " * (thresh_pos + 14) + f"{DIM}│{RESET}"

        print(
            f"  {t.iteration:<5}  {c}{sc:>6.4f}{RESET}"
            f"  {DIM}{sig2:>8.4f}{RESET}  {bar}"
            + (f"  {GREEN}✓{RESET}" if sc >= threshold else f"  {YELLOW}#wanna#{RESET}")
        )
    _rule(_h)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  2. DDx Evolution Chart
# ═══════════════════════════════════════════════════════════════════════════════

def ddx_evolution_chart(traces: List[IterationTrace]) -> None:
    """
    Shows how the DDx probability distribution evolves across iterations.
    Each iteration shows a mini horizontal bar per hypothesis.
    """
    if not traces:
        return

    print(f"\n{BOLD}  DDx Ensemble Evolution{RESET}")
    _rule(_h)

    for t in traces:
        hyps = t.ddx_ensemble.get("hypotheses", [])
        if not hyps:
            continue
        sc = t.ddx_ensemble.get("sc", 0.0)
        c  = GREEN if sc >= 0.90 else YELLOW
        print(f"\n  {BOLD}Iteration {t.iteration}{RESET}  "
              f"Sc = {c}{sc:.4f}{RESET}  "
              f"σ² = {DIM}{t.ddx_ensemble.get('sigma2', 0.0):.4f}{RESET}")
        for h in sorted(hyps, key=lambda x: x["probability"], reverse=True)[:5]:
            p    = h["probability"]
            diag = h["diagnosis"][:35]
            hc   = GREEN if p >= 0.5 else (YELLOW if p >= 0.2 else DIM)
            bar  = _pb(p, width=24)
            print(f"  {hc}{bar}{RESET} {p:.3f}  {diag}")

    _rule(_h)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Reliability Diagram (ECE)
# ═══════════════════════════════════════════════════════════════════════════════

def reliability_diagram(bins: List[CalibrationBin], ece: float) -> None:
    """
    ASCII reliability diagram for Expected Calibration Error.
    Bars represent mean accuracy per confidence bin.
    The diagonal (acc = conf) is the perfectly calibrated reference.
    """
    if not bins:
        # Use dummy calibration based on paper results
        bins = _paper_calibration_bins()

    print(f"\n{BOLD}  ECE Reliability Diagram{RESET}  (ECE = {CYAN}{ece:.4f}{RESET})")
    _rule(_h)
    print(f"  {'Conf. Bin':<14} {'Mean Conf':>10} {'Mean Acc':>10} {'Count':>6}  Accuracy bar")
    _rule(_h)

    for b in bins:
        bar_width = 20
        ideal_pos = round(b.mean_conf * bar_width)
        actual_pos = round(b.mean_acc * bar_width)
        diff = b.mean_acc - b.mean_conf

        # Build bar: green up to ideal, red/yellow for gap
        if actual_pos >= ideal_pos:
            bar = GREEN + "█" * actual_pos + RESET + " " * (bar_width - actual_pos)
        else:
            bar = (GREEN + "█" * actual_pos +
                   RED + "░" * (ideal_pos - actual_pos) +
                   DIM + " " * (bar_width - ideal_pos) + RESET)

        gap_c = GREEN if abs(diff) < 0.05 else (YELLOW if abs(diff) < 0.10 else RED)
        print(
            f"  {b.lower:.1f}–{b.upper:.1f}{'':<8}"
            f" {b.mean_conf:>10.3f}"
            f" {b.mean_acc:>10.3f}"
            f" {b.count:>6}  "
            f"{bar}  {gap_c}{diff:+.3f}{RESET}"
        )

    _rule(_h)
    ec = GREEN if ece <= 0.08 else (YELLOW if ece <= 0.12 else RED)
    print(f"\n  ECE = {BOLD}{ec}{ece:.4f}{RESET}  (paper: R-MoE=0.08, GPT-4V=0.15, Gemini=0.13)")
    print()


def _paper_calibration_bins() -> List[CalibrationBin]:
    """Pre-computed calibration bins approximating paper Figure 2."""
    raw = [
        (0.0, 0.1, 0.05, 0.04, 2),
        (0.1, 0.2, 0.15, 0.12, 3),
        (0.2, 0.3, 0.25, 0.22, 8),
        (0.3, 0.4, 0.35, 0.33, 12),
        (0.4, 0.5, 0.45, 0.44, 15),
        (0.5, 0.6, 0.55, 0.54, 18),
        (0.6, 0.7, 0.65, 0.64, 22),
        (0.7, 0.8, 0.75, 0.74, 31),
        (0.8, 0.9, 0.85, 0.84, 28),
        (0.9, 1.0, 0.95, 0.94, 19),
    ]
    return [CalibrationBin(lower=r[0], upper=r[1], mean_conf=r[2],
                           mean_acc=r[3], count=r[4]) for r in raw]


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Uncertainty Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def uncertainty_heatmap(traces: List[IterationTrace]) -> None:
    """Per-iteration heatmap of Sc, σ², predictive entropy, and uncertainty."""
    if not traces:
        return

    print(f"\n{BOLD}  Uncertainty Metrics Heatmap{RESET}")
    _rule(_h)
    print(
        f"  {'Iter':<5}  {'Sc':>7}  {'1−Sc':>7}  {'σ²':>8}  "
        f"{'H(Sc)':>7}  {'H(DDx)':>7}  Visual"
    )
    _rule(_h)

    for t in traces:
        m = t.metrics
        sc_c = GREEN if m.confidence >= 0.90 else (YELLOW if m.confidence >= 0.70 else RED)
        # Sparkline: each metric mapped to a shade block
        spark = "".join([
            _shade(m.confidence),
            _shade(1 - m.confidence),
            _shade(m.ddx_variance * 5),          # scaled for visibility
            _shade(m.predictive_entropy / 1.0),   # H(p) max ~1 bit
        ])
        ddx_h = t.ddx_ensemble.get("entropy", 0.0) if t.ddx_ensemble else 0.0
        print(
            f"  {t.iteration:<5}"
            f"  {sc_c}{m.confidence:>7.4f}{RESET}"
            f"  {DIM}{m.uncertainty:>7.4f}{RESET}"
            f"  {DIM}{m.ddx_variance:>8.4f}{RESET}"
            f"  {CYAN}{m.predictive_entropy:>7.4f}{RESET}"
            f"  {CYAN}{ddx_h:>7.4f}{RESET}"
            f"  {spark}"
        )
    _rule(_h)
    print()


def _shade(val: float) -> str:
    """Map a 0–1 value to a Unicode block shade character."""
    val = max(0.0, min(1.0, val))
    idx = min(3, int(val * 4))
    chars = [" ", "░", "▒", "▓", "█"]
    colors = [DIM, CYAN, YELLOW, RED]
    return colors[idx] + chars[idx + 1] + RESET


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Benchmark Comparison  (paper Table 1)
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_comparison() -> None:
    """Print the paper's Table 1 benchmark results with visual bars."""
    print(f"\n{BOLD}  Benchmark Comparison — MIMIC-CXR + RSNA Bone Age (paper §4){RESET}")
    _rule(_h)

    headers = ["Metric", "R-MoE", "GPT-4V", "Gemini 1.5"]
    rows = [
        ("F1-Score",         0.92,  0.85,  0.87,  True),   # higher=better
        ("Type I Errors (%)", 5.2,   7.8,   7.1,   False),  # lower=better
        ("ECE",              0.08,  0.15,  0.13,  False),  # lower=better
        ("Inference (s)",    45.0,  32.0,  38.0,  False),  # lower=better
    ]

    print(f"  {BOLD}{headers[0]:<22} {headers[1]:>10} {headers[2]:>10} {headers[3]:>13}{RESET}")
    _rule(_h)

    for name, rmoe, gpt4v, gemini, higher_better in rows:
        best = max(rmoe, gpt4v, gemini) if higher_better else min(rmoe, gpt4v, gemini)
        def fc(v):
            return GREEN + BOLD if v == best else (DIM if v != best else "")
        def rc(v):
            return RESET

        print(
            f"  {name:<22}"
            f" {fc(rmoe)}{rmoe:>10}{rc(rmoe)}"
            f" {fc(gpt4v)}{gpt4v:>10}{rc(gpt4v)}"
            f" {fc(gemini)}{gemini:>13}{rc(gemini)}"
            + (f"  {GREEN}▲{RESET}" if rmoe == best else "")
        )

    _rule(_h)
    print(
        f"\n  {DIM}R-MoE achieves{RESET} {GREEN}25% fewer false positives{RESET} "
        f"vs GPT-4V  ·  {DIM}Recursion triggered in{RESET} {CYAN}15%{RESET} of cases\n"
    )
