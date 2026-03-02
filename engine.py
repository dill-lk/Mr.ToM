#!/usr/bin/env python3
"""
engine.py — R-MoE v2.0 CLI entry-point.

Recursive Multi-Agent Mixture-of-Experts for Autonomous Clinical Diagnostics.

──────────────────────────────────────────────────────────────────────────────
WHY PYTHON (llama-cpp-python) AND NOT C++?
──────────────────────────────────────────────────────────────────────────────
The R-MoE architecture was originally designed and validated in a C++
implementation using the llama.cpp native API (GGML/CUDA kernels).  The C++
engine is kept in the repository under legacy/ for reference and academic
reproducibility.

At runtime we use llama-cpp-python because:
  1. pip-installing a pre-built CUDA wheel takes < 60 s on Colab.
  2. Compiling the C++ backend with -DGGML_CUDA=ON takes 15–20 minutes and
     frequently exhausts Colab free-tier rate-limits before the build finishes.
  3. llama-cpp-python calls the *identical* GGML/CUDA kernels internally —
     there is zero inference accuracy difference vs the C++ binary.
  4. The Python engine adds multi-temperature ensemble, Vector RAG, ICD-11/
     SNOMED CT ontology, ECE calibration, and HITL loop — none of which are
     practical to build interactively inside a C++ binary.

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
  # Demo / mock mode (no model files needed):
  python engine.py --image patient.png

  # Full GPU run (Colab):
  python engine.py --image /content/models/test_patient.png \\
                   --settings settings/rmoe_settings.json  \\
                   --audit-log audit_trail.json             \\
                   --eval

  # Interactive Q&A + zoom commands:
  python engine.py --image patient.png --chat-target auto

  # Non-interactive (no HITL prompts):
  python engine.py --image patient.png --hitl disabled

  # With prior scan (temporal analysis):
  python engine.py --image current.png --prior prior.png

  # Print benchmark comparison table only:
  python engine.py --benchmark-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── Make sure the repo root is on PYTHONPATH so `rmoe` is importable ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Lazy import guard ─────────────────────────────────────────────────────────
try:
    from rmoe import (
        MrTom, WannaStateMachine, HITLMode, ExpertTarget,
        print_banner, print_run_summary, print_clinical_report,
    )
    from rmoe.ui import (
        BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW, WHITE,
        _rule, _kv,
    )
    from rmoe.charts import (
        sc_progression_chart, ddx_evolution_chart, uncertainty_heatmap,
        reliability_diagram, benchmark_comparison, _paper_calibration_bins,
    )
    from rmoe.calibration import print_reliability_diagram
    from rmoe.audit import SessionReportGenerator
    from rmoe.hitl import ExpertQueryRouter
except ImportError as exc:
    print(f"\n❌  Cannot import rmoe package: {exc}")
    print("   Run from the repository root: python engine.py")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engine.py",
        description="R-MoE v2.0 — Recursive Multi-Agent MoE Clinical Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    p.add_argument("--image",        default="test_patient.png",
                   help="Path to patient image (.png/.jpg/.dcm)  [default: test_patient.png]")
    p.add_argument("--prior",        dest="prior_image", default=None,
                   help="Prior scan for temporal comparison")
    p.add_argument("--settings",     default="settings/rmoe_settings.json",
                   help="JSON settings file")
    p.add_argument("--prompts-dir",  default="prompts",
                   help="Directory containing system-prompt .txt files")

    # ── Models ────────────────────────────────────────────────────────────────
    p.add_argument("--vision-proj",  default=None, help="CLIP mmproj .gguf path")
    p.add_argument("--vision-text",  default=None, help="Vision text backbone .gguf path")
    p.add_argument("--reasoning",    default=None, help="ARLL model .gguf path")
    p.add_argument("--clinical",     default=None, help="CSR model .gguf path")

    # ── Inference hyper-params ────────────────────────────────────────────────
    p.add_argument("--temperature",  type=float, default=None,
                   help="Sampling temperature (default: from settings, usually 0.2)")
    p.add_argument("--n-predict",    type=int,   default=None,
                   help="Max tokens per inference step")
    p.add_argument("--n-gpu-layers", type=int,   default=None,
                   help="GPU layers to offload (-1=all, 0=CPU)")
    p.add_argument("--threshold",    type=float, default=None,
                   help="Confidence threshold θ  (default: 0.90)")
    p.add_argument("--max-iter",     type=int,   default=None,
                   help="Max recursive iterations  (default: 3)")

    # ── Session mode ──────────────────────────────────────────────────────────
    p.add_argument("--hitl",         choices=["interactive", "auto", "disabled"],
                   default="auto",
                   help="HITL mode: interactive=always prompt, auto=TTY-detect, disabled=silent")
    p.add_argument("--chat-target",  choices=["reasoning", "clinical", "auto", "none"],
                   default="auto",
                   help="Post-diagnosis Q&A expert  (none = skip Q&A)")
    p.add_argument("--audit-log",    default=None,
                   help="Write JSON audit trail to this path")
    p.add_argument("--session-report", default=None,
                   help="Write human-readable session report to this path")

    # ── Output flags ──────────────────────────────────────────────────────────
    p.add_argument("--eval",         action="store_true",
                   help="Print ECE calibration chart + benchmark comparison")
    p.add_argument("--charts",       action="store_true",
                   help="Print all ASCII charts (Sc progress, DDx evolution, etc.)")
    p.add_argument("--benchmark-only", action="store_true",
                   help="Print benchmark comparison table and exit")
    p.add_argument("--benchmark",    action="store_true",
                   help="Run full benchmark evaluation on dataset and print results")
    p.add_argument("--benchmark-dataset", default=None,
                   help="Path to benchmark CSV (default: built-in 20-case dataset)")
    p.add_argument("--benchmark-max", type=int, default=None,
                   help="Limit benchmark to first N cases (useful for quick test)")
    p.add_argument("--save-results", default=None,
                   help="Save benchmark JSON results to this path")
    p.add_argument("--latex",        action="store_true",
                   help="Print LaTeX tabular block for paper Table 1")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress HITL prompts and banner (useful for scripting)")

    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Benchmark-only shortcut ───────────────────────────────────────────────
    if args.benchmark_only:
        print_banner()
        benchmark_comparison()
        reliability_diagram(_paper_calibration_bins(), 0.08)
        return 0

    # ── Full benchmark evaluation (paper §4 metrics) ──────────────────────────
    if args.benchmark:
        from rmoe.eval import BenchmarkRunner, BenchmarkDataset
        if not args.quiet:
            print_banner()
        print(f"\n  Loading benchmark dataset …")
        dataset = BenchmarkDataset(args.benchmark_dataset)

        sm      = WannaStateMachine(hard_limit=args.max_iter or 3,
                                    threshold=args.threshold or 0.90)
        mr_tom  = MrTom(sm, hitl_mode=HITLMode.Disabled)
        if os.path.exists(args.settings):
            mr_tom.load_settings(args.settings)

        runner  = BenchmarkRunner(mr_tom, verbose=not args.quiet)
        results = runner.run(dataset, max_cases=args.benchmark_max)
        runner.print_report(results)

        if args.latex:
            print("\n" + runner.print_latex(results))

        if args.save_results:
            runner.save_results(results, args.save_results)
            print(f"\n  Results saved → {args.save_results}")
        return 0

    if not args.quiet:
        print_banner()

    # ── Build MrTom ───────────────────────────────────────────────────────────
    threshold = args.threshold or 0.90
    max_iter  = args.max_iter  or 3
    sm        = WannaStateMachine(hard_limit=max_iter, threshold=threshold)

    hitl_map  = {
        "interactive": HITLMode.Interactive,
        "auto":        HITLMode.Auto,
        "disabled":    HITLMode.Disabled,
    }
    hitl_mode = hitl_map.get(args.hitl, HITLMode.Auto)
    if args.quiet:
        hitl_mode = HITLMode.Disabled

    mr_tom = MrTom(sm, hitl_mode=hitl_mode, prompt_dir=args.prompts_dir)

    # Load JSON settings first (may be overridden by CLI flags below)
    if os.path.exists(args.settings):
        mr_tom.load_settings(args.settings)

    # ── CLI overrides ─────────────────────────────────────────────────────────
    if args.vision_proj and args.vision_text:
        mr_tom.set_vision_model(args.vision_proj, args.vision_text)
    if args.reasoning:
        mr_tom.set_reasoning_model(args.reasoning)
    if args.clinical:
        mr_tom.set_clinical_model(args.clinical)
    if args.temperature is not None:
        mr_tom.set_temperature(args.temperature)
    if args.n_predict is not None:
        mr_tom.set_max_tokens(args.n_predict)
    if args.n_gpu_layers is not None:
        mr_tom.set_gpu_layers(args.n_gpu_layers)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    t0 = time.time()
    summary = mr_tom.process_patient_case(
        image_path=args.image,
        prior_image=args.prior_image,
        audit_log_path=args.audit_log,
    )
    elapsed = time.time() - t0

    # ── Print results ─────────────────────────────────────────────────────────
    print_run_summary(summary, max_iter)

    if summary.final_report_json:
        print_clinical_report(summary.final_report_json)

    # ── Charts ────────────────────────────────────────────────────────────────
    if args.charts or args.eval:
        sc_progression_chart(summary.trace, threshold)
        ddx_evolution_chart(summary.trace)
        uncertainty_heatmap(summary.trace)
        print_reliability_diagram()
        benchmark_comparison()

    # ── Session report ────────────────────────────────────────────────────────
    if args.session_report:
        report_text = SessionReportGenerator().generate(summary)
        try:
            with open(args.session_report, "w", encoding="utf-8") as fh:
                fh.write(report_text)
            print(f"\n  {DIM}Session report written → {args.session_report}{RESET}")
        except OSError as exc:
            print(f"  {RED}Could not write session report: {exc}{RESET}")

    # ── Interactive Q&A loop ──────────────────────────────────────────────────
    if args.chat_target != "none" and not args.quiet:
        if args.chat_target == "clinical":
            default_target = ExpertTarget.Clinical
        elif args.chat_target == "reasoning":
            default_target = ExpertTarget.Reasoning
        else:
            default_target = None   # auto-route

        expert_label = (
            ExpertQueryRouter.label(default_target)
            if default_target else "auto-routed (ExpertQueryRouter)"
        )
        print(
            f"\n{DIM}  Post-diagnosis Q&A  ·  Expert: {expert_label}\n"
            f"  Commands: 'zoom <region>'  ·  'switch clinical'  ·  "
            f"'switch reasoning'  ·  'exit'{RESET}"
        )
        _rule()

        mr_tom.run_qa_loop(default_target=default_target)

    # ── Final status ──────────────────────────────────────────────────────────
    _rule()
    sc  = GREEN if summary.success else (YELLOW if summary.escalated_to_human else RED)
    msg = ("✓ Diagnosis complete"
           if summary.success
           else ("⚠ Escalated to human radiologist"
                 if summary.escalated_to_human
                 else "✗ Pipeline failed"))
    print(f"\n  {sc}{BOLD}{msg}{RESET}  "
          f"({summary.iterations_executed} iter · {elapsed:.1f} s)\n")

    return 0 if (summary.success or summary.escalated_to_human) else 1


if __name__ == "__main__":
    sys.exit(main())
