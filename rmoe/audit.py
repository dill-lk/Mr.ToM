"""
rmoe/audit.py — Session audit trail and research report generator.

Produces:
  1. JSON audit trail   — machine-readable per-iteration log for HITL review
  2. SessionReport      — human-readable text report, ready for clinical audit
                          or paper appendix (LaTeX table snippet included)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import List, Optional

from rmoe.models import RunSummary


# ═══════════════════════════════════════════════════════════════════════════════
#  Audit Logger
# ═══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Writes a structured JSON audit trail for HITL review and reproducibility.

    The audit log records every inference event, iteration trace,
    doctor feedback, and the final clinical report so the complete
    reasoning chain can be replayed and validated.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path    = path
        self._events: List[dict] = []

    def log(self, event: str, data: dict) -> None:
        self._events.append({
            "timestamp": time.time(),
            "event":     event,
            **data,
        })

    def flush(self, summary: RunSummary) -> None:
        if not self._path:
            return
        audit = {
            "schema_version":  "2.0",
            "session_id":      summary.session_id,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "success":         summary.success,
            "escalated":       summary.escalated_to_human,
            "iterations":      summary.iterations_executed,
            "total_elapsed_s": round(summary.total_elapsed_s, 3),
            "models": {
                "vision":    os.path.basename(summary.model_vision),
                "reasoning": os.path.basename(summary.model_reasoning),
                "clinical":  os.path.basename(summary.model_clinical),
            },
            "trace": [
                {
                    "iteration":    t.iteration,
                    "decision":     t.decision,
                    "sc":           round(t.metrics.confidence, 4),
                    "sigma2":       round(t.metrics.ddx_variance, 6),
                    "entropy_sc":   round(t.metrics.predictive_entropy, 4),
                    "entropy_ddx":  round(t.metrics.ddx_entropy, 4),
                    "elapsed_s":    round(t.elapsed_s, 3),
                    "ddx_ensemble": t.ddx_ensemble,
                    "rag_refs":     t.rag_references,
                    "temporal":     t.temporal_note,
                    "doctor_hint":  t.doctor_feedback,
                }
                for t in summary.trace
            ],
            "calibration_bins": [
                {"confidence": round(c, 4), "accuracy": a, "count": n}
                for c, a, n in summary.calibration_bins
            ],
            "events": self._events,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)),
                        exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(audit, fh, indent=2)
        except OSError as exc:
            print(f"[audit] Could not write: {exc}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
#  Session Report Generator
# ═══════════════════════════════════════════════════════════════════════════════

class SessionReportGenerator:
    """
    Generate a human-readable research-quality session report from a RunSummary.

    Includes:
      - System configuration
      - Per-iteration DDx evolution table
      - Uncertainty metric table
      - Final clinical report (formatted)
      - LaTeX table snippet for paper inclusion
    """

    def generate(self, summary: RunSummary) -> str:
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("  R-MoE v2.0 — DIAGNOSTIC SESSION REPORT")
        lines.append("  Recursive Multi-Agent Mixture-of-Experts")
        lines.append("=" * 72)
        lines.append(f"  Session ID  : {summary.session_id}")
        lines.append(f"  Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        lines.append(f"  Image       : {os.path.basename(summary.image_path)}")
        lines.append(f"  Prior Scan  : {os.path.basename(summary.prior_image_path) or 'None'}")
        lines.append(f"  Status      : {'SUCCESS' if summary.success else 'ESCALATED' if summary.escalated_to_human else 'FAILED'}")
        lines.append(f"  Iterations  : {summary.iterations_executed}")
        lines.append(f"  Elapsed     : {summary.total_elapsed_s:.1f} s")
        lines.append("")

        lines.append("── Models ───────────────────────────────────────────────────")
        lines.append(f"  Vision (MPE)      : {os.path.basename(summary.model_vision)}")
        lines.append(f"  Reasoning (ARLL)  : {os.path.basename(summary.model_reasoning)}")
        lines.append(f"  Clinical (CSR)    : {os.path.basename(summary.model_clinical)}")
        lines.append("")

        lines.append("── Iteration Trace ──────────────────────────────────────────")
        lines.append(
            f"  {'#':>3}  {'Decision':<24}  {'Sc':>8}  {'σ²':>8}  {'H(Sc)':>7}  {'t(s)':>6}"
        )
        lines.append("  " + "-" * 66)
        for t in summary.trace:
            lines.append(
                f"  {t.iteration:>3}  {t.decision:<24}  "
                f"{t.metrics.confidence:>8.4f}  "
                f"{t.metrics.ddx_variance:>8.4f}  "
                f"{t.metrics.predictive_entropy:>7.4f}  "
                f"{t.elapsed_s:>6.1f}"
            )
        lines.append("")

        # DDx evolution
        for t in summary.trace:
            hyps = t.ddx_ensemble.get("hypotheses", [])
            if hyps:
                lines.append(f"  Iteration {t.iteration} DDx  (Sc={t.ddx_ensemble.get('sc',0):.4f})")
                for h in sorted(hyps, key=lambda x: x["probability"], reverse=True)[:4]:
                    bar = "█" * max(1, int(h["probability"] * 20))
                    lines.append(f"    {bar:<20} {h['probability']:.3f}  {h['diagnosis']}")
                lines.append("")

        # Final report
        if summary.final_report_json:
            lines.append("── Clinical Report ──────────────────────────────────────────")
            try:
                rep = json.loads(summary.final_report_json)
                lines.append(f"  ICD-11    : {rep.get('standard', 'N/A')}")
                lines.append(f"  SNOMED CT : {rep.get('snomed_ct', 'N/A')}")
                rs = rep.get("risk_stratification", {})
                if isinstance(rs, dict):
                    lines.append(
                        f"  {rs.get('scale','Risk')} : "
                        f"{rs.get('score','N/A')} — {rs.get('interpretation','')}"
                    )
                lines.append(f"  Final Sc  : {rep.get('final_sc', 'N/A')}")
                lines.append(f"  ECE       : {rep.get('ece_estimate', 'N/A')}")
                lines.append("")
                lines.append("  IMPRESSION:")
                narr = rep.get("narrative", rep.get("summary", ""))
                for chunk in [narr[i:i+70] for i in range(0, min(len(narr), 700), 70)]:
                    lines.append(f"    {chunk}")
                lines.append("")
                lines.append("  RECOMMENDATIONS:")
                for line in rep.get("treatment_recommendations", "").split("\n")[:4]:
                    if line.strip():
                        lines.append(f"    {line.strip()}")
            except (json.JSONDecodeError, TypeError):
                lines.append(summary.final_report_json[:400])

        lines.append("")
        lines.append("── LaTeX Snippet (for paper inclusion) ──────────────────────")
        lines.extend(self._latex_table(summary))
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    def _latex_table(self, summary: RunSummary) -> List[str]:
        """Generate a LaTeX tabular block for the paper appendix."""
        rows = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \begin{tabular}{lccccc}",
            r"    \hline",
            r"    \textbf{Iter} & \textbf{Sc} & \textbf{$\sigma^2$} & "
            r"\textbf{H(Sc)} & \textbf{t(s)} & \textbf{Decision} \\",
            r"    \hline",
        ]
        for t in summary.trace:
            rows.append(
                f"    {t.iteration} & {t.metrics.confidence:.4f} & "
                f"{t.metrics.ddx_variance:.4f} & "
                f"{t.metrics.predictive_entropy:.4f} & "
                f"{t.elapsed_s:.1f} & "
                f"{t.decision.replace('_', ' ')} \\\\"
            )
        rows += [
            r"    \hline",
            r"  \end{tabular}",
            r"  \caption{R-MoE iteration trace}",
            r"  \label{tab:rmoe_trace}",
            r"\end{table}",
        ]
        return ["  " + r for r in rows]
