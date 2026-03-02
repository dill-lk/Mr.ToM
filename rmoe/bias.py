"""
rmoe/bias.py — Cognitive Bias Detector for ARLL reasoning output.

Implements paper §"Error Patterns and Bias Mitigation" (Table 2):

  Bias Type                    Frequency   Clinical Impact
  ─────────────────────────────────────────────────────────────────
  Anchoring Bias               14.3 %      Delayed surgical intervention
  Difficulty w/ Conflicting    21.4 %      Inconsistent treatment recs
  Limited Alternative Consid.  28.6 %      Missed secondary pathologies
  Overthinking / Rationaliz.   35.7 %      Increased hallucination risk

Paper note (§ Cognitive Modeling, para 5):
  "Research suggests a strong correlation between the length of the
   reasoning chain and the probability of error. Extended explanations
   often signal latent uncertainty or an attempt to rationalize a
   conclusion that is not fully supported by the primary data."

The detector runs automatically after every ARLL pass and returns a
BiasReport that is:
  • printed to the terminal during the diagnostic run
  • stored in the IterationTrace audit record
  • injected back into the next ARLL prompt as a self-correction hint

Bias flags do NOT block the pipeline — they raise a warning and add a
correction hint so the ARLL can self-audit on the next iteration.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Bias taxonomy (paper Table 2)
# ═══════════════════════════════════════════════════════════════════════════════

class BiasType(Enum):
    Anchoring            = "Anchoring Bias"
    ConflictingData      = "Difficulty with Conflicting Data"
    LimitedAlternatives  = "Limited Alternative Consideration"
    Overthinking         = "Overthinking / Rationalization"


# Paper-reported baseline frequencies (used for calibrated thresholds)
_PAPER_FREQ: Dict[BiasType, float] = {
    BiasType.Anchoring:           0.143,
    BiasType.ConflictingData:     0.214,
    BiasType.LimitedAlternatives: 0.286,
    BiasType.Overthinking:        0.357,
}

# Clinical-impact severity (for report colouring)
_CLINICAL_IMPACT: Dict[BiasType, str] = {
    BiasType.Anchoring:           "Risk of delayed surgical intervention",
    BiasType.ConflictingData:     "Inconsistent treatment recommendations",
    BiasType.LimitedAlternatives: "Missed secondary pathologies",
    BiasType.Overthinking:        "Increased hallucination probability",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BiasFlag:
    """One detected bias instance."""
    bias_type:   BiasType
    confidence:  float          # 0–1, how confident we are this bias is present
    evidence:    str            # human-readable evidence snippet
    correction:  str            # self-correction hint for next ARLL iteration


@dataclass
class BiasReport:
    """Complete bias analysis for one ARLL reasoning pass."""
    flags:           List[BiasFlag] = field(default_factory=list)
    reasoning_tokens: int           = 0    # approximate CoT length in chars
    hypothesis_count: int           = 0
    top1_dominance:  float          = 0.0  # P(top1) / sum(P all)
    entropy:         float          = 0.0  # DDx entropy
    clean:           bool           = True # no significant biases detected

    def correction_hints(self) -> str:
        """Concatenated correction hints for ARLL self-reflection injection."""
        if not self.flags:
            return ""
        lines = ["[BIAS AUDIT — self-correction required]"]
        for f in self.flags:
            lines.append(f"  ⚠ {f.bias_type.value}: {f.evidence}")
            lines.append(f"    → {f.correction}")
        return "\n".join(lines)

    def worst_bias(self) -> Optional[BiasFlag]:
        if not self.flags:
            return None
        return max(self.flags, key=lambda f: f.confidence)


# ═══════════════════════════════════════════════════════════════════════════════
#  CognitiveBiasDetector
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveBiasDetector:
    """
    Detect cognitive biases in ARLL reasoning output (paper §Bias Mitigation).

    Inputs:
        cot_text          — the Chain-of-Thought reasoning string
        ddx_hypotheses    — list of {"diagnosis": str, "probability": float}
        temporal_note     — interval change note from TemporalComparator
        sc                — current confidence score

    Detection logic:
        Anchoring          → top-1 hypothesis probability > 0.75 of total mass
                             AND CoT references top hypothesis name ≥ 5×
                             while barely mentioning alternatives
        ConflictingData    → temporal note shows Progressed/New AND
                             CoT does not contain temporal keywords
        LimitedAlternatives→ fewer than 3 distinct hypotheses in DDx
        Overthinking       → CoT character length > threshold (2 000 chars)
                             AND confidence < 0.80 (rationalization heuristic)
    """

    # Tunable thresholds
    ANCHORING_DOMINANCE_THRESHOLD:  float = 0.65   # top-1 > 65% of total
    ANCHORING_MENTION_RATIO:        float = 5.0    # top-1 mentioned N× more
    CONFLICTING_KEYWORDS = frozenset({
        "temporal", "prior", "interval", "progression",
        "change", "previous", "compared", "months",
    })
    MIN_HYPOTHESES:          int   = 3
    OVERTHINKING_CHAR_LIMIT: int   = 2_000
    OVERTHINKING_SC_LIMIT:   float = 0.80

    def __init__(
        self,
        anchoring_threshold:  float = ANCHORING_DOMINANCE_THRESHOLD,
        min_hypotheses:       int   = MIN_HYPOTHESES,
        overthinking_limit:   int   = OVERTHINKING_CHAR_LIMIT,
    ) -> None:
        self._anchor_thr     = anchoring_threshold
        self._min_hyp        = min_hypotheses
        self._ot_limit       = overthinking_limit

    def analyse(
        self,
        cot_text:          str,
        ddx_hypotheses:    List[Dict],
        temporal_note:     str   = "",
        sc:                float = 1.0,
    ) -> BiasReport:
        """Run all bias checks and return a BiasReport."""
        flags: List[BiasFlag] = []

        probs = [h.get("probability", 0.0) for h in ddx_hypotheses]
        names = [h.get("diagnosis", "") for h in ddx_hypotheses]

        entropy   = self._entropy(probs)
        dominance = self._top1_dominance(probs)
        n_hyp     = len([p for p in probs if p > 0.01])

        # ── 1. Anchoring Bias ─────────────────────────────────────────────────
        if names and dominance > self._anchor_thr:
            top_name   = names[0] if names else ""
            top_count  = cot_text.lower().count(top_name.lower().split()[0])
            alt_count  = sum(
                cot_text.lower().count(n.lower().split()[0])
                for n in names[1:] if n
            ) or 1
            ratio = top_count / alt_count
            if ratio > self.ANCHORING_MENTION_RATIO:
                flags.append(BiasFlag(
                    bias_type=BiasType.Anchoring,
                    confidence=min(1.0, dominance + (ratio - 5) * 0.02),
                    evidence=(
                        f"Top hypothesis '{top_name}' dominates "
                        f"probability mass ({dominance:.0%}) and is mentioned "
                        f"{ratio:.1f}× more than alternatives in CoT."
                    ),
                    correction=(
                        "Re-examine top-2 and top-3 alternatives with equal rigour. "
                        "Explicitly rule out each competing hypothesis with evidence. "
                        "Do not repeat the leading diagnosis more than once before "
                        "considering alternatives."
                    ),
                ))

        # ── 2. Conflicting Data Handling ──────────────────────────────────────
        if temporal_note and "progressed" in temporal_note.lower() or \
           "new finding" in temporal_note.lower():
            kw_present = any(kw in cot_text.lower() for kw in self.CONFLICTING_KEYWORDS)
            if not kw_present:
                flags.append(BiasFlag(
                    bias_type=BiasType.ConflictingData,
                    confidence=0.75,
                    evidence=(
                        "Temporal note indicates disease progression / new finding, "
                        "but CoT reasoning does not reference interval change or "
                        "temporal keywords."
                    ),
                    correction=(
                        "Incorporate temporal interval change into reasoning. "
                        "State how lesion progression affects the differential. "
                        "Adjust confidence downward if new progression is unexplained."
                    ),
                ))

        # ── 3. Limited Alternative Consideration ──────────────────────────────
        if n_hyp < self._min_hyp:
            flags.append(BiasFlag(
                bias_type=BiasType.LimitedAlternatives,
                confidence=min(1.0, 1.0 - n_hyp / self._min_hyp),
                evidence=(
                    f"DDx ensemble contains only {n_hyp} hypothesis/hypotheses "
                    f"(threshold: ≥ {self._min_hyp}). Secondary pathologies may "
                    "be missed."
                ),
                correction=(
                    f"Generate at least {self._min_hyp} distinct diagnoses with "
                    "non-trivial probability (> 1%). Consider rare-but-dangerous "
                    "diagnoses (e.g., malignancy, vascular emergency) even if less "
                    "likely, and assign explicit probability to each."
                ),
            ))

        # ── 4. Overthinking / Rationalization ─────────────────────────────────
        cot_len = len(cot_text)
        if cot_len > self._ot_limit and sc < self.OVERTHINKING_SC_LIMIT:
            flags.append(BiasFlag(
                bias_type=BiasType.Overthinking,
                confidence=min(1.0, (cot_len / self._ot_limit - 1) * 0.4 +
                               (1 - sc) * 0.6),
                evidence=(
                    f"CoT reasoning is {cot_len} characters long (limit: "
                    f"{self._ot_limit}), while confidence Sc = {sc:.4f} < 0.80. "
                    "Extended rationale with low confidence is a hallucination risk."
                ),
                correction=(
                    "Shorten the reasoning chain. State the top conclusion in ≤ 3 "
                    "sentences. If genuine uncertainty remains, emit #wanna# rather "
                    "than rationalising a forced conclusion. Do not pad reasoning "
                    "to appear thorough."
                ),
            ))

        report = BiasReport(
            flags=flags,
            reasoning_tokens=cot_len,
            hypothesis_count=n_hyp,
            top1_dominance=dominance,
            entropy=entropy,
            clean=len(flags) == 0,
        )
        return report

    # ── Statistics helpers ────────────────────────────────────────────────────

    @staticmethod
    def _entropy(probs: List[float]) -> float:
        total = sum(probs) or 1.0
        h = 0.0
        for p in probs:
            p /= total
            if p > 0:
                h -= p * math.log2(p)
        return h

    @staticmethod
    def _top1_dominance(probs: List[float]) -> float:
        if not probs:
            return 0.0
        total = sum(probs) or 1.0
        return max(probs) / total


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretty printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_bias_report(report: BiasReport) -> None:
    """Print a bias report to the terminal (uses rmoe.ui colours)."""
    try:
        from rmoe.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, _rule
    except ImportError:
        BOLD = CYAN = DIM = GREEN = RED = RESET = YELLOW = ""
        def _rule(): print("─" * 72)

    _rule()
    if report.clean:
        print(f"  {GREEN}✓ Bias Audit PASS{RESET}  "
              f"entropy={report.entropy:.3f}  "
              f"dominance={report.top1_dominance:.2f}  "
              f"CoT={report.reasoning_tokens} chars")
        return

    print(f"  {YELLOW}{BOLD}⚠ Bias Audit — {len(report.flags)} flag(s){RESET}  "
          f"entropy={report.entropy:.3f}  dominance={report.top1_dominance:.2f}")

    for flag in report.flags:
        sev = RED if flag.confidence > 0.70 else YELLOW
        print(f"\n  {sev}[{flag.bias_type.value}]  conf={flag.confidence:.2f}{RESET}")
        print(f"  {DIM}Evidence:{RESET}   {flag.evidence}")
        print(f"  {CYAN}Correction:{RESET} {flag.correction}")
        impact = _CLINICAL_IMPACT.get(flag.bias_type, "")
        if impact:
            print(f"  {DIM}Clinical impact: {impact}{RESET}")
