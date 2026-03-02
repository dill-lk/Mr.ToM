"""
rmoe/hitl.py — Human-in-the-Loop (HITL) coordination.

Components:
  • HITLCoordinator   — prompts doctor, parses zoom commands, logs feedback
  • ExpertQueryRouter — keyword-based auto-routing to Reasoning or Clinical expert

Doctor-in-the-Loop features (v2.0):
  • "Show me the fracture site"  →  MPE zooms on that anatomical region
  • "Explain the findings"       →  routed to ARLL (reasoning expert)
  • "What treatment is needed?"  →  routed to CSR (clinical expert)
  • "switch clinical / reasoning"  →  explicit expert switch mid-session
"""
from __future__ import annotations

import sys
from typing import List, Optional

from rmoe.models import DoctorFeedback, ExpertTarget, HITLMode


# ═══════════════════════════════════════════════════════════════════════════════
#  Expert Query Router
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertQueryRouter:
    """
    Classify a doctor's free-text question to the most appropriate expert.
    Uses additive keyword scoring — no extra model required.

    Clinical expert (CSR / MedGemma-2B):
        Treatment, medication, surgery, urgent, biopsy, dose, discharge…

    Reasoning expert (ARLL / DeepSeek-R1-Distill):
        Probability, DDx, evidence, explain, fracture, imaging, confidence…
    """

    _CLINICAL_KW = [
        "treat", "treatment", "medication", "medicine", "drug", "prescri",
        "surgery", "operation", "immediate", "urgent", "dose", "follow-up",
        "refer", "biopsy", "procedure", "protocol", "discharge", "care",
        "manage", "admit", "antibiotics", "chemo", "radiation", "resect",
        "transplant", "intervention", "what should", "what to do",
    ]

    _REASONING_KW = [
        "probabilit", "likelihood", "chance", "fracture", "diagnos", "ddx",
        "differential", "explain", "finding", "evidence", "why", "how",
        "confidence", "uncertain", "scan", "imaging", "compar", "lesion",
        "mass", "opacity", "cot", "reasoning", "saliency", "ensemble",
        "sigma", "variance", "wanna", "show me", "zoom", "focus", "look at",
        "what is", "what does",
    ]

    @classmethod
    def route(cls, question: str) -> ExpertTarget:
        q = question.lower()
        c = sum(1 for kw in cls._CLINICAL_KW  if kw in q)
        r = sum(1 for kw in cls._REASONING_KW if kw in q)
        return ExpertTarget.Clinical if c > r else ExpertTarget.Reasoning

    @classmethod
    def label(cls, target: ExpertTarget) -> str:
        return ("CSR (MedGemma-2B · clinical / treatment)"
                if target == ExpertTarget.Clinical
                else "ARLL (DeepSeek-R1-Distill · diagnostic reasoning)")


# ═══════════════════════════════════════════════════════════════════════════════
#  HITL Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class HITLCoordinator:
    """
    Manages all doctor-facing interactions during the diagnostic pipeline:
      1. Prompt for clarification / zoom hint during #wanna# iterations
      2. Parse zoom commands ("Show me the fracture site")
      3. Provide an interactive post-diagnosis Q&A loop
    """

    # Keywords that indicate a zoom / focus command
    _ZOOM_KW = [
        "show", "zoom", "focus", "look at", "check", "highlight",
        "mark", "point", "fracture", "lesion", "mass", "opacity",
        "region", "area", "site", "spot", "vertebr", "rib", "lobe",
        "lung", "bone", "hip", "knee", "spine", "brain", "liver",
    ]

    def __init__(self, mode: HITLMode = HITLMode.Auto) -> None:
        self._mode = mode

    def is_interactive(self) -> bool:
        if self._mode == HITLMode.Disabled:
            return False
        if self._mode == HITLMode.Interactive:
            return True
        return sys.stdin.isatty()

    def prompt_wanna(
        self,
        request: str,
        payload: str,
        iteration: int,
    ) -> Optional[DoctorFeedback]:
        """
        Prompt the doctor for a zoom hint during a #wanna# iteration.
        Returns None if not interactive or doctor presses Enter (skip).
        """
        if not self.is_interactive():
            return None

        from rmoe.ui import YELLOW, BOLD, DIM, GREEN, CYAN, RESET

        print(
            f"\n{YELLOW}{BOLD}"
            f"  ╔═ R-MoE → DOCTOR  (Iteration {iteration} · Low confidence) ════╗\n"
            f"  ║  Requesting : {request:<52}║\n"
            f"  ║  Target     : {payload:<52}║\n"
            f"  ╠════════════════════════════════════════════════════════════════╣\n"
            f"  ║  Your hint narrows my next scan.  Examples:{' ' * 20}║\n"
            f"  ║   • 'Show me the fracture site'{' ' * 33}║\n"
            f"  ║   • 'Focus on the left upper lobe'{' ' * 29}║\n"
            f"  ║   • 'Zoom into T4-T6 vertebrae'{' ' * 34}║\n"
            f"  ║   (press Enter to let me auto-refocus){' ' * 25}║\n"
            f"  ╚════════════════════════════════════════════════════════════════╝\n"
            f"{RESET}"
        )
        print(f"  {GREEN}{BOLD}[DOCTOR]{RESET} ", end="", flush=True)
        try:
            raw = input().strip()
        except EOFError:
            return None

        if not raw:
            return None

        return self.parse_zoom_command(raw)

    def parse_zoom_command(self, text: str) -> DoctorFeedback:
        """
        Parse free text into a DoctorFeedback.
        Detects zoom/focus commands and extracts the target region.
        """
        is_zoom = any(kw in text.lower() for kw in self._ZOOM_KW)

        # Extract region: everything after the zoom verb
        region = text
        for kw in ["show me", "zoom into", "zoom on", "focus on",
                   "look at", "check", "highlight", "point to"]:
            if kw in text.lower():
                idx = text.lower().index(kw) + len(kw)
                region = text[idx:].strip(" .,;:")
                break

        return DoctorFeedback(
            message=text,
            zoom_region=region or text,
            is_zoom_command=is_zoom,
            raw_input=text,
        )

    def run_qa_loop(
        self,
        ask_fn,                   # Callable[[str, Optional[ExpertTarget]], str]
        default_target: Optional[ExpertTarget] = None,
    ) -> None:
        """
        Interactive post-diagnosis Q&A loop.

        The doctor types questions; ExpertQueryRouter decides which expert answers.
        Special commands:
          'switch clinical'   → force CSR expert for next question
          'switch reasoning'  → force ARLL expert
          'exit' / Ctrl-C     → end session
        """
        from rmoe.ui import (GREEN, BOLD, CYAN, DIM, RESET,
                              print_qa_header, _rule)
        from rmoe.ui import WHITE

        current_target = default_target   # None = auto-route
        expert_label   = (ExpertQueryRouter.label(current_target)
                          if current_target else "auto-routed")
        print_qa_header(expert_label)

        try:
            while True:
                print(f"\n  {GREEN}{BOLD}[DOCTOR]{RESET} ", end="", flush=True)
                try:
                    query = input()
                except EOFError:
                    break

                query = query.strip()
                if not query:
                    continue

                q_lower = query.lower()

                # Session control commands
                if q_lower in ("exit", "quit", "q"):
                    break
                if "switch clinical" in q_lower:
                    current_target = ExpertTarget.Clinical
                    print(f"  {DIM}[switched to CSR · clinical expert]{RESET}")
                    continue
                if "switch reasoning" in q_lower:
                    current_target = ExpertTarget.Reasoning
                    print(f"  {DIM}[switched to ARLL · reasoning expert]{RESET}")
                    continue
                if "switch auto" in q_lower:
                    current_target = None
                    print(f"  {DIM}[switched to auto-routing]{RESET}")
                    continue

                # Determine expert
                if current_target is None:
                    target   = ExpertQueryRouter.route(query)
                    label    = ExpertQueryRouter.label(target)
                    print(f"  {DIM}[routing → {label}]{RESET}")
                else:
                    target = current_target

                response = ask_fn(query, target)
                print(f"  {CYAN}[Mr.ToM]{RESET}  {response}")

        except KeyboardInterrupt:
            pass

        print()
        from rmoe.ui import _rule
        _rule()
        print(f"  {DIM}Session closed.{RESET}")
        _rule()
        print()
