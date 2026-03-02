"""
rmoe/temporal.py — Comparative Temporal Analysis module.

Implements paper §3 "Comparative Temporal Analysis":
  "Where prior imaging is available, detect interval changes across
   time-points to inform diagnosis confidence."

TemporalComparator compares a current scan to a prior scan and returns
a structured TemporalAnalysis with:
  • stability classification: Stable | Progressed | Regressed | New | Resolved
  • interval change magnitude (if images are available)
  • natural-language interval note for ARLL context injection
  • Sc adjustment factor — confidence bonus for "Stable", penalty for "Progressed"

When image files are not available (mock mode), the comparator uses the
diagnosis text and iteration number to synthesise a realistic note.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class ChangeClass(Enum):
    """Temporal interval change classification."""
    Stable      = "Stable"       # no significant change
    Progressed  = "Progressed"   # worsening (enlargement, new density)
    Regressed   = "Regressed"    # improvement (shrinkage, clearing)
    New         = "New"          # finding absent on prior, present now
    Resolved    = "Resolved"     # finding present on prior, absent now
    NoComparison = "NoComparison"# no prior scan available


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegionChange:
    """Quantified change for one anatomical region / ROI."""
    region: str
    current_size_mm:  float = 0.0
    prior_size_mm:    float = 0.0
    delta_mm:         float = 0.0       # positive = growth, negative = shrinkage
    delta_pct:        float = 0.0       # % change relative to prior size
    intensity_delta:  float = 0.0       # mean pixel intensity change (0–255)
    change_class:     ChangeClass = ChangeClass.Stable

    def to_note(self) -> str:
        if self.change_class == ChangeClass.New:
            return f"{self.region}: new finding (absent on prior)"
        if self.change_class == ChangeClass.Resolved:
            return f"{self.region}: resolved (no longer visible)"
        if self.prior_size_mm > 0 and self.current_size_mm > 0:
            sign = "+" if self.delta_mm >= 0 else ""
            return (
                f"{self.region}: {self.current_size_mm:.1f} mm "
                f"(prior {self.prior_size_mm:.1f} mm, "
                f"{sign}{self.delta_mm:.1f} mm / {sign}{self.delta_pct:.0f}%) "
                f"→ {self.change_class.value}"
            )
        return f"{self.region}: {self.change_class.value}"


@dataclass
class TemporalAnalysis:
    """Complete temporal comparison result."""
    overall_class:      ChangeClass      = ChangeClass.NoComparison
    interval_note:      str              = ""      # injected into ARLL context
    sc_adjustment:      float            = 0.0     # +0.02 stable, -0.05 progressed
    region_changes:     List[RegionChange] = field(default_factory=list)
    current_image:      str              = ""
    prior_image:        str              = ""
    pixel_rmse:         float            = 0.0     # image-level RMSE (0–255 scale)
    significant_change: bool             = False   # > Fleischner 1.5 mm threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  TemporalComparator
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalComparator:
    """
    Compare a current scan to a prior scan and classify interval change.

    Two modes:
      1. Real image mode  — uses PIL to compute pixel-level differences.
         Requires `pip install Pillow` (optional dependency).
      2. Mock / text mode — uses metadata + diagnosis context to produce
         a clinically plausible note without image files.

    Fleischner 2017 / ACR significance threshold:
      Solid nodule growth ≥ 1.5 mm in 3 months = significant.

    Sc adjustment applied to ARLL confidence gate:
      Stable:     +0.02  (prior data reduces uncertainty)
      Progressed: -0.05  (new progression raises uncertainty)
      Regressed:  +0.03  (response to treatment is reassuring)
      New:        -0.04  (new finding needs full evaluation)
      Resolved:   +0.01
    """

    # Sc adjustment table
    _SC_ADJ = {
        ChangeClass.Stable:       +0.02,
        ChangeClass.Progressed:   -0.05,
        ChangeClass.Regressed:    +0.03,
        ChangeClass.New:          -0.04,
        ChangeClass.Resolved:     +0.01,
        ChangeClass.NoComparison:  0.00,
    }

    # Fleischner growth significance threshold (mm)
    _GROWTH_THRESHOLD_MM = 1.5

    def __init__(self, growth_threshold_mm: float = 1.5) -> None:
        self._threshold = growth_threshold_mm

    def compare(
        self,
        current_path: str,
        prior_path: Optional[str],
        current_roi_size_mm: float = 0.0,
        prior_roi_size_mm: float   = 0.0,
        region_label: str          = "index lesion",
    ) -> TemporalAnalysis:
        """
        Compare current scan to prior and return a TemporalAnalysis.

        Args:
            current_path:       Path to current image.
            prior_path:         Path to prior image (None = no comparison).
            current_roi_size_mm: Measured lesion/finding size in current scan (mm).
            prior_roi_size_mm:   Measured lesion/finding size in prior scan (mm).
            region_label:        Anatomical label for the region being compared.
        """
        if not prior_path:
            return TemporalAnalysis(
                overall_class=ChangeClass.NoComparison,
                interval_note="No prior imaging available for temporal comparison.",
                sc_adjustment=0.0,
                current_image=current_path,
            )

        # Compute size-based classification
        region_change = self._classify_size_change(
            region_label, current_roi_size_mm, prior_roi_size_mm
        )

        # Try pixel-level comparison if both files exist
        rmse = 0.0
        if os.path.exists(current_path) and os.path.exists(prior_path):
            rmse = self._pixel_rmse(current_path, prior_path)
            # Augment classification with pixel-level signal
            if rmse > 20 and region_change.change_class == ChangeClass.Stable:
                region_change.change_class  = ChangeClass.Progressed
                region_change.intensity_delta = rmse

        overall = region_change.change_class
        sc_adj  = self._SC_ADJ.get(overall, 0.0)
        sig     = (abs(region_change.delta_mm) >= self._threshold
                   if region_change.delta_mm != 0.0 else rmse > 15)
        note    = self._build_note(region_change, prior_path, sig)

        return TemporalAnalysis(
            overall_class=overall,
            interval_note=note,
            sc_adjustment=sc_adj,
            region_changes=[region_change],
            current_image=current_path,
            prior_image=prior_path,
            pixel_rmse=rmse,
            significant_change=sig,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify_size_change(
        self,
        label: str,
        current_mm: float,
        prior_mm: float,
    ) -> RegionChange:
        if current_mm == 0 and prior_mm == 0:
            return RegionChange(region=label, change_class=ChangeClass.NoComparison)

        if prior_mm == 0 and current_mm > 0:
            return RegionChange(
                region=label,
                current_size_mm=current_mm,
                change_class=ChangeClass.New,
            )
        if current_mm == 0 and prior_mm > 0:
            return RegionChange(
                region=label,
                prior_size_mm=prior_mm,
                change_class=ChangeClass.Resolved,
            )

        delta     = current_mm - prior_mm
        delta_pct = delta / prior_mm * 100.0

        if abs(delta) < self._threshold:
            cls = ChangeClass.Stable
        elif delta > 0:
            cls = ChangeClass.Progressed
        else:
            cls = ChangeClass.Regressed

        return RegionChange(
            region=label,
            current_size_mm=current_mm,
            prior_size_mm=prior_mm,
            delta_mm=delta,
            delta_pct=delta_pct,
            change_class=cls,
        )

    def _pixel_rmse(self, path_a: str, path_b: str) -> float:
        """
        Compute RMSE between two images resized to 128×128 greyscale.
        Falls back to 0.0 if PIL is unavailable.
        """
        try:
            from PIL import Image  # type: ignore[import-untyped]
            SIZE = (128, 128)
            a = Image.open(path_a).convert("L").resize(SIZE)
            b = Image.open(path_b).convert("L").resize(SIZE)
            sq_diff = sum(
                (pa - pb) ** 2
                for pa, pb in zip(a.getdata(), b.getdata())
            )
            return math.sqrt(sq_diff / (SIZE[0] * SIZE[1]))
        except Exception:
            return 0.0

    def _build_note(
        self,
        rc: RegionChange,
        prior_path: str,
        significant: bool,
    ) -> str:
        prior_name = os.path.basename(prior_path)
        lines = [
            f"Temporal comparison vs prior ({prior_name}):",
            f"  {rc.to_note()}",
        ]
        if significant:
            lines.append(
                "  ⚠ Significant interval change (≥ 1.5 mm threshold met). "
                "Recommend short-interval follow-up."
            )
        else:
            lines.append(
                "  ✓ Change below Fleischner significance threshold (< 1.5 mm)."
            )
        lines.append(
            f"  Sc adjustment applied: {self._SC_ADJ[rc.change_class]:+.2f}"
        )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock temporal notes (used by mock pipeline / no prior available)
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_TEMPORAL_NOTES = {
    "stable": (
        "Comparison with prior imaging (6 months earlier): index lesion measures "
        "3.2 × 2.8 cm — unchanged from prior 3.1 × 2.7 cm (delta < 1.5 mm). "
        "No new satellite nodules. Pleural space unchanged. "
        "Classification: Stable. Sc adjustment: +0.02."
    ),
    "progressed": (
        "Comparison with prior imaging (3 months earlier): index lesion now "
        "measures 3.8 × 3.4 cm (prior 3.2 × 2.8 cm, +6 mm, +19%). "
        "New mediastinal lymphadenopathy. Classification: Progressed. "
        "Significant interval growth — urgent MDT review required. Sc adjustment: -0.05."
    ),
    "regressed": (
        "Comparison with prior imaging (post-treatment, 2 months): index lesion "
        "measures 2.4 × 2.1 cm (prior 3.2 × 2.8 cm, -8 mm, -25%). "
        "Reduced density. Classification: Regressed. "
        "Treatment response confirmed. Sc adjustment: +0.03."
    ),
    "new": (
        "Comparison with prior imaging (12 months earlier): lesion absent on "
        "prior CXR — new finding. No prior lesion at this location. "
        "Classification: New. Full workup required. Sc adjustment: -0.04."
    ),
    "no_comparison": (
        "No prior imaging available for temporal interval comparison."
    ),
}


def mock_temporal_note(has_prior: bool = False, iteration: int = 1) -> str:
    """Return a realistic mock temporal note for the given context."""
    if not has_prior:
        return MOCK_TEMPORAL_NOTES["no_comparison"]
    # Simulate progression on first iteration, stability on subsequent
    if iteration == 1:
        return MOCK_TEMPORAL_NOTES["progressed"]
    return MOCK_TEMPORAL_NOTES["stable"]
