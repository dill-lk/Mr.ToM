"""
rmoe/mcv.py — Multi-Modal Contextual Vectors (MCV) for inter-agent transfer.

Implements paper §"Technical and Operational Challenges":
  "Researchers are exploring the use of Multi-Modal Contextual Vectors (MCV)
   that preserve the last hidden states of the perception model across
   multiple layers, serving as latent instructions for the reasoning agent."

Why MCV matters:
  When the MPE sends its output to the ARLL, critical spatial / intensity
  information can be suppressed during text tokenisation. MCV preserves this
  perceptual signal in a structured form so ARLL receives richer context
  than raw text alone.

This module provides:
  MCVBuilder  — constructs an MCV from MPE PerceptionEvidence output.
  MCVInjector — serialises an MCV into a compact context-injection string
                that is prepended to the ARLL system prompt.

MCV structure (all fields are runtime-computable without model internals):
  spatial_features   — normalised bounding-box attention weights per region
  intensity_profile  — mean/std pixel intensity per region (0–1 normalised)
  modality_tokens    — one-hot modality encoding [CXR, CT, MRI, US, PET]
  temporal_delta     — interval change vector from TemporalComparator
  perception_conf    — MPE per-region confidence scores
  token_budget_used  — how many visual tokens the MPE consumed (0–1)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

MODALITIES = ["CXR", "CT", "MRI", "US", "PET", "OTHER"]


@dataclass
class SpatialFeature:
    """Spatial attention weight for one anatomical region."""
    region:      str
    x1_norm:     float = 0.0   # bounding box normalised to [0, 1]
    y1_norm:     float = 0.0
    x2_norm:     float = 1.0
    y2_norm:     float = 1.0
    attention:   float = 0.0   # 0 = ignored, 1 = high focus
    confidence:  float = 0.0   # MPE per-region confidence


@dataclass
class MCVTensor:
    """
    Multi-Modal Contextual Vector — compact latent representation of
    MPE perception output for injection into ARLL context.
    """
    # Spatial attention per detected region
    spatial_features:   List[SpatialFeature] = field(default_factory=list)

    # Intensity statistics per region {region: (mean, std)} normalised 0–1
    intensity_profile:  Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Modality one-hot [CXR, CT, MRI, US, PET, OTHER]
    modality_tokens:    List[int]  = field(default_factory=lambda: [0]*6)

    # Temporal delta vector [stability_class_idx, delta_pct_norm, sc_adj]
    temporal_delta:     List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Saliency crop coordinates (normalised 0–1)
    primary_crop:       Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    # Visual token budget (0–1, 1 = full budget used)
    token_budget_used:  float = 0.5

    # Global MPE confidence
    perception_conf:    float = 0.0

    # Raw MPE text evidence (preserved for ARLL)
    evidence_text:      str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  MCVBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class MCVBuilder:
    """
    Build an MCVTensor from MPE PerceptionEvidence output.

    Used in core.py after Phase 1 completes and before Phase 2 starts.
    """

    # Temporal change-class → delta index (for temporal_delta vector)
    _TEMP_CLASS_IDX = {
        "NoComparison": 0,
        "Stable":       1,
        "Progressed":   2,
        "Regressed":    3,
        "New":          4,
        "Resolved":     5,
    }

    def build(
        self,
        evidence,                        # PerceptionEvidence dataclass
        modality:         str = "CXR",
        temporal_analysis = None,        # TemporalAnalysis | None
        image_width:      int = 512,
        image_height:     int = 512,
    ) -> MCVTensor:
        """
        Construct an MCV from MPE output.

        Args:
            evidence:          PerceptionEvidence (rmoe.models)
            modality:          Imaging modality string
            temporal_analysis: TemporalAnalysis from rmoe.temporal (optional)
            image_width:       Source image width for coordinate normalisation
            image_height:      Source image height for coordinate normalisation
        """
        mcv = MCVTensor()

        # ── Spatial features ──────────────────────────────────────────────────
        mcv.perception_conf  = getattr(evidence, "confidence", 0.0)
        mcv.evidence_text    = getattr(evidence, "feature_summary", "")
        crop_str             = getattr(evidence, "saliency_crop", "")
        mcv.primary_crop     = self._parse_crop(crop_str, image_width, image_height)
        mcv.token_budget_used = self._estimate_token_budget(mcv.evidence_text)

        # Build spatial feature from saliency crop
        x1n, y1n, x2n, y2n = mcv.primary_crop
        mcv.spatial_features.append(SpatialFeature(
            region="primary_roi",
            x1_norm=x1n, y1_norm=y1n,
            x2_norm=x2n, y2_norm=y2n,
            attention=min(1.0, mcv.perception_conf + 0.1),
            confidence=mcv.perception_conf,
        ))

        # Add finding-level regions from evidence text
        mcv.spatial_features.extend(
            self._extract_region_features(mcv.evidence_text)
        )

        # ── Modality one-hot ──────────────────────────────────────────────────
        mod_upper = modality.upper()
        mcv.modality_tokens = [
            1 if m == mod_upper else 0 for m in MODALITIES
        ]

        # ── Temporal delta ────────────────────────────────────────────────────
        if temporal_analysis is not None:
            cls_name = temporal_analysis.overall_class.value
            cls_idx  = self._TEMP_CLASS_IDX.get(cls_name, 0)
            delta_pct = 0.0
            if temporal_analysis.region_changes:
                delta_pct = temporal_analysis.region_changes[0].delta_pct / 100.0
            sc_adj   = temporal_analysis.sc_adjustment
            mcv.temporal_delta = [float(cls_idx) / 5.0, delta_pct, sc_adj]

        # ── Intensity profile (mock / text-derived) ───────────────────────────
        mcv.intensity_profile = self._derive_intensity_profile(mcv.evidence_text)

        return mcv

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_crop(
        crop_str: str,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float, float]:
        import re
        nums = re.findall(r"\d+", crop_str)
        if len(nums) >= 4:
            x1, y1, x2, y2 = [int(n) for n in nums[:4]]
            return (
                x1 / img_w, y1 / img_h,
                min(1.0, x2 / img_w), min(1.0, y2 / img_h),
            )
        return (0.0, 0.0, 1.0, 1.0)

    @staticmethod
    def _estimate_token_budget(text: str) -> float:
        """Approximate visual token budget used (0–1)."""
        words = len(text.split())
        return min(1.0, words / 200.0)

    @staticmethod
    def _extract_region_features(text: str) -> List[SpatialFeature]:
        """Heuristically extract anatomical region mentions from MPE text."""
        features: List[SpatialFeature] = []
        region_keywords = {
            "left upper lobe":  (0.0, 0.0, 0.5, 0.4),
            "right upper lobe": (0.5, 0.0, 1.0, 0.4),
            "left lower lobe":  (0.0, 0.6, 0.5, 1.0),
            "right lower lobe": (0.5, 0.6, 1.0, 1.0),
            "mediastinum":      (0.35, 0.0, 0.65, 0.8),
            "cardiac":          (0.3, 0.4, 0.7, 0.9),
            "pleural":          (0.0, 0.5, 1.0, 1.0),
            "hilum":            (0.35, 0.3, 0.65, 0.6),
        }
        text_l = text.lower()
        for keyword, (x1, y1, x2, y2) in region_keywords.items():
            if keyword in text_l:
                features.append(SpatialFeature(
                    region=keyword,
                    x1_norm=x1, y1_norm=y1,
                    x2_norm=x2, y2_norm=y2,
                    attention=0.6,
                    confidence=0.7,
                ))
        return features

    @staticmethod
    def _derive_intensity_profile(text: str) -> Dict[str, Tuple[float, float]]:
        """Derive rough intensity characteristics from MPE description."""
        profile: Dict[str, Tuple[float, float]] = {}
        text_l = text.lower()
        # Common radiological density descriptors → approximate HU mapping
        density_map = {
            "hyperechoic":     (0.85, 0.05),
            "hypoechoic":      (0.25, 0.05),
            "hyperdense":      (0.80, 0.08),
            "hypodense":       (0.20, 0.08),
            "ground glass":    (0.45, 0.10),
            "consolidation":   (0.70, 0.10),
            "opacity":         (0.65, 0.12),
            "lucency":         (0.15, 0.05),
        }
        for descriptor, (mean, std) in density_map.items():
            if descriptor in text_l:
                profile[descriptor] = (mean, std)
        return profile


# ═══════════════════════════════════════════════════════════════════════════════
#  MCVInjector — serialise MCV into ARLL prompt context
# ═══════════════════════════════════════════════════════════════════════════════

class MCVInjector:
    """
    Serialise an MCVTensor into a compact context-injection string for ARLL.

    The injected block is placed at the TOP of the ARLL system prompt, before
    the MPE evidence text, so the reasoning model receives full perceptual
    grounding from the start.
    """

    def inject(self, mcv: MCVTensor, modality: str = "CXR") -> str:
        """Return MCV context block as a formatted string."""
        lines = [
            "=== MULTI-MODAL CONTEXTUAL VECTOR (MCV) ===",
            f"Modality      : {modality}",
            f"Perception Sc : {mcv.perception_conf:.4f}",
            f"Token Budget  : {mcv.token_budget_used:.2f} (1.0 = full Naive Dynamic Resolution)",
            "",
        ]

        # Spatial features
        lines.append("Spatial Attention Map:")
        for sf in mcv.spatial_features[:5]:  # top 5 regions
            lines.append(
                f"  [{sf.region}] bbox=({sf.x1_norm:.2f},{sf.y1_norm:.2f},"
                f"{sf.x2_norm:.2f},{sf.y2_norm:.2f})  "
                f"attn={sf.attention:.2f}  conf={sf.confidence:.2f}"
            )

        # Intensity profile
        if mcv.intensity_profile:
            lines.append("\nDensity / Intensity Descriptors:")
            for desc, (mean, std) in list(mcv.intensity_profile.items())[:4]:
                lines.append(f"  {desc}: mean={mean:.2f}  std={std:.2f}")

        # Temporal delta
        cls_labels = ["NoComparison", "Stable", "Progressed", "Regressed", "New", "Resolved"]
        td = mcv.temporal_delta
        cls_idx = min(5, int(round(td[0] * 5)))
        lines.append(
            f"\nTemporal Delta : class={cls_labels[cls_idx]}"
            f"  Δ%={td[1]*100:.1f}  Sc_adj={td[2]:+.3f}"
        )

        # Primary saliency crop
        c = mcv.primary_crop
        lines.append(
            f"Primary Crop   : ({c[0]:.2f},{c[1]:.2f}) → ({c[2]:.2f},{c[3]:.2f})"
        )

        lines.append("=== END MCV ===\n")
        return "\n".join(lines)

    def to_json(self, mcv: MCVTensor) -> str:
        """Serialise MCV to JSON for audit logging."""
        return json.dumps({
            "spatial_features": [
                {"region": s.region, "bbox": [s.x1_norm, s.y1_norm, s.x2_norm, s.y2_norm],
                 "attention": round(s.attention, 3), "confidence": round(s.confidence, 3)}
                for s in mcv.spatial_features
            ],
            "intensity_profile": {
                k: {"mean": round(v[0], 3), "std": round(v[1], 3)}
                for k, v in mcv.intensity_profile.items()
            },
            "modality_tokens": mcv.modality_tokens,
            "temporal_delta":  [round(v, 4) for v in mcv.temporal_delta],
            "primary_crop":    [round(v, 4) for v in mcv.primary_crop],
            "token_budget_used": round(mcv.token_budget_used, 3),
            "perception_conf": round(mcv.perception_conf, 4),
        }, indent=2)
