"""
rmoe/saliency.py — Saliency-Aware Cropping and Visual Attention Processing.

Implements the MPE "Saliency-Aware Cropping" capability described in paper §2.1:
  "Focus attention on anatomically critical regions identified via
   gradient-based saliency maps; provide crop coordinates for ARLL
   feedback loops."

SaliencyProcessor:
  • Parses bounding-box strings from MPE output ("x1,y1,x2,y2")
  • Crops the specified region from the source image
  • Resizes the crop to the full original image dimensions (zoom effect)
  • Optionally overlays a saliency heatmap (red channel boost on crop region)
  • Saves the processed crop to a temp file for the next iteration
  • Falls back gracefully when PIL is unavailable (returns original path)

CropCoordinates:
  • Parsed from MPE saliency_crop field
  • Normalised to image dimensions (pixel coords)

AttentionMap:
  • Lightweight 2D grid of attention weights
  • Used to compute a "top-k crop" suggestion independent of model output
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CropCoordinates:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()

    def is_valid(self) -> bool:
        return self.width() > 0 and self.height() > 0

    def scale(self, factor: float) -> "CropCoordinates":
        """Scale coordinates around the crop centre by factor."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        hw = self.width()  / 2
        hh = self.height() / 2
        return CropCoordinates(
            x1=max(0, int(cx - hw * factor)),
            y1=max(0, int(cy - hh * factor)),
            x2=int(cx + hw * factor),
            y2=int(cy + hh * factor),
        )

    def clamp(self, img_w: int, img_h: int) -> "CropCoordinates":
        return CropCoordinates(
            x1=max(0, min(self.x1, img_w - 1)),
            y1=max(0, min(self.y1, img_h - 1)),
            x2=max(0, min(self.x2, img_w)),
            y2=max(0, min(self.y2, img_h)),
        )

    def to_string(self) -> str:
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    @classmethod
    def from_string(cls, s: str) -> "CropCoordinates":
        """Parse 'x1,y1,x2,y2' bounding box string."""
        parts = re.findall(r"\d+", s)
        if len(parts) >= 4:
            return cls(*[int(p) for p in parts[:4]])
        return cls()

    @classmethod
    def from_region_label(
        cls, label: str, img_w: int = 512, img_h: int = 512
    ) -> "CropCoordinates":
        """
        Derive approximate bounding box from an anatomical region label.
        Used when the doctor provides a zoom command like 'focus on LUL'.
        """
        label = label.lower()

        # Chest zones (approximate quadrant mapping on 512×512 PA CXR)
        mapping = {
            "left upper":  CropCoordinates(0,   0,   256, 200),
            "right upper": CropCoordinates(256, 0,   512, 200),
            "left lower":  CropCoordinates(0,   300, 256, 512),
            "right lower": CropCoordinates(256, 300, 512, 512),
            "left lung":   CropCoordinates(0,   0,   256, 512),
            "right lung":  CropCoordinates(256, 0,   512, 512),
            "mediastinum": CropCoordinates(180, 0,   330, 400),
            "cardiac":     CropCoordinates(160, 200, 370, 450),
            "upper lobe":  CropCoordinates(0,   0,   512, 200),
            "lower lobe":  CropCoordinates(0,   300, 512, 512),
            "liver":       CropCoordinates(260, 300, 512, 512),
            "spine":       CropCoordinates(200, 0,   310, 512),
            "hip":         CropCoordinates(100, 300, 420, 512),
            "brain":       CropCoordinates(50,  30,  460, 480),
        }
        for key, box in mapping.items():
            if key in label:
                return box.clamp(img_w, img_h)

        # Fallback: centre crop
        margin_x = img_w // 4
        margin_y = img_h // 4
        return CropCoordinates(
            margin_x, margin_y, img_w - margin_x, img_h - margin_y
        )


@dataclass
class CropResult:
    """Result of a saliency crop operation."""
    output_path: str         = ""    # path to cropped+resized image
    coordinates: CropCoordinates = field(default_factory=CropCoordinates)
    zoom_factor: float       = 1.0
    original_size: Tuple[int, int] = (0, 0)
    crop_size: Tuple[int, int]     = (0, 0)
    method: str              = "unknown"   # "pil" | "fallback"
    note: str                = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SaliencyProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class SaliencyProcessor:
    """
    Crop and zoom anatomical sub-regions for the MPE feedback loop.

    The processor takes the source image + bounding box coordinates from
    MPE Phase 1 output, crops the specified region, resizes it to full
    image dimensions to simulate higher effective resolution, and optionally
    marks the crop with a red-channel boost for visual audit.

    When PIL is unavailable, it returns the original image path unchanged
    and notes the fallback so the pipeline continues uninterrupted.
    """

    def __init__(
        self,
        output_dir: str = "/tmp/rmoe_crops",
        mark_region: bool = False,
        jpeg_quality: int = 95,
    ) -> None:
        """
        Args:
            output_dir:    Directory for cropped output images.
            mark_region:   Draw a red border around the crop on the zoomed output.
            jpeg_quality:  JPEG save quality (1–100).
        """
        self._output_dir    = output_dir
        self._mark_region   = mark_region
        self._jpeg_quality  = jpeg_quality
        self._pil_available = self._check_pil()

    @staticmethod
    def _check_pil() -> bool:
        try:
            import PIL  # noqa: F401
            return True
        except ImportError:
            return False

    def crop_and_zoom(
        self,
        source_path: str,
        crop_coords: CropCoordinates,
        zoom_factor: float = 2.5,
        label: str = "crop",
    ) -> CropResult:
        """
        Crop the specified region and resize to full image dimensions.

        Args:
            source_path:  Input image path.
            crop_coords:  Bounding box to crop.
            zoom_factor:  Not used for resize (we always fill full frame);
                          recorded in CropResult for audit.
            label:        Label for output filename.
        Returns:
            CropResult with path to processed image.
        """
        if not crop_coords.is_valid():
            return CropResult(
                output_path=source_path,
                coordinates=crop_coords,
                zoom_factor=zoom_factor,
                method="fallback",
                note="Invalid crop coordinates — using original image.",
            )

        if not self._pil_available:
            return CropResult(
                output_path=source_path,
                coordinates=crop_coords,
                zoom_factor=zoom_factor,
                method="fallback",
                note="PIL unavailable — install Pillow for real cropping.",
            )

        return self._pil_crop(source_path, crop_coords, zoom_factor, label)

    def crop_from_feedback(
        self,
        source_path: str,
        feedback_payload: str,
        zoom: float = 2.5,
    ) -> CropResult:
        """
        Crop from a #wanna# feedback payload string.
        Payload format: "region=<name>;zoom=<factor>"  or  "x1,y1,x2,y2"
        """
        # Try to parse as x1,y1,x2,y2
        nums = re.findall(r"\d+", feedback_payload.split(";")[0])
        if len(nums) >= 4:
            coords = CropCoordinates.from_string(",".join(nums[:4]))
        else:
            # Parse region name and zoom
            region_match = re.search(r"region=([^;]+)", feedback_payload)
            zoom_match   = re.search(r"zoom=([\d.]+)", feedback_payload)
            region = region_match.group(1).strip() if region_match else feedback_payload
            zoom   = float(zoom_match.group(1))    if zoom_match   else zoom
            coords = CropCoordinates.from_region_label(region)

        return self.crop_and_zoom(source_path, coords, zoom_factor=zoom)

    def saliency_crop_from_string(
        self,
        source_path: str,
        crop_string: str,
    ) -> CropResult:
        """Convenience wrapper: parse 'x1,y1,x2,y2' from MPE output."""
        coords = CropCoordinates.from_string(crop_string)
        return self.crop_and_zoom(source_path, coords)

    # ── PIL implementation ─────────────────────────────────────────────────────

    def _pil_crop(
        self,
        source_path: str,
        coords: CropCoordinates,
        zoom_factor: float,
        label: str,
    ) -> CropResult:
        try:
            from PIL import Image, ImageDraw  # type: ignore[import-untyped]

            os.makedirs(self._output_dir, exist_ok=True)
            img = Image.open(source_path).convert("RGB")
            orig_w, orig_h = img.size

            clamped = coords.clamp(orig_w, orig_h)
            if not clamped.is_valid():
                return CropResult(
                    output_path=source_path, coordinates=coords,
                    zoom_factor=zoom_factor, original_size=(orig_w, orig_h),
                    method="fallback", note="Clamped coords invalid.",
                )

            crop_box = (clamped.x1, clamped.y1, clamped.x2, clamped.y2)
            cropped  = img.crop(crop_box)
            # Resize back to original dimensions → apparent zoom
            zoomed   = cropped.resize((orig_w, orig_h), Image.LANCZOS)

            if self._mark_region:
                draw = ImageDraw.Draw(zoomed)
                bw   = max(3, orig_w // 80)
                draw.rectangle([0, 0, orig_w - 1, orig_h - 1],
                               outline=(255, 50, 50), width=bw)

            # Save
            base = os.path.splitext(os.path.basename(source_path))[0]
            out_name = f"{base}_{label}_{clamped.to_string()}.png"
            out_path = os.path.join(self._output_dir, out_name)
            zoomed.save(out_path, "PNG")

            return CropResult(
                output_path=out_path,
                coordinates=clamped,
                zoom_factor=zoom_factor,
                original_size=(orig_w, orig_h),
                crop_size=(clamped.width(), clamped.height()),
                method="pil",
                note=f"Cropped {clamped.to_string()} → resized to {orig_w}×{orig_h}",
            )
        except Exception as exc:
            return CropResult(
                output_path=source_path, coordinates=coords,
                zoom_factor=zoom_factor, method="fallback",
                note=f"PIL crop failed: {exc}",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  AttentionMap — lightweight gradient-free attention approximation
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionMap:
    """
    Compute a coarse spatial attention map from image pixel statistics.
    Used to generate saliency crop suggestions when no model gradient is
    available (pure Python, no PyTorch/numpy required).

    Method: variance-based saliency — blocks with high local variance are
    more likely to contain diagnostically relevant structures.
    """

    def __init__(self, grid_size: int = 8) -> None:
        """
        Args:
            grid_size: Divide the image into grid_size × grid_size blocks.
        """
        self._grid = grid_size

    def compute_top_crop(
        self,
        image_path: str,
        top_k: int = 1,
    ) -> List[CropCoordinates]:
        """
        Return top_k highest-variance regions as CropCoordinates.
        Falls back to centre crop if PIL unavailable.
        """
        if not os.path.exists(image_path):
            return [CropCoordinates(64, 64, 448, 448)]

        try:
            from PIL import Image  # type: ignore[import-untyped]
            img = Image.open(image_path).convert("L")
            w, h = img.size
            bw = w // self._grid
            bh = h // self._grid

            variances: List[Tuple[float, CropCoordinates]] = []
            for gi in range(self._grid):
                for gj in range(self._grid):
                    x1 = gi * bw
                    y1 = gj * bh
                    x2 = min(w, x1 + bw)
                    y2 = min(h, y1 + bh)
                    block = img.crop((x1, y1, x2, y2))
                    pixels = list(block.getdata())
                    if not pixels:
                        continue
                    mu  = sum(pixels) / len(pixels)
                    var = sum((p - mu) ** 2 for p in pixels) / len(pixels)
                    variances.append((var, CropCoordinates(x1, y1, x2, y2)))

            variances.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in variances[:top_k]]
        except Exception:
            # Centre-crop fallback
            return [CropCoordinates(0, 0, 512, 512)]
