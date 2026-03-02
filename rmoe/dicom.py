"""
rmoe/dicom.py — DICOM Preprocessing and Window/Level Normalisation.

Converts DICOM files (and raw pixel arrays) to windowed PNG slices
ready for the MPE vision model.

Windowing presets (W/L):
  Preset          WL      WW     Clinical use
  ─────────────────────────────────────────────────────────────────────
  Lung           -600    1500   Lung parenchyma, nodule detection
  Mediastinum      40     400   Soft tissue, mediastinal masses
  Bone            400    1800   Cortical bone, fractures
  Brain            40      80   Grey/white matter, haemorrhage
  Brain stroke     35      35   Subtle early ischaemic change (ASPECTS)
  Liver            60     150   Hepatic lesions
  Soft Tissue      60     400   Abdominal soft tissue
  Spine bone      400    1000   Vertebral body, pedicles
  Pet / SUV       n/a    n/a    PET SUV-max scale (0–10)

Usage:
  # With pydicom installed (real DICOM):
  from rmoe.dicom import DICOMProcessor
  proc = DICOMProcessor()
  png_path = proc.dicom_to_png("slice.dcm", window="Lung")

  # From raw HU array (numpy / list-of-lists):
  png_path = proc.array_to_png(hu_array, window="Brain", out_path="brain.png")

  # Without pydicom — returns original path unchanged:
  png_path = proc.dicom_to_png("image.png", window="Lung")  # no-op for PNG
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import pydicom  # type: ignore[import-untyped]
    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False

try:
    from PIL import Image  # type: ignore[import-untyped]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Window presets
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WindowPreset:
    name:   str
    level:  float   # window centre (HU)
    width:  float   # window width  (HU)

    @property
    def lower(self) -> float:
        return self.level - self.width / 2.0

    @property
    def upper(self) -> float:
        return self.level + self.width / 2.0


WINDOW_PRESETS: Dict[str, WindowPreset] = {
    "lung":         WindowPreset("lung",        level=-600, width=1500),
    "mediastinum":  WindowPreset("mediastinum", level= 40,  width= 400),
    "bone":         WindowPreset("bone",        level= 400, width=1800),
    "spine_bone":   WindowPreset("spine_bone",  level= 400, width=1000),
    "brain":        WindowPreset("brain",       level=  40, width=  80),
    "brain_stroke": WindowPreset("brain_stroke",level=  35, width=  35),
    "liver":        WindowPreset("liver",       level=  60, width= 150),
    "soft_tissue":  WindowPreset("soft_tissue", level=  60, width= 400),
    "default":      WindowPreset("default",     level=  40, width= 400),
}


def get_window(name: str) -> WindowPreset:
    """Return a WindowPreset by name (case-insensitive). Falls back to 'default'."""
    return WINDOW_PRESETS.get(name.lower().replace(" ", "_"),
                               WINDOW_PRESETS["default"])


# ═══════════════════════════════════════════════════════════════════════════════
#  DICOMProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class DICOMProcessor:
    """
    Load DICOM slices, apply windowing, and export as PNG.

    Graceful degradation:
      • pydicom + PIL available → full DICOM decode + windowing + PNG save
      • PIL only               → can window numpy/list arrays, save PNG
      • Neither available      → returns original path unchanged (no-op)
    """

    def __init__(
        self,
        output_dir: str = "/tmp/rmoe_dicom",
        default_window: str = "default",
    ) -> None:
        self._output_dir     = output_dir
        self._default_window = default_window

    # ── Public API ────────────────────────────────────────────────────────────

    def dicom_to_png(
        self,
        dicom_path: str,
        window:     str = "",
        out_path:   Optional[str] = None,
    ) -> str:
        """
        Convert a DICOM file to a windowed PNG.

        Args:
            dicom_path: Path to .dcm file (or any image file).
            window:     Window preset name ('' = auto-detect from DICOM metadata).
            out_path:   Output PNG path (None = auto-generated in output_dir).

        Returns:
            Path to output PNG (or original path if conversion impossible).
        """
        # If not a DICOM file, return as-is
        if not self._is_dicom(dicom_path):
            return dicom_path

        if not _PYDICOM_AVAILABLE:
            return dicom_path

        try:
            return self._convert_dicom(dicom_path, window, out_path)
        except Exception as exc:
            import sys
            print(f"[dicom] conversion failed for '{dicom_path}': {exc}",
                  file=sys.stderr)
            return dicom_path

    def array_to_png(
        self,
        hu_array,                    # 2D list-of-lists or numpy array
        window: str        = "default",
        out_path: str      = "/tmp/rmoe_dicom/windowed.png",
    ) -> str:
        """
        Apply windowing to a raw HU pixel array and save as PNG.

        Args:
            hu_array: 2D array of Hounsfield units.
            window:   Window preset name.
            out_path: Output PNG path.

        Returns:
            Path to saved PNG, or '' on failure.
        """
        if not _PIL_AVAILABLE:
            return ""

        try:
            preset = get_window(window)
            windowed = self._apply_window_array(hu_array, preset)
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            img = Image.fromarray(windowed, mode="L")
            img.save(out_path)
            return out_path
        except Exception as exc:
            import sys
            print(f"[dicom] array_to_png failed: {exc}", file=sys.stderr)
            return ""

    def get_modality(self, dicom_path: str) -> str:
        """
        Return imaging modality string from DICOM header.
        Returns 'Unknown' if pydicom unavailable or tag missing.
        """
        if not _PYDICOM_AVAILABLE or not self._is_dicom(dicom_path):
            return "Unknown"
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            return str(getattr(ds, "Modality", "Unknown"))
        except Exception:
            return "Unknown"

    def auto_window(self, dicom_path: str) -> str:
        """
        Return the most appropriate window preset name for a DICOM file
        based on its Modality and BodyPartExamined tags.
        """
        if not _PYDICOM_AVAILABLE:
            return self._default_window
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", "")).upper()
            body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
            return self._infer_window(modality, body_part)
        except Exception:
            return self._default_window

    def dicom_metadata(self, dicom_path: str) -> Dict[str, str]:
        """Return a dict of key DICOM metadata fields."""
        if not _PYDICOM_AVAILABLE or not self._is_dicom(dicom_path):
            return {}
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            fields = [
                "PatientID", "StudyDate", "Modality",
                "BodyPartExamined", "SliceThickness",
                "PixelSpacing", "Rows", "Columns",
                "WindowCenter", "WindowWidth",
                "RescaleIntercept", "RescaleSlope",
            ]
            return {
                f: str(getattr(ds, f, ""))
                for f in fields
                if hasattr(ds, f)
            }
        except Exception:
            return {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _convert_dicom(
        self,
        dicom_path: str,
        window:     str,
        out_path:   Optional[str],
    ) -> str:
        ds      = pydicom.dcmread(dicom_path)
        pixels  = ds.pixel_array.astype(float)

        # Apply rescale slope / intercept → Hounsfield units
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slope     = float(getattr(ds, "RescaleSlope",     1))
        hu        = pixels * slope + intercept

        # Pick window
        if not window:
            modality   = str(getattr(ds, "Modality",          "")).upper()
            body_part  = str(getattr(ds, "BodyPartExamined",  "")).upper()
            window     = self._infer_window(modality, body_part)

            # Prefer DICOM-embedded WC/WW if available
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth",  None)
            if wc is not None and ww is not None:
                wc = float(wc[0]) if hasattr(wc, "__len__") else float(wc)
                ww = float(ww[0]) if hasattr(ww, "__len__") else float(ww)
                preset = WindowPreset("embedded", level=wc, width=ww)
            else:
                preset = get_window(window)
        else:
            preset = get_window(window)

        windowed = self._apply_window_array(hu, preset)

        # Generate output path
        if out_path is None:
            os.makedirs(self._output_dir, exist_ok=True)
            base     = os.path.splitext(os.path.basename(dicom_path))[0]
            out_path = os.path.join(
                self._output_dir, f"{base}_{preset.name}.png"
            )

        if _PIL_AVAILABLE:
            img = Image.fromarray(windowed, mode="L")
            img.save(out_path)
        else:
            # Minimal PPM fallback (no PIL)
            h, w = windowed.shape if hasattr(windowed, "shape") else \
                   (len(windowed), len(windowed[0]))
            out_path = out_path.replace(".png", ".pgm")
            with open(out_path, "wb") as fh:
                fh.write(f"P5\n{w} {h}\n255\n".encode())
                for row in windowed:
                    fh.write(bytes(int(v) for v in row))

        return out_path

    @staticmethod
    def _apply_window_array(hu, preset: WindowPreset):
        """
        Apply window/level to a 2D HU array → 0–255 uint8.
        Works with nested lists or numpy arrays.
        """
        lower = preset.lower
        upper = preset.upper
        span  = upper - lower or 1.0

        # Try numpy path
        try:
            import numpy as np  # type: ignore[import-untyped]
            arr = np.array(hu, dtype=float)
            arr = np.clip(arr, lower, upper)
            arr = ((arr - lower) / span * 255.0).astype(np.uint8)
            return arr
        except ImportError:
            pass

        # Pure Python fallback (slow but dependency-free)
        result = []
        for row in hu:
            new_row = []
            for val in row:
                v = max(lower, min(upper, float(val)))
                new_row.append(int((v - lower) / span * 255))
            result.append(new_row)
        return result

    @staticmethod
    def _is_dicom(path: str) -> bool:
        """Return True if the file appears to be a DICOM file."""
        if not os.path.exists(path):
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext in (".dcm", ".dicom", ""):
            # Check DICOM magic bytes: offset 128 = "DICM"
            try:
                with open(path, "rb") as fh:
                    fh.seek(128)
                    return fh.read(4) == b"DICM"
            except OSError:
                return False
        return False

    @staticmethod
    def _infer_window(modality: str, body_part: str) -> str:
        """Infer best window preset from DICOM modality + body part tags."""
        if modality in ("CR", "DR", "DX"):
            return "mediastinum"  # chest X-ray
        if modality == "MR":
            if "BRAIN" in body_part or "HEAD" in body_part:
                return "brain"
            return "soft_tissue"
        if modality == "CT":
            if "CHEST" in body_part or "THORAX" in body_part:
                return "lung"
            if "BRAIN" in body_part or "HEAD" in body_part:
                return "brain"
            if "SPINE" in body_part or "VERTEBRA" in body_part:
                return "spine_bone"
            if "ABDOMEN" in body_part or "LIVER" in body_part:
                return "liver"
            return "soft_tissue"
        if modality in ("NM", "PT"):
            return "soft_tissue"
        return "default"
