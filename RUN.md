# R-MoE v2.0 — Colab Run Guide

> **Recursive Multi-Agent Mixture-of-Experts for Autonomous Clinical Diagnostics**
> Paper benchmarks: F1 = 0.92 · ECE = 0.08 · 25% fewer false positives vs GPT-4V

---

## 🚀 Quick Start — run all cells top to bottom

**No Google Drive account needed.** Cell 1 downloads every model file automatically
from the public shared folder:

> 📂 https://drive.google.com/drive/folders/1NbTL4BFFrySVmFt05wEh-B1q3mqLE3C5

Files are saved to `/content/models/` on the Colab instance SSD.

---

## 🤖 Recommended Models for T4 GPU (16 GB VRAM)

> All models run **one at a time** via `ExpertSwapper` — peak VRAM usage < 4 GB per phase.

| Role | Model | Filename | Size |
|------|-------|----------|------|
| **Vision MPE** | Moondream2 2B int8 | `vision_text.gguf` | ~2.5 GB |
| **Vision CLIP** | Moondream2 mmproj | `vision_proj.gguf` | ~400 MB |
| **Reasoning ARLL** | DeepSeek-R1-Distill-Qwen-1.5B Q8 | `reasoning_expert.gguf` | ~1.8 GB |
| **Clinical CSR** | MedGemma-2B-it Q8 | `clinical_expert.gguf` | ~2.2 GB |

> **Alternatives (if above are unavailable):**
> - Reasoning: `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` (~5 GB, higher quality)
> - Clinical: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` + radiology fine-tune
> - Vision: `Qwen2-VL-2B-Instruct-Q8_0.gguf` (requires `vision_proj` companion)

---

## 🚀 Cell-by-Cell Colab Instructions

### Cell 1 — Download Model Files (from public shared Drive folder)
```python
import os

# Install gdown (already available on Colab, pinned here for local runs)
!pip install gdown --quiet

import gdown

MODELS_DIR = "/content/models"
FOLDER_URL = "https://drive.google.com/drive/folders/1NbTL4BFFrySVmFt05wEh-B1q3mqLE3C5"

os.makedirs(MODELS_DIR, exist_ok=True)
print("📥 Downloading model files from public shared Drive folder…")
print("   This may take several minutes depending on your connection speed.\n")

gdown.download_folder(FOLDER_URL, output=MODELS_DIR, quiet=False, use_cookies=False)
```

### Cell 2 — Clone Repository
```python
!git clone --depth=1 https://github.com/dill-lk/Mr.ToM.git /content/Mr.ToM
%cd /content/Mr.ToM
```

### Cell 3 — Install llama-cpp-python (CUDA wheel, ~60 s)

> ⚡ **Why not `CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python`?**
> Compiling from source on Colab free-tier takes **15–20 minutes** and frequently
> hits session rate-limits before finishing.  We use the official **pre-built CUDA
> wheel** instead — same GGML/CUDA kernels, installed in < 60 seconds, zero
> recompilation. The C++ implementation is preserved in `legacy/` for academic
> reproducibility.

```python
# Install pre-built CUDA 12.1 wheel (fastest — no compilation)
!pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --upgrade --quiet

# Verify GPU is visible
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader'], capture_output=True, text=True)
print("GPU:", result.stdout.strip())
```

### Cell 4 — Verify Models & Set Up Paths
```python
import sys, os

MODELS_DIR = "/content/models"
REPO_DIR   = "/content/Mr.ToM"

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

_expected = [
    "vision_text.gguf",
    "vision_proj.gguf",
    "reasoning_expert.gguf",
    "clinical_expert.gguf",
]
_missing = [f for f in _expected if not os.path.exists(os.path.join(MODELS_DIR, f))]
if _missing:
    print("⚠️  Missing model files:", _missing)
    print("   Re-run Cell 1 to download them.")
else:
    print("✅ All model files verified at", MODELS_DIR)
```

### Cell 5 — Upload Patient Image

> **New in v2.0** — The pipeline now asks for the patient scan before it starts.
> Run this cell to upload any PNG, JPEG, or DICOM file directly from your computer.
> The image is saved to `/content/models/` and passed automatically to the engine.
> Skip this cell if you want to use the sample `test_patient.png` downloaded in Cell 1.

> **Alternatively**, to use the included sample scan, skip this cell and set in Cell 6:
> ```python
> image_path = "/content/models/test_patient.png"
> ```

```python
import sys
sys.path.insert(0, '/content/Mr.ToM')
from image_handler import upload_patient_image

# Opens a file-upload button in the cell output.
# Click "Choose Files", select your X-ray / CT / DICOM scan, then wait for
# the ✅ confirmation before running the next cell.
image_path = upload_patient_image()
print("Image ready:", image_path)
```

---

### Cell 6 — Run R-MoE (full demo with all charts)
```python
from colab_runner import run_python_engine

run_python_engine(
    image        = image_path,   # set by Cell 5 (or hard-code a path)
    temperature  = 0.2,          # clinical precision temperature
    n_predict    = 512,
    n_gpu_layers = -1,            # -1 = full GPU offload (T4)
    audit_log    = "audit_trail.json",
    session_report = "session_report.txt",
    eval_mode    = True,          # show ECE calibration + benchmarks
    charts       = True,          # ASCII Sc/DDx/uncertainty charts
    hitl         = "auto",        # prompt doctor when Sc < 0.90
    chat_target  = "auto",        # auto-route Q&A to right expert
)
```

### Cell 7 — Interactive Doctor Q&A (post-diagnosis)
```python
from colab_runner import quick_interactive
quick_interactive()

# In the prompt, try commands like:
#   "Show me the fracture site"           → MPE zooms on that region
#   "What is the probability of cancer?"  → routed to ARLL (reasoning)
#   "What treatment does this patient need?" → routed to CSR (clinical)
#   "switch clinical"                     → force clinical expert
#   "exit"                                → end session
```

### Cell 8 — Benchmark comparison only (no models needed)
```python
from colab_runner import quick_benchmark
quick_benchmark()
```

### Cell 9 — Read audit trail
```python
import json
with open("audit_trail.json") as f:
    audit = json.load(f)

print(f"Session: {audit['session_id']}")
print(f"Iterations: {audit['iterations']}")
for t in audit['trace']:
    print(f"  Iter {t['iteration']}: Sc={t['sc']:.4f}  Decision={t['decision']}")
```

### Cell 10 — Read session report
```python
with open("session_report.txt") as f:
    print(f.read())
```

---

## 🔧 CLI Usage (local / HPC cluster)

```bash
# Demo mode (no model files needed — uses realistic mock data):
python engine.py --image patient.png

# Full run with all options:
python engine.py \
  --image        patient.png          \
  --prior        prior_scan.png       \
  --settings     settings/rmoe_settings.json \
  --vision-proj  models/vision_proj.gguf      \
  --vision-text  models/vision_text.gguf      \
  --reasoning    models/reasoning_expert.gguf \
  --clinical     models/clinical_expert.gguf  \
  --temperature  0.2                  \
  --n-gpu-layers -1                   \
  --hitl         interactive          \
  --chat-target  auto                 \
  --audit-log    audit_trail.json     \
  --session-report session_report.txt \
  --eval --charts

# Benchmark table only:
python engine.py --benchmark-only

# Quiet scripting mode:
python engine.py --image patient.png --quiet --hitl disabled --chat-target none
```

---

## 🏗️ Architecture Overview

```
INPUT (image + clinical notes)
  │
  ▼
╔═════════════════════════════╗      ╔═══════════════════╗
║  PHASE 1: MPE (Perception)  ║◄─────║  DOCTOR UPLOAD /  ║
║  [Moondream2 / Qwen2-VL]    ║      ║  ZOOM COMMAND     ║
║  • Dynamic Res. Adaptation  ║      ╚═══════════════════╝
║  • Visual Token Merger      ║
║  • Saliency-Aware Crop      ║
╚══════════════╤══════════════╝
               │  MPE Confidence Gate
               ▼
╔═════════════════════════════╗      ╔═══════════════════╗
║  PHASE 2: ARLL (Reasoning)  ║◄─────║  DOCTOR QUERY     ║
║  [DeepSeek-R1-Distill]      ║      ║  "Explain this"   ║
║  • Chain-of-Thought (CoT)   ║      ╚═══════════════════╝
║  • DDx Ensemble  Sc=1−σ²   ║
║  • Vector RAG (BM25)        ║
╚══════════════╤══════════════╝
               │  ARLL Confidence Gate  Sc ≥ 0.90?
               │
          ┌────┴────┐
          │NO       │YES
          ▼         ▼
  #wanna# LOOP   ╔══════════════════╗
  1. Zoom MPE    ║  PHASE 3: CSR    ║
  2. Re-eval     ║  [MedGemma-2B]   ║
  3. Ask Doctor  ║  • ICD-11/SNOMED ║
  (max 3 iter)   ║  • Lung-RADS etc ║
          │      ╚════════╤═════════╝
          │               │
          └───────────────▼
                    FINAL REPORT
                + HITL Radiologist Flag
```

---

## 📊 Paper Benchmarks (MIMIC-CXR + RSNA Bone Age)

| Metric | **R-MoE** | GPT-4V | Gemini 1.5 Pro |
|--------|-----------|--------|----------------|
| F1-Score | **0.92** | 0.85 | 0.87 |
| Type I Errors (%) | **5.2** | 7.8 | 7.1 |
| ECE | **0.08** | 0.15 | 0.13 |
| Inference Time (s) | 45 | 32 | 38 |

> Recursion triggered in 15% of cases · 25% fewer false positives vs GPT-4V

---

## 📦 Module Map

```
rmoe/
├── __init__.py    — public exports
├── models.py      — all dataclasses, enums, DDxEnsemble (Sc = 1−σ²)
├── agents.py      — VisionExpert, ReasoningExpert, ReportingExpert, ExpertSwapper
├── core.py        — DiagnosticEngine, WannaStateMachine, MPEConfidenceGate, MrTom
├── hitl.py        — HITLCoordinator, ExpertQueryRouter, zoom command parser
├── rag.py         — VectorRAGEngine (BM25, 25 KB medical knowledge base)
├── ontology.py    — ICD-11 / SNOMED CT tables, all risk scales
├── calibration.py — CalibrationTracker, ECE, Brier score
├── ensemble.py    — MultiTemperatureEnsemble for cross-temp σ²
├── audit.py       — AuditLogger, SessionReportGenerator (+ LaTeX snippet)
├── charts.py      — Sc progress, DDx evolution, reliability diagram, heatmap
├── ui.py          — box-drawing terminal UI, ANSI colours, all print helpers
└── mock.py        — realistic 3-iteration mock responses (demo / offline mode)
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `⚠️  Still missing` after Cell 1 | Re-run Cell 1; check your internet connection or download manually from the shared Drive link |
| `llama_cpp not found` | Re-run Cell 3; check `!pip show llama-cpp-python` |
| CUDA out of memory | Reduce `n_ctx` in `settings/rmoe_settings.json` to 1024 |
| Model not found | Check file names exactly match the table above |
| Mock outputs showing | Model files not found → engine falls back to mock mode |
| `git clone` fails | `!rm -rf /content/Mr.ToM` then re-run Cell 2 |

---

## 📄 Citation

```bibtex
@article{rmoe2025,
  title   = {Recursive Multi-Agent Mixture-of-Experts (RMoE)
             for Autonomous Clinical Diagnostics},
  year    = {2026},
  note    = {Open-source implementation at github.com/dill-lk/Mr.ToM}
}
```
