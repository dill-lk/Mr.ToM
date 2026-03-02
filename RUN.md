# R-MoE v2.0 — Colab Run Guide

> **Recursive Multi-Agent Mixture-of-Experts for Autonomous Clinical Diagnostics**
> Paper benchmarks: F1 = 0.92 · ECE = 0.08 · 25% fewer false positives vs GPT-4V

---

## 📁 Drive Setup (one-time)

Create this folder structure in your Google Drive **before** running any cell:

```
MyDrive/
└── Medical_MoE_Models/
    ├── vision_text.gguf        ← Moondream2 vision backbone
    ├── vision_proj.gguf        ← CLIP mmproj file (same release as vision_text)
    ├── reasoning_expert.gguf   ← DeepSeek-R1-Distill reasoning model
    ├── clinical_expert.gguf    ← MedGemma-2B clinical synthesis model
    └── test_patient.png        ← Your patient chest X-ray / scan
```

---

## 🤖 Recommended Models for T4 GPU (16 GB VRAM)

> All models run **one at a time** via `ExpertSwapper` — peak VRAM usage < 4 GB per phase.

| Role | Model | File to rename | Size | Download |
|------|-------|----------------|------|----------|
| **Vision MPE** | Moondream2 2B int8 | `vision_text.gguf` | ~2.5 GB | [Hugging Face ↗](https://huggingface.co/vikhyatk/moondream2) |
| **Vision CLIP** | Moondream2 mmproj | `vision_proj.gguf` | ~400 MB | Same release as above |
| **Reasoning ARLL** | DeepSeek-R1-Distill-Qwen-1.5B Q8 | `reasoning_expert.gguf` | ~1.8 GB | [Hugging Face ↗](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) |
| **Clinical CSR** | MedGemma-2B-it Q8 | `clinical_expert.gguf` | ~2.2 GB | [Hugging Face ↗](https://huggingface.co/google/medgemma-2b-it) |

> **Alternatives (if above are unavailable):**
> - Reasoning: `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` (~5 GB, higher quality)
> - Clinical: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` + radiology fine-tune
> - Vision: `Qwen2-VL-2B-Instruct-Q8_0.gguf` (requires `vision_proj` companion)

---

## 🚀 Cell-by-Cell Colab Instructions

### Cell 1 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
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

### Cell 4 — Stage Models from Drive → /content/models
```python
import sys
sys.path.insert(0, '/content/Mr.ToM')
from colab_runner import setup_environment
setup_environment()
```

### Cell 5 — Run R-MoE (full demo with all charts)
```python
from colab_runner import run_python_engine

run_python_engine(
    image        = "/content/models/test_patient.png",
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

### Cell 6 — Interactive Doctor Q&A (post-diagnosis)
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

### Cell 7 — Benchmark comparison only (no models needed)
```python
from colab_runner import quick_benchmark
quick_benchmark()
```

### Cell 8 — Read audit trail
```python
import json
with open("audit_trail.json") as f:
    audit = json.load(f)

print(f"Session: {audit['session_id']}")
print(f"Iterations: {audit['iterations']}")
for t in audit['trace']:
    print(f"  Iter {t['iteration']}: Sc={t['sc']:.4f}  Decision={t['decision']}")
```

### Cell 9 — Read session report
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
| `❌ Drive folder not found` | Run Cell 1 (mount Drive) before Cell 4 |
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
  year    = {2025},
  note    = {Open-source implementation at github.com/dill-lk/Mr.ToM}
}
```
