"""
colab_runner.py — Google Colab launcher for R-MoE v2.0.

Run each numbered cell in order inside a Colab notebook.
See RUN.md for the complete guide with model download links.

Expected Drive layout:
    MyDrive/
    └── Medical_MoE_Models/
        ├── vision_text.gguf        ← Moondream2-2B (moondream-2b-int8.gguf)
        ├── vision_proj.gguf        ← CLIP mmproj companion file
        ├── reasoning_expert.gguf   ← DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf
        ├── clinical_expert.gguf    ← MedGemma-2B-it-Q8_0.gguf
        └── test_patient.png        ← Patient image for diagnosis
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
DRIVE_DIR        = "/content/drive/MyDrive/Medical_MoE_Models"
LOCAL_MODELS_DIR = "/content/models"
REPO_DIR         = "/content/Mr.ToM"

_MODEL_FILES = [
    "vision_text.gguf",
    "vision_proj.gguf",
    "reasoning_expert.gguf",
    "clinical_expert.gguf",
    "test_patient.png",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 1 — Mount Google Drive
# ═══════════════════════════════════════════════════════════════════════════════

def mount_drive() -> bool:
    """Mount Google Drive.  Skip silently if already mounted."""
    if os.path.exists("/content/drive/MyDrive"):
        print("✅ Drive already mounted.")
        return True
    try:
        from google.colab import drive  # type: ignore[import-untyped]
        drive.mount("/content/drive")
        print("✅ Drive mounted.")
        return True
    except ImportError:
        print("⚠️  google.colab not available — are you inside Colab?")
        return False
    except Exception as exc:
        print(f"❌ Drive mount failed: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 2 — Install llama-cpp-python (CUDA wheel)
# ═══════════════════════════════════════════════════════════════════════════════

def install_dependencies(force: bool = False) -> bool:
    """
    Install llama-cpp-python with CUDA support.

    We use a pre-built wheel from the llama-cpp-python releases — this takes
    < 60 seconds, vs 15–20 minutes to compile from source with CMAKE_ARGS.

    Why not C++?
      Compiling the C++ backend with -DGGML_CUDA=ON on Colab free-tier
      hits CUDA rate-limits before the build finishes.  The pip wheel gives
      identical GGML/CUDA performance without any compilation step.
    """
    try:
        import llama_cpp  # type: ignore[import-untyped]
        if not force:
            print(f"✅ llama-cpp-python already installed ({llama_cpp.__version__})")
            return True
    except ImportError:
        pass

    print("📦 Installing llama-cpp-python (CUDA pre-built wheel) …")
    # Use the pre-built CUDA 12.1 wheel — no C++ compilation required
    cmd = (
        "pip install llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 "
        "--upgrade --quiet"
    )
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print("⚠️  CUDA wheel failed — falling back to CPU wheel …")
        ret2 = subprocess.run("pip install llama-cpp-python --quiet", shell=True)
        if ret2.returncode != 0:
            print("❌ llama-cpp-python installation failed.")
            return False
    print("✅ llama-cpp-python installed.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 3 — Clone / update repository
# ═══════════════════════════════════════════════════════════════════════════════

def setup_repo(branch: str = "main") -> bool:
    """Clone or update the Mr.ToM repository."""
    if os.path.isdir(os.path.join(REPO_DIR, "rmoe")):
        print(f"✅ Repo already present at {REPO_DIR}")
        return True
    print("📥 Cloning Mr.ToM …")
    ret = subprocess.run(
        ["git", "clone", "--depth=1",
         f"--branch={branch}",
         "https://github.com/dill-lk/Mr.ToM.git",
         REPO_DIR],
    )
    if ret.returncode != 0:
        print("❌ git clone failed.")
        return False
    print(f"✅ Repo cloned → {REPO_DIR}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 4 — Stage model files from Drive
# ═══════════════════════════════════════════════════════════════════════════════

def setup_environment() -> bool:
    """
    Copy model files from Google Drive into /content/models (fast local SSD).
    Skips files that are already staged.
    """
    if not os.path.exists("/content"):
        print("⚠️  /content not found — are you running inside Google Colab?")
        return False
    if not os.path.exists(DRIVE_DIR):
        print(
            f"❌ Drive folder not found: {DRIVE_DIR}\n"
            "   Please mount your Drive and place the model files there.\n"
            "   See RUN.md for the complete setup guide."
        )
        return False

    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    all_ready = True

    for fname in _MODEL_FILES:
        src  = os.path.join(DRIVE_DIR, fname)
        dst  = os.path.join(LOCAL_MODELS_DIR, fname)

        if not os.path.exists(src):
            print(f"  ❌ MISSING on Drive: {src}")
            all_ready = False
            continue

        if os.path.exists(dst):
            src_size = os.path.getsize(src)
            dst_size = os.path.getsize(dst)
            if src_size == dst_size:
                print(f"  ⚡ {fname} already staged ({dst_size // 1024 // 1024} MB)")
                continue
            print(f"  🔄 {fname} size mismatch — re-copying …")

        size_mb = os.path.getsize(src) // 1024 // 1024
        print(f"  🚚 Copying {fname} ({size_mb} MB) …", end="", flush=True)
        shutil.copy2(src, dst)
        print(" ✅")

    if all_ready:
        print(f"\n✅ All model files ready at {LOCAL_MODELS_DIR}")
    return all_ready


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 5 — Run Python engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_python_engine(
    image: Optional[str]    = None,
    prior_image: Optional[str] = None,
    temperature: float       = 0.2,
    n_predict: int           = 512,
    n_gpu_layers: int        = -1,
    audit_log: Optional[str] = "audit_trail.json",
    chat_target: str         = "auto",
    hitl: str                = "auto",
    eval_mode: bool          = False,
    charts: bool             = True,
    quiet: bool              = False,
    session_report: Optional[str] = "session_report.txt",
) -> None:
    """
    Launch the R-MoE v2.0 Python engine.

    Args:
        image:          Patient image path.  Defaults to Drive test image.
        prior_image:    Optional prior scan for temporal analysis.
        temperature:    Sampling temperature (0.2 = clinical precision).
        n_predict:      Max tokens per inference step.
        n_gpu_layers:   GPU layers to offload (-1 = all layers = full T4 GPU).
        audit_log:      JSON audit trail output path.
        chat_target:    Post-diagnosis Q&A expert: reasoning | clinical | auto | none.
        hitl:           HITL mode: interactive | auto | disabled.
        eval_mode:      Print ECE calibration + benchmark comparison.
        charts:         Print ASCII Sc/DDx/uncertainty charts.
        quiet:          Suppress banner and HITL prompts.
        session_report: Write full text session report to this path.
    """
    if not setup_environment():
        print("🛑 Setup failed — check that all model files are on Drive.")
        return

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    image = image or os.path.join(LOCAL_MODELS_DIR, "test_patient.png")

    argv = [
        "--image",       image,
        "--settings",    os.path.join(REPO_DIR, "settings", "rmoe_settings.json"),
        "--prompts-dir", os.path.join(REPO_DIR, "prompts"),
        "--vision-proj", os.path.join(LOCAL_MODELS_DIR, "vision_proj.gguf"),
        "--vision-text", os.path.join(LOCAL_MODELS_DIR, "vision_text.gguf"),
        "--reasoning",   os.path.join(LOCAL_MODELS_DIR, "reasoning_expert.gguf"),
        "--clinical",    os.path.join(LOCAL_MODELS_DIR, "clinical_expert.gguf"),
        "--temperature", str(temperature),
        "--n-predict",   str(n_predict),
        "--n-gpu-layers", str(n_gpu_layers),
        "--hitl",        hitl,
        "--chat-target", chat_target,
    ]
    if prior_image:
        argv += ["--prior", prior_image]
    if audit_log:
        argv += ["--audit-log", audit_log]
    if session_report:
        argv += ["--session-report", session_report]
    if eval_mode or charts:
        argv += ["--eval"]
    if charts:
        argv += ["--charts"]
    if quiet:
        argv += ["--quiet"]

    try:
        import engine  # type: ignore[import-untyped]
        engine.main(argv)
    except SystemExit:
        pass
    except ImportError as exc:
        print(f"❌ Cannot import engine.py: {exc}")
        print(f"   Ensure {REPO_DIR}/engine.py exists.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience one-liners for Colab cells
# ═══════════════════════════════════════════════════════════════════════════════

def quick_demo() -> None:
    """Run a full demo with all charts and evaluation output."""
    run_python_engine(eval_mode=True, charts=True, chat_target="none", hitl="disabled")


def quick_interactive() -> None:
    """Run with interactive doctor Q&A loop."""
    run_python_engine(hitl="interactive", chat_target="auto")


def quick_benchmark() -> None:
    """Print benchmark comparison table only (no inference)."""
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    try:
        from rmoe.charts import benchmark_comparison, reliability_diagram, _paper_calibration_bins
        from rmoe.ui import print_banner
        print_banner()
        benchmark_comparison()
        reliability_diagram(_paper_calibration_bins(), 0.08)
    except ImportError as exc:
        print(f"❌ {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry-point  (python colab_runner.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mount_drive()
    install_dependencies()
    setup_repo()
    run_python_engine(eval_mode=True, charts=True)
