"""
HF Training Job entry point for DAEDALUS.

This script has ZERO external imports at the top level — it runs in a
bare Python 3.11 environment and bootstraps the full dependency stack
in the correct order before launching train_hf.py.

Dependency order is critical:
  1. torch 2.7.0 + cu124  — must be first so unsloth detects GPU correctly
  2. unsloth              — must come alone so it pins transformers/trl/peft
  3. everything else      — no torch/unsloth/transformers pins here

The daedalus/ package is provided by the mounted volume at /workspace.
"""

import os
import subprocess
import sys


def run(cmd: list, **kw) -> None:
    print(f">>> {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


print("=" * 60)
print("DAEDALUS Job Runner — bootstrapping dependencies")
print("=" * 60, flush=True)

# ── Step 1: PyTorch 2.7.0 + CUDA 12.4 ────────────────────────────────────
# torch.int1 needs 2.6+; torch.utils._pytree.register_constant needs 2.7+
# torchvision intentionally omitted — not needed for LLM text training.
print("\n[step 1/3] Installing torch 2.7.0 (cu124)...")
run([
    sys.executable, "-m", "pip", "install",
    "torch==2.7.0",
    "--extra-index-url", "https://download.pytorch.org/whl/cu124",
    "-q",
])

# ── Step 2: Unsloth alone ─────────────────────────────────────────────────
print("\n[step 2/3] Installing unsloth...")
run([
    sys.executable, "-m", "pip", "install",
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "-q",
])

# ── Step 3: Remaining dependencies ───────────────────────────────────────
print("\n[step 3/3] Installing remaining dependencies...")
run([
    sys.executable, "-m", "pip", "install",
    "huggingface_hub>=0.26.0",
    "hf_transfer>=0.1.8",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "peft>=0.13.0",
    "trl>=0.12.0",
    "bitsandbytes>=0.45.0",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25.0",
    "python-dotenv>=1.0.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "-q",
])

print("\n[bootstrap] all dependencies installed — launching train_hf.py", flush=True)
print("=" * 60, flush=True)

# The training Space is mounted at /workspace by the Job volume config.
# train_hf.py already has logic to add /workspace to sys.path for daedalus/.
os.chdir("/workspace")
run([sys.executable, "-u", "train_hf.py"])
