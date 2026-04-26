"""
Deploy DAEDALUS training as a Hugging Face Docker Space.

Steps:
  1. Reads HF_TOKEN from env (NEVER hardcoded).
  2. Builds space_deploy/ with:
       train_hf.py, daedalus/, Dockerfile, requirements.txt, README.md
  3. Creates / updates the Space repo.
  4. Uploads all files (the step that actually ships the trainer).
  5. Adds HF_TOKEN as a Space Secret.
  6. Sets TRAIN_MODE + HUB_MODEL_ID as Space variables.
  7. Requests GPU hardware.
  8. Restarts the Space to trigger a rebuild.

Environment variables:
    HF_TOKEN                  (required) Hugging Face write token
    DAEDALUS_TRAINING_SPACE   Space repo id  (default: Laksh718/daedalus-training-space)
    DAEDALUS_HUB_MODEL_ID     Output model   (default: Laksh718/daedalus-designer)
    DAEDALUS_HARDWARE         GPU hardware   (default: a100-large)
    DAEDALUS_TRAIN_MODE       smoke|short|long|full  (default: long)

Usage:
    export HF_TOKEN="hf_..."
    python deploy_training_space.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — override via env vars, never hardcode tokens
# ---------------------------------------------------------------------------
REPO_ID      = os.environ.get("DAEDALUS_TRAINING_SPACE", "Laksh718/daedalus-training-space")
HUB_MODEL_ID = os.environ.get("DAEDALUS_HUB_MODEL_ID",   "Laksh718/daedalus-designer")
HARDWARE     = os.environ.get("DAEDALUS_HARDWARE",        "a100x4")
TRAIN_MODE   = os.environ.get("DAEDALUS_TRAIN_MODE",      "long")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN environment variable is not set.\n"
        "  export HF_TOKEN='hf_...'\n"
        "  python deploy_training_space.py",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT       = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "space_deploy"

# ---------------------------------------------------------------------------
# Dockerfile — root cause analysis and definitive fix
#
# Every runtime error so far traces to this single chain:
#   unsloth → unsloth_zoo → transformers (latest)
#          → quantizers/auto.py (unconditional import since 4.46)
#          → quantizer_torchao.py
#          → torchao.__init__
#          → torch.utils._pytree.register_constant   ← ADDED IN torch 2.7.0
#
# torch.int1 (needed by torchao.quant_primitives) was added in torch 2.6.0.
# torch.utils._pytree.register_constant was added in torch 2.7.0.
# We must therefore use torch 2.7.0.  There is no lower version that works
# with the current unsloth + transformers + torchao stack.
#
# Install order matters:
#   1. torch 2.7.0 + cu124 FIRST — unsloth detects GPU at install time.
#   2. unsloth ALONE — lets it pin the exact transformers/trl/peft it needs.
#   3. everything else — no torch/unsloth/transformers here.
#
# Key fixes vs prior attempts:
#   - torchvision removed: LLM text training does not use it; its cu124 wheel
#     for 0.22.0 was the actual cause of the Step-1 build failure.
#   - --extra-index-url instead of --index-url: keeps PyPI accessible so that
#     packages not mirrored on the PyTorch CDN still resolve correctly.
# ---------------------------------------------------------------------------
DOCKERFILE = """\
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    HF_HOME=/tmp/hf_cache \\
    TRANSFORMERS_CACHE=/tmp/hf_cache \\
    HF_HUB_ENABLE_HF_TRANSFER=1 \\
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \\
        git build-essential ca-certificates curl && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: PyTorch 2.7.0 + CUDA 12.4 ─────────────────────────────────────
# torch.int1 added in 2.6.0; torch.utils._pytree.register_constant in 2.7.0.
# torchvision intentionally omitted — not needed for LLM text training and its
# cu124 wheel for 0.22.0 caused the Step-1 build failure in prior attempts.
# --extra-index-url keeps PyPI accessible (--index-url replaces it entirely).
RUN pip install --upgrade pip && \\
    pip install torch==2.7.0 \\
        --extra-index-url https://download.pytorch.org/whl/cu124

# ── Step 2: Unsloth alone ──────────────────────────────────────────────────
# Own pip step so unsloth controls its transformers/peft/trl pins without
# interference from other packages in the same resolver run.
RUN pip install \\
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ── Step 3: Remaining dependencies ─────────────────────────────────────────
# torch / unsloth / transformers intentionally absent — already installed.
# bitsandbytes 0.45+ is required for torch 2.7 compatibility.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "train_hf.py"]
"""

# ---------------------------------------------------------------------------
# requirements.txt
# torch        — Step 1 Dockerfile
# unsloth      — Step 2 Dockerfile
# transformers — owned by unsloth, do NOT pin here
# ---------------------------------------------------------------------------
REQUIREMENTS = """\
huggingface_hub>=0.26.0
hf_transfer>=0.1.8
datasets>=3.0.0
accelerate>=1.0.0
peft>=0.13.0
trl>=0.12.0
bitsandbytes>=0.45.0
sentencepiece>=0.2.0
protobuf>=4.25.0
python-dotenv>=1.0.0
"""

README_TEMPLATE = """\
---
title: DAEDALUS Training
emoji: \U0001f3db
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
hardware: {hardware}
---

# DAEDALUS Training Space (v5)

One-shot Docker Space that trains the DAEDALUS mechanism designer using
Unsloth + Qwen2.5-0.5B-Instruct + two-phase SFT → GRPO:

1. **Load** `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` (4-bit, ~2× faster startup).
2. **Attach a single LoRA** (rank 16, all attn + MLP, `lora_dropout=0` for Unsloth fast path).
3. **SFT** — teach the JSON output schema on synthetic `(observation, valid_mechanism)` pairs.
4. **GRPO** — five reward signals on the SAME LoRA:
   - `reward_format`    — schema coverage ∈ [−1, 1]
   - `reward_welfare`   — social welfare ratio W ∈ [0, 1]
   - `reward_fairness`  — 1 − Gini(surplus) ∈ [0, 1]
   - `reward_stability` — 1 − 3σ(welfare) ∈ [0, 1]
   - `reward_composite` — full R = W × F × P × S × anti_collusion
5. **Merge & push** the full 16-bit model to `{hub_model_id}`.

Sentinel line in container logs: `[grpo v5] five-reward single-adapter`

| `TRAIN_MODE` | SFT examples | GRPO steps | A100 time | Use case                      |
|--------------|--------------|------------|-----------|-------------------------------|
| `smoke`      | 24           | 4          | ~3 min    | shake out build errors        |
| `short`      | 320          | 120        | ~10 min   | quick sanity check            |
| `long`       | 800          | 300        | ~30 min   | solid model (default)         |
| `full`       | 2000         | 500        | ~75 min   | best quality                  |

Load the trained model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{hub_model_id}")
tok   = AutoTokenizer.from_pretrained("{hub_model_id}")
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def p(msg: str) -> None:
    print(msg, flush=True)


def build_deploy_dir() -> None:
    p(f"[1/8] preparing {DEPLOY_DIR.name}/ ...")
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir()

    # Trainer script
    shutil.copy(ROOT / "train_hf.py", DEPLOY_DIR / "train_hf.py")

    # Canonical daedalus package (build artifacts — pycache excluded)
    target_pkg = DEPLOY_DIR / "daedalus"
    shutil.copytree(
        ROOT / "daedalus",
        target_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    (DEPLOY_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (DEPLOY_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (DEPLOY_DIR / "README.md").write_text(
        README_TEMPLATE.format(hardware=HARDWARE, hub_model_id=HUB_MODEL_ID),
        encoding="utf-8",
    )

    sha = file_sha256(DEPLOY_DIR / "train_hf.py")
    p(f"        deploy dir ready: {DEPLOY_DIR}")
    p(f"        train_hf.py sha256: {sha[:16]}...")


def main() -> None:
    api = HfApi(token=HF_TOKEN)
    me = api.whoami()
    p(f"[auth] authenticated as: {me.get('name', '?')}")

    build_deploy_dir()

    # ── 2. Create / ensure Space exists ────────────────────────────────────
    p(f"[2/8] ensuring Space '{REPO_ID}' exists ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )

    # ── 3. Upload all files ─────────────────────────────────────────────────
    p("[3/8] uploading files to Space ...")
    commit_info = api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo=".",
        ignore_patterns=["__pycache__/*", "*.pyc", "*.pyo"],
        commit_message=f"deploy v5 ({TRAIN_MODE} mode, {HARDWARE})",
    )
    commit_url = getattr(commit_info, "commit_url", None) or getattr(commit_info, "oid", "?")
    p(f"        upload commit: {commit_url}")

    # ── 4. HF_TOKEN secret ─────────────────────────────────────────────────
    p("[4/8] installing HF_TOKEN as Space secret ...")
    try:
        api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=HF_TOKEN)
        p("        HF_TOKEN secret installed")
    except Exception as e:
        p(f"        WARNING: could not set secret: {e}")

    # ── 6. TRAIN_MODE + HUB_MODEL_ID variables ─────────────────────────────
    # NOTE: SPACE_ID is a reserved HF variable — it is injected automatically
    # by HF Spaces at runtime and MUST NOT be set manually.
    p(f"[5/8] setting Space variables (TRAIN_MODE={TRAIN_MODE}, HUB_MODEL_ID={HUB_MODEL_ID}) ...")
    for key, val in [("TRAIN_MODE", TRAIN_MODE), ("HUB_MODEL_ID", HUB_MODEL_ID)]:
        try:
            api.add_space_variable(repo_id=REPO_ID, key=key, value=val)
            p(f"        {key}={val}")
        except Exception as e:
            p(f"        WARNING: could not set {key}: {e}")

    # ── 6. Request GPU hardware ─────────────────────────────────────────────
    p(f"[6/8] requesting hardware: {HARDWARE} ...")
    try:
        api.request_space_hardware(repo_id=REPO_ID, hardware=HARDWARE)
        p(f"        {HARDWARE} requested")
    except Exception as e:
        p(
            f"        WARNING: hardware request failed: {e}\n"
            f"        → Set hardware manually in Space settings if needed."
        )

    # ── 7. Restart to rebuild ───────────────────────────────────────────────
    p("[7/7] restarting Space (triggers Docker rebuild + training run) ...")
    try:
        api.restart_space(repo_id=REPO_ID, factory_reboot=True)
        p("        restart triggered (factory reboot for clean rebuild)")
    except Exception as e:
        p(f"        WARNING: restart failed: {e}")

    space_url = f"https://huggingface.co/spaces/{REPO_ID}"
    model_url = f"https://huggingface.co/{HUB_MODEL_ID}"
    p("")
    p("=" * 70)
    p(f"Space:       {space_url}")
    p(f"Build logs:  {space_url}?logs=container")
    p(f"Model out:   {model_url}")
    p(f"Mode:        TRAIN_MODE={TRAIN_MODE}  hardware={HARDWARE}")
    p("=" * 70)
    p(
        "Watch for sentinel in container logs:\n"
        "    [grpo v5] five-reward single-adapter\n"
        "If absent, the old code shipped — re-run this script (do NOT just click Restart)."
    )


if __name__ == "__main__":
    main()
