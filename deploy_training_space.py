"""
Deploy DAEDALUS training as a Hugging Face Docker Space.

This script:
  1. Reads HF_TOKEN from the environment (NEVER hardcoded).
  2. Builds a slim space_deploy/ directory with:
       - train_hf.py        (the SFT + merge + GRPO trainer)
       - daedalus/          (the OpenEnv environment package)
       - Dockerfile         (python:3.11-slim + CUDA torch wheel)
       - requirements.txt
       - README.md          (Space metadata header)
  3. Creates / updates the kabilesh-c/daedalus-training-space Space.
  4. Uploads the directory to the Space (returns a commit SHA -> proof
     that train_hf.py actually shipped, fixes the silent-no-op
     restart-only failure mode we hit twice).
  5. Adds HF_TOKEN as a Space Secret so train_hf.py can push the adapter.
  6. Requests t4-medium hardware.
  7. Calls restart_space() to resume + rebuild.

Usage (PowerShell):
    $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    python deploy_training_space.py

Optional env vars:
    DAEDALUS_TRAINING_SPACE  override Space repo id
    DAEDALUS_HARDWARE        override GPU flavor (default: t4-medium)
    DAEDALUS_TRAIN_MODE      'short' (default) or 'long' - forwarded as a
                             Space variable so train_hf.py picks it up.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables from .env
load_dotenv()


REPO_ID = os.environ.get("DAEDALUS_TRAINING_SPACE", "kabilesh-c/daedalus-training-space")
HARDWARE = os.environ.get("DAEDALUS_HARDWARE", "t4-medium")
TRAIN_MODE = os.environ.get("DAEDALUS_TRAIN_MODE", "short")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN environment variable is not set.\n"
        "  PowerShell:  $env:HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxxxx'\n"
        "  Then re-run: python deploy_training_space.py",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "space_deploy"

DOCKERFILE = """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    HF_HOME=/app/.cache \\
    TRANSFORMERS_CACHE=/app/.cache \\
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
        git build-essential ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "train_hf.py"]
"""

REQUIREMENTS = """openenv-core>=0.2.3
huggingface_hub>=0.26.0
hf_transfer>=0.1.8
transformers>=4.45.0
datasets>=3.0.0
accelerate>=0.34.0
peft>=0.13.0
trl>=0.12.0
torch>=2.4.0
"""

README_TEMPLATE = """---
title: DAEDALUS Training
emoji: \U0001f3db
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
hardware: t4-medium
---

# DAEDALUS Training Space (v3)

One-shot Docker Space that trains the DAEDALUS mechanism designer
on a T4 GPU using a robust three-step pipeline:

1. **SFT warmup** on `Qwen/Qwen2.5-0.5B-Instruct` with synthetic
   `(prompt, valid mechanism JSON)` pairs - teaches the JSON output format.
2. **Merge step** - bake the SFT LoRA into the base weights via
   `merge_and_unload()`. This avoids the `AutoPeftModel` "frozen
   adapter" trap that broke GRPO at step 0 in earlier versions.
3. **GRPO refinement** with a fresh LoRA (passed to `GRPOTrainer` via
   `peft_config`, so TRL handles `requires_grad` correctly by
   construction) and a format-shaped reward.

Look for the sentinel line `[grpo v3] using merge+fresh-adapter approach`
in container logs to confirm the new code is live.

The trained LoRA adapter and `training_history.json` are pushed to
[`kabilesh-c/daedalus-designer`](https://huggingface.co/kabilesh-c/daedalus-designer)
and the Space auto-pauses on completion.

Mode is controlled by the `TRAIN_MODE` Space variable
(default: `short`, ~10-15 min training; `long` ~45-60 min).
"""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def p(msg: str) -> None:
    """Flushed print so the user sees progress immediately on slow shells."""
    print(msg, flush=True)


def build_deploy_dir() -> None:
    p(f"[1/7] preparing {DEPLOY_DIR.name}/ ...")
    DEPLOY_DIR.mkdir(exist_ok=True)

    shutil.copy(ROOT / "train_hf.py", DEPLOY_DIR / "train_hf.py")

    target_pkg = DEPLOY_DIR / "daedalus"
    if target_pkg.exists():
        shutil.rmtree(target_pkg)
    shutil.copytree(
        ROOT / "daedalus",
        target_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    (DEPLOY_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (DEPLOY_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (DEPLOY_DIR / "README.md").write_text(README_TEMPLATE, encoding="utf-8")

    sha = file_sha256(DEPLOY_DIR / "train_hf.py")
    p(f"        wrote {DEPLOY_DIR}")
    p(f"        train_hf.py sha256: {sha[:16]}... (compare with the post-upload SHA below)")


def main() -> None:
    api = HfApi(token=HF_TOKEN)
    me = api.whoami()
    p(f"[auth] authenticated as: {me.get('name', '?')}")

    build_deploy_dir()

    p(f"[2/7] ensuring Space {REPO_ID} exists ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    p(f"[3/7] uploading files to Space (this is the step that ACTUALLY ships train_hf.py) ...")
    commit_info = api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo=".",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message=f"deploy v3 ({TRAIN_MODE} mode)",
    )
    commit_oid = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_url", None)
    p(f"        upload commit: {commit_oid}")

    p(f"[4/7] adding HF_TOKEN as a Space secret ...")
    try:
        api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=HF_TOKEN)
        p("        HF_TOKEN secret installed")
    except Exception as e:
        p(f"        WARNING: could not set secret: {e}")

    p(f"[5/7] setting Space variable TRAIN_MODE={TRAIN_MODE} ...")
    try:
        api.add_space_variable(repo_id=REPO_ID, key="TRAIN_MODE", value=TRAIN_MODE)
        p(f"        TRAIN_MODE={TRAIN_MODE} installed")
    except Exception as e:
        p(f"        WARNING: could not set variable: {e}")

    p(f"[6/7] requesting GPU hardware: {HARDWARE} ...")
    try:
        api.request_space_hardware(repo_id=REPO_ID, hardware=HARDWARE)
        p(f"        {HARDWARE} requested")
    except Exception as e:
        p(
            f"        WARNING: hardware request failed: {e}\n"
            f"        (Open the Space settings and set hardware to {HARDWARE} manually if needed.)"
        )

    p(f"[7/7] restarting Space (resumes if paused, rebuilds image) ...")
    try:
        api.restart_space(repo_id=REPO_ID)
        p("        restart triggered")
    except Exception as e:
        p(f"        WARNING: restart failed: {e}")

    space_url = f"https://huggingface.co/spaces/{REPO_ID}"
    model_url = "https://huggingface.co/kabilesh-c/daedalus-designer"
    p("")
    p("=" * 64)
    p(f"Space:        {space_url}")
    p(f"Build logs:   {space_url}?logs=container")
    p(f"Adapter out:  {model_url}")
    p(f"Mode:         TRAIN_MODE={TRAIN_MODE}")
    p("=" * 64)
    p(
        "Watch the container logs for the line:\n"
        "    [grpo v3] using merge+fresh-adapter approach\n"
        "If you DON'T see it, the new code did not ship and you should re-run\n"
        "this script (NOT just click Restart on the Space settings page)."
    )


if __name__ == "__main__":
    main()
