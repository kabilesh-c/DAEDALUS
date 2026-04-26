"""
Submit DAEDALUS training as a Hugging Face Job.

Hardware: 4x Nvidia L4  (96 GB VRAM total, 48 vCPU, 186 GB RAM)
Cost:     ~$0.063/min (~$3.80/hr)
Expected: ~60-90 min for TRAIN_MODE=full → total ~$4-6

The job:
  1. Mounts Laksh718/daedalus-training-space at /workspace
     (gives job access to train_hf.py and the daedalus/ package)
  2. Runs job_runner.py which bootstraps torch→unsloth→deps, then
     executes /workspace/train_hf.py
  3. train_hf.py pushes the merged model + training history to
     Laksh718/daedalus-designer on completion.

Usage:
    export HF_TOKEN="hf_..."
    python submit_job.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, Volume

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: set HF_TOKEN env var first", file=sys.stderr)
    sys.exit(1)

TRAINING_SPACE = os.environ.get("DAEDALUS_TRAINING_SPACE", "Laksh718/daedalus-training-space")
HUB_MODEL_ID   = os.environ.get("DAEDALUS_HUB_MODEL_ID",   "Laksh718/daedalus-designer")
FLAVOR         = os.environ.get("DAEDALUS_JOB_FLAVOR",      "l4x4")
TRAIN_MODE     = os.environ.get("DAEDALUS_TRAIN_MODE",       "full")
TIMEOUT        = os.environ.get("DAEDALUS_JOB_TIMEOUT",      "3h")

api = HfApi(token=HF_TOKEN)
me = api.whoami()
print(f"[auth] authenticated as: {me.get('name', '?')}")
print(f"[job] flavor={FLAVOR}  mode={TRAIN_MODE}  model→{HUB_MODEL_ID}")

# Mount the training Space at /workspace so job_runner.py can run train_hf.py
# and find the daedalus/ package (train_hf.py adds /workspace to sys.path).
volume = Volume(
    type="space",
    source=TRAINING_SPACE,
    mount_path="/workspace",
)

print(f"[job] mounting {TRAINING_SPACE} → /workspace")
print(f"[job] submitting to HF Jobs infrastructure ...")

try:
    job = api.run_uv_job(
        script="job_runner.py",       # local file — gets uploaded to the job
        dependencies=None,            # job_runner.py bootstraps its own deps
        secrets={"HF_TOKEN": HF_TOKEN},
        env={
            "TRAIN_MODE":    TRAIN_MODE,
            "HUB_MODEL_ID":  HUB_MODEL_ID,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/tmp/hf_cache",
        },
        flavor=FLAVOR,
        timeout=TIMEOUT,
        volumes=[volume],
        token=HF_TOKEN,
    )

    print("=" * 60)
    print(f"Job submitted!")
    print(f"Job ID:    {getattr(job, 'id', job)}")
    print(f"Logs:      {getattr(job, 'url', 'check HF dashboard')}")
    print(f"Model out: https://huggingface.co/{HUB_MODEL_ID}")
    print("=" * 60)
    print("Watch for sentinel in logs:")
    print("    [grpo v5] five-reward single-adapter")

except Exception as e:
    print(f"ERROR: job submission failed: {e}", file=sys.stderr)
    raise
