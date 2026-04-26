import os
import sys
from huggingface_hub import HfApi, Volume

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: set HF_TOKEN env var first", file=sys.stderr)
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
space_volume = Volume(type="space", source="kabilesh-c/daedalus", mount_path="/workspace")

print("Submitting parallel 50-step training job to Hugging Face...")
try:
    job = api.run_uv_job(
        script="train_hf.py",
        dependencies=["trl", "unsloth", "torch", "transformers", "datasets", "accelerate", "openenv", "fastapi", "pydantic", "vllm"],
        flavor="l4x1",
        volumes=[space_volume],
        env={
            "GRPO_STEPS": "50",
            "N_GRPO_PROMPTS": "50",
            "HUB_MODEL_ID": "kabilesh-c/daedalus-designer-50steps"
        },
        timeout="2h"
    )
    print("---------------------------------------------------------")
    print(f"Success! Parallel job (50 steps) submitted.")
    print(f"View Live Logs: {job.url}")
    print("---------------------------------------------------------")
except Exception as e:
    print(f"Failed to submit: {e}")
