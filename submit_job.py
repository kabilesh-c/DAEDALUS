import os
import sys
from huggingface_hub import HfApi, Volume

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(
        "ERROR: set HF_TOKEN env var first  (PowerShell: $env:HF_TOKEN = 'hf_xxx')",
        file=sys.stderr,
    )
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Mount the Space where our repository lives into /workspace
space_volume = Volume(type="space", source="kabilesh-c/daedalus", mount_path="/workspace")

print("Submitting training job to Hugging Face Compute via Python API...")
try:
    job = api.run_uv_job(
        script="train_hf.py",
        dependencies=["trl", "unsloth", "torch", "transformers", "datasets", "accelerate", "openenv", "fastapi", "pydantic", "vllm"],
        flavor="t4-small", # Requesting T4 GPU directly
        volumes=[space_volume],
        timeout="4h"
    )
    print("---------------------------------------------------------")
    print(f"Success! Job submitted successfully! Training is now running on a T4 GPU.")
    print(f"View Live Logs: {job.url}")
    print("---------------------------------------------------------")
except Exception as e:
    safe_err = str(e).encode('ascii', 'ignore').decode('ascii')
    print(f"Failed to submit job: {safe_err}")
