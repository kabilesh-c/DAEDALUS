import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

JOB_ID = "69edade7d70108f37acdfb3f"

try:
    print(f"Fetching logs for job {JOB_ID}...")
    # fetch_job_logs expects keyword-only job_id
    logs = api.fetch_job_logs(job_id=JOB_ID)
    if hasattr(logs, "__iter__") and not isinstance(logs, str):
        for line in logs:
            print(line, end="")
    else:
        print(logs)
except Exception as e:
    print(f"Error fetching logs: {e}")
