import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
try:
    # api.list_jobs is likely available or we can use another method
    # Let's try to find experimental jobs API
    print("Listing jobs for owner 'kabilesh-c'...")
    # NOTE: run_uv_job is experimental, listing might be too.
    # We'll try to use the general api.list_repo_commits or similar if list_jobs fails
    # But let's check what experimental methods exist
    from huggingface_hub.utils import _experimental
    
    # Try common names
    for attr in dir(api):
        if "job" in attr.lower():
            print(f"Found attribute: {attr}")

    if hasattr(api, "list_jobs"):
        jobs = list(api.list_jobs())
        if jobs:
            for j in jobs[:20]: # Show last 20
                print(f"Job ID: {j.id}, Status: {j.status.stage}")
                if j.id == "69eda907d70108f37acdfaa3":
                    print(f"FOUND TARGET JOB! Status: {j.status}")
        else:
            print("No jobs found.")

except Exception as e:
    print(f"Error: {e}")
