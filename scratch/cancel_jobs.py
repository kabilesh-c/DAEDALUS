import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# IDs of old jobs that are stuck or we want to replace
OLD_JOB_IDS = ["69eda907d70108f37acdfaa3", "69edaabdd70108f37acdfad1"]

for jid in OLD_JOB_IDS:
    try:
        print(f"Cancelling job {jid}...")
        api.cancel_job(jid)
    except Exception as e:
        print(f"Failed to cancel {jid}: {e}")

print("Done cancelling. Now run submit_job.py again.")
