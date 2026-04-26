import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

jid = "69edaabdd70108f37acdfad1"
try:
    info = api.inspect_job(job_id=jid)
    print(f"Job Status: {info.status}")
    print(f"Compute Flavor: {info.compute_flavor}")
except Exception as e:
    print(f"Error: {e}")
