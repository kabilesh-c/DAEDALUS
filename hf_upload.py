import os
import sys
try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("huggingface_hub not found. Please install it.")
    sys.exit(1)

TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "kabilesh-c/daedalus"

if not TOKEN:
    print(
        "ERROR: set HF_TOKEN env var first  (PowerShell: $env:HF_TOKEN = 'hf_xxx')",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Logging in to Hugging Face...")
login(token=TOKEN)

api = HfApi()

print(f"Ensuring Space {REPO_ID} exists...")
try:
    api.create_repo(
        repo_id=REPO_ID, 
        repo_type="space", 
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
    print(f"Space {REPO_ID} is ready.")
except Exception as e:
    print(f"Issue creating space: {e}")

print(f"Uploading files to Space...")
try:
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=[
            "*.pyc", 
            "__pycache__/*", 
            ".git/*", 
            "hf_upload.py",
            "daedalus-checkpoints/*"
        ]
    )
    print("Upload successful!")
    print(f"Your DAEDALUS dashboard is deploying at: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"Upload failed: {e}")
