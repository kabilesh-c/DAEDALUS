"""
Deploy the DAEDALUS *Dashboard* as a Hugging Face Space.

This script bundles the frontend (index.html, app.js, styles.css) and the 
FastAPI backend (server.py) into a Docker Space with GPU support.
"""

from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = os.environ.get("DAEDALUS_DASHBOARD_SPACE", "kabilesh-c/daedalus-dashboard")
HARDWARE = os.environ.get("DAEDALUS_DASHBOARD_HARDWARE", "t4-small")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "dashboard_deploy"

DOCKERFILE = """FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git build-essential curl ca-certificates && \\
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
torch>=2.1.0
transformers>=4.45.0
accelerate>=0.30.0
huggingface_hub>=0.24.0
python-dotenv>=1.0.0
"""

def build():
    DEPLOY_DIR.mkdir(exist_ok=True)
    for f in ["server.py", "index.html", "app.js", "styles.css"]:
        shutil.copy(ROOT / f, DEPLOY_DIR / f)
    
    target_pkg = DEPLOY_DIR / "daedalus"
    if target_pkg.exists(): shutil.rmtree(target_pkg)
    shutil.copytree(ROOT / "daedalus", target_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    (DEPLOY_DIR / "Dockerfile").write_text(DOCKERFILE)
    (DEPLOY_DIR / "requirements.txt").write_text(REQUIREMENTS)
    (DEPLOY_DIR / "README.md").write_text(f"---\ntitle: Daedalus Dashboard\nsdk: docker\nhardware: {HARDWARE}\n---")

def main():
    api = HfApi(token=HF_TOKEN)
    build()
    print(f"Deploying to {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(DEPLOY_DIR), repo_id=REPO_ID, repo_type="space")
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=HF_TOKEN)
    try:
        api.request_space_hardware(repo_id=REPO_ID, hardware=HARDWARE)
    except:
        pass
    print(f"Live at: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    main()
