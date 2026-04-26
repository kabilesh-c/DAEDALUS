"""
Deploy the DAEDALUS *environment* as its own Hugging Face Space.

This Space hosts the OpenEnv-compliant `DaedalusOpenEnv` over HTTP, so
judges (and any external user) can connect via the OpenEnv `EnvClient`:

    from openenv.core.env_client import EnvClient
    client = EnvClient(base_url="https://kabilesh-c-daedalus-env.hf.space")
    obs = client.reset(seed=42).sync()

This is intentionally separate from the training Space
(`kabilesh-c/daedalus-training-space`), which is a one-shot trainer
that auto-pauses on completion. The environment Space is a long-lived
service (`hardware: cpu-basic` is fine; the env is pure Python).

Usage (PowerShell):
    $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    python deploy_env_space.py
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


REPO_ID = os.environ.get("DAEDALUS_ENV_SPACE", "kabilesh-c/daedalus-env")
HARDWARE = os.environ.get("DAEDALUS_ENV_HARDWARE", "cpu-basic")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN environment variable is not set.\n"
        "  PowerShell:  $env:HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxxxx'\n"
        "  Then re-run: python deploy_env_space.py",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "env_space_deploy"

DOCKERFILE = """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "openenv_app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """openenv-core>=0.2.3
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
"""

README = """---
title: DAEDALUS Environment
emoji: \U0001f3db
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
hardware: cpu-basic
---

# DAEDALUS Environment - OpenEnv-compliant HTTP Service

OpenEnv-compliant environment for **mechanism design via adversarial RL**.

An LLM designer agent picks auction rules; a heterogeneous population of
adaptive sub-agents (truthful bidders, shaders, colluders, strategic
dropouts, budget exploiters) probes the mechanism for exploits. The
composite reward
```
R = welfare * fairness * participation * stability
```
forces multi-objective tradeoffs that single-objective gaming cannot
satisfy.

**Hackathon themes:**
- #1 Multi-Agent Interactions (primary)
- #4 Self-Improvement (secondary, via 5-stage curriculum)

## Use it

```python
from openenv.core.env_client import EnvClient

client = EnvClient(base_url="https://kabilesh-c-daedalus-env.hf.space")
obs = client.reset(seed=42).sync()
obs = client.step({
    "auction_type": "second_price",
    "reserve_price": 0.15,
    "collusion_penalty": 1.5,
    "coalition_policy": "penalize_suspected",
}).sync()
print(obs.reward, obs.done)
```

## Trained designer

A LoRA adapter trained against this environment lives at
[`kabilesh-c/daedalus-designer`](https://huggingface.co/kabilesh-c/daedalus-designer)
(Qwen2.5-0.5B-Instruct + SFT warmup + GRPO refinement, see the training
Space [`kabilesh-c/daedalus-training-space`](https://huggingface.co/spaces/kabilesh-c/daedalus-training-space)).

## OpenEnv API

The server exposes the standard OpenEnv HTTP/WebSocket endpoints:

- `POST /reset`      - start a new episode
- `POST /step`       - apply a `DaedalusAction`, get a `DaedalusObservation` back
- `GET  /state`      - inspect server-side episode state
- `GET  /metadata`   - environment description
- `GET  /schema`     - JSON schemas for action / observation / state
- `GET  /health`     - health probe
- `WS   /ws`         - WebSocket session (concurrent sessions enabled)
"""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def p(msg: str) -> None:
    print(msg, flush=True)


def build_deploy_dir() -> None:
    p(f"[1/6] preparing {DEPLOY_DIR.name}/ ...")
    DEPLOY_DIR.mkdir(exist_ok=True)

    shutil.copy(ROOT / "openenv_app.py", DEPLOY_DIR / "openenv_app.py")

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
    (DEPLOY_DIR / "README.md").write_text(README, encoding="utf-8")

    sha = file_sha256(DEPLOY_DIR / "openenv_app.py")
    p(f"        wrote {DEPLOY_DIR}")
    p(f"        openenv_app.py sha256: {sha[:16]}...")


def main() -> None:
    api = HfApi(token=HF_TOKEN)
    me = api.whoami()
    p(f"[auth] authenticated as: {me.get('name', '?')}")

    build_deploy_dir()

    p(f"[2/6] ensuring Space {REPO_ID} exists ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    p(f"[3/6] uploading files to Space ...")
    commit_info = api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo=".",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message="deploy DAEDALUS OpenEnv environment",
    )
    p(f"        upload commit: {getattr(commit_info, 'oid', commit_info)}")

    p(f"[4/6] requesting hardware: {HARDWARE} ...")
    try:
        api.request_space_hardware(repo_id=REPO_ID, hardware=HARDWARE)
        p(f"        {HARDWARE} requested")
    except Exception as e:
        p(f"        WARNING: hardware request failed: {e}")

    p(f"[5/6] restarting Space (resumes if paused, rebuilds image) ...")
    try:
        api.restart_space(repo_id=REPO_ID)
        p("        restart triggered")
    except Exception as e:
        p(f"        WARNING: restart failed: {e}")

    space_url = f"https://huggingface.co/spaces/{REPO_ID}"
    api_url = f"https://{REPO_ID.replace('/', '-')}.hf.space"
    p("")
    p("=" * 64)
    p(f"Space:        {space_url}")
    p(f"Build logs:   {space_url}?logs=container")
    p(f"OpenEnv URL:  {api_url}")
    p(f"Health:       {api_url}/health")
    p(f"Schema:       {api_url}/schema")
    p("=" * 64)
    p(
        "Once the build finishes (~2-3 min for cpu-basic, no GPU), connect with:\n"
        f"    from openenv.core.env_client import EnvClient\n"
        f"    client = EnvClient(base_url=\"{api_url}\")\n"
        f"    obs = client.reset(seed=42).sync()"
    )


if __name__ == "__main__":
    main()
