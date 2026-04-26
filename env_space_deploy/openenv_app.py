"""
DAEDALUS OpenEnv HTTP server.

This is the canonical entry point for hosting the DAEDALUS environment
on Hugging Face Spaces (or any Docker host). It uses OpenEnv's official
`create_app` factory so the resulting FastAPI service is fully
compatible with OpenEnv's `EnvClient` and the `openenv push` CLI.

Run locally:
    uvicorn openenv_app:app --host 0.0.0.0 --port 7860

Run from a client (judges, users):
    from openenv.core.env_client import EnvClient
    client = EnvClient(base_url="https://kabilesh-c-daedalus-env.hf.space")
    obs = client.reset(seed=42).sync()
    obs = client.step({"auction_type": "second_price", "reserve_price": 0.15, ...}).sync()
"""

from __future__ import annotations

from openenv.core.env_server import create_app

from daedalus.openenv_env import DaedalusOpenEnv
from daedalus.openenv_models import DaedalusAction, DaedalusObservation


# IMPORTANT: pass the *class* (factory), NOT an instance, so each
# WebSocket session gets its own DaedalusOpenEnv (per OpenEnv docs).
app = create_app(
    DaedalusOpenEnv,
    DaedalusAction,
    DaedalusObservation,
    env_name="daedalus",
)
