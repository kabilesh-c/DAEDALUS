---
title: DAEDALUS Environment
emoji: 🏛
colorFrom: indigo
colorTo: blue
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
