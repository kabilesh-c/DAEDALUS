# DAEDALUS — Mechanism Design via Adversarial RL

> **OpenEnv Hackathon India 2026** | Themes: **#1 Multi-Agent Interactions** & **#4 Self-Improvement**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laksh718/DAEDALUS/blob/main/train_colab.ipynb)
[![HF Space — Environment](https://img.shields.io/badge/HF%20Space-Environment-blue?logo=huggingface)](https://huggingface.co/spaces/Laksh718/daedalus-env)
[![HF Space — Training](https://img.shields.io/badge/HF%20Space-Training-orange?logo=huggingface)](https://huggingface.co/spaces/Laksh718/daedalus-training-space)
[![Trained Model](https://img.shields.io/badge/Model-daedalus--designer-green?logo=huggingface)](https://huggingface.co/Laksh718/daedalus-designer)

**Live Environment (OpenEnv):** https://huggingface.co/spaces/Laksh718/daedalus-env  
**Training Pipeline:** https://huggingface.co/spaces/Laksh718/daedalus-training-space  
**Trained Designer Model:** https://huggingface.co/Laksh718/daedalus-designer

---

## The Problem: Training an LLM to Design Fair Markets

Most AI agents are **players** — they optimize within a fixed set of rules.  
**DAEDALUS** inverts this: it trains an LLM to be the **referee**.

The model learns to design auction mechanisms (pricing rules, information transparency, reserve prices, penalties) that steer a population of self-interested, adversarial agents toward socially optimal outcomes. This tests exactly the capabilities that current LLMs lack:

- **Equilibrium reasoning**: Predict what behavior a rule will produce in self-interested agents
- **Adversarial resilience**: Counter cartels, bid shading, strategic dropout in real time
- **Multi-objective tradeoff**: Jointly maximize Welfare, Fairness, Participation, and Stability — if any one collapses, the market fails

---

## Architecture

DAEDALUS implements a **Three-Layer POMDP** using the OpenEnv framework:

```
┌─────────────────────────────────────────────────────────┐
│  LLM Designer Agent (Qwen2.5-0.5B + LoRA)               │
│  Observes: mechanism config, last 20 market outcomes,    │
│  population proxies (bid correlation, dropout rate, ...)  │
│  Acts:     full mechanism JSON (11 fields)               │
└────────────────────────┬────────────────────────────────┘
                         │  mechanism config
┌────────────────────────▼────────────────────────────────┐
│  Market Simulator (5 rounds per step)                    │
│  5 adversarial sub-agent types:                          │
│   TruthfulBidder  — baseline honest participant          │
│   BidShader       — Bayes-Nash strategic bid suppression │
│   Colluder        — cartel rotation at reserve price     │
│   StrategicDropout — exits if surplus < outside option   │
│   BudgetExploiter — probes payment rule edge cases       │
└────────────────────────┬────────────────────────────────┘
                         │  (welfare, fairness, participation, stability)
┌────────────────────────▼────────────────────────────────┐
│  Composite Reward  R = W × F × P × S × anti_collusion   │
│  Five independent signals via OpenEnv Rubric system      │
└─────────────────────────────────────────────────────────┘
```

### Why the multiplicative reward matters

A purely additive reward lets an agent game individual components.  
With `R = W × F × P × S`, **all four must be non-zero simultaneously** — the agent cannot sacrifice fairness for welfare, or stability for participation. This is what makes the problem genuinely hard.

---

## OpenEnv Compliance

```python
from openenv.core.env_client import EnvClient

client = EnvClient(base_url="https://Laksh718-daedalus-env.hf.space")
obs = client.reset(seed=42).sync()

obs = client.step({
    "auction_type": "second_price",
    "reserve_price": 0.15,
    "collusion_penalty": 1.5,
    "coalition_policy": "penalize_suspected",
    "reveal_winner_identity": False,
    "reveal_clearing_price": True,
}).sync()

print(f"reward={obs.reward:.4f}  done={obs.done}")
```

- Built on `openenv-core>=0.2.3`
- `DaedalusOpenEnv` inherits from `openenv.core.env_server.interfaces.Environment`
- `DaedalusAction`, `DaedalusObservation`, `DaedalusState` inherit from OpenEnv Pydantic base types
- `DaedalusCompositeRubric` uses OpenEnv's composable `Rubric` system
- Valid `openenv.yaml` manifest included

---

## Training: Unsloth + TRL GRPO

Two-phase training pipeline:

**Phase 1 — SFT** (schema warmup)  
Synthetic `(observation, valid_mechanism_json)` pairs teach the output format. Prevents the GRPO phase from spending compute on malformed JSON.

**Phase 2 — GRPO** (reinforcement)  
Five reward signals on the same LoRA adapter:

| Signal | Formula | What it measures |
|---|---|---|
| `reward_format` | schema coverage ∈ [−1, 1] | JSON field completeness |
| `reward_welfare` | W = Σsurplus / Σvaluation | Allocative efficiency |
| `reward_fairness` | 1 − Gini(surplus) | Payment equity |
| `reward_stability` | 1 − 3σ(welfare) | Consistency over time |
| `reward_composite` | W × F × P × S × anti_collusion | Full objective |

### Training Modes

| `TRAIN_MODE` | SFT examples | GRPO steps | ~Time (A10G) |
|---|---|---|---|
| `smoke` | 24 | 4 | ~3 min |
| `short` | 320 | 120 | ~15 min |
| `long` | 800 | 300 | ~35 min |
| `full` | 2000 | 500 | ~90 min |

### Run in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laksh718/DAEDALUS/blob/main/train_colab.ipynb)

The Colab notebook runs the full `long` training on a T4/A100 and pushes the model to `Laksh718/daedalus-designer`.

### Run as a HF Job (4×L4)

```bash
export HF_TOKEN="hf_..."
python submit_job.py
```

---

## Results & Training Evidence

After GRPO training the designer consistently discovers **Targeted Opacity + Adaptive Reserves**:
- Hides winner identity → breaks cartel rotation
- Dynamic reserve near 0.10-0.15 → prevents strategic dropout  
- Second-price + collusion penalty → incentive-compatible bids

### Reward Curves

![Composite Reward](plots/reward_curve.png)  
*Composite reward R = W×F×P×S over 300 GRPO steps. Smoothed (solid) vs raw (transparent).*

![Per-Component Signals](plots/component_curves.png)  
*Individual reward components. Each must stay non-zero for the composite to improve.*

**Before training** (untrained Qwen2.5-0.5B): avg composite reward ≈ 0.12–0.18  
**After training** (300 GRPO steps): avg composite reward ≈ 0.50–0.62

Training history JSON is pushed to the model repo at:  
https://huggingface.co/Laksh718/daedalus-designer/blob/main/training_history.json

---

## 5-Stage Curriculum

The environment automatically escalates adversary difficulty:

| Stage | Sub-agent mix | Challenge |
|---|---|---|
| 0 | All truthful | Baseline — learn the format |
| 1 | + Shaders (30%) | Bid suppression begins |
| 2 | + Colluders (20%) | Cartel rotation |
| 3 | + Dropouts (15%) | Market liquidity pressure |
| 4 | + Exploiters (15%) | Payment rule probing |

---

## Repo Structure

```
DAEDALUS/
├── daedalus/
│   ├── openenv_env.py      # DaedalusOpenEnv (OpenEnv-compliant)
│   ├── openenv_models.py   # DaedalusAction / Observation / State (Pydantic)
│   ├── rubrics.py          # Composable OpenEnv Rubrics (W/F/P/S)
│   ├── env.py              # Market simulator (5-round auction)
│   ├── agents.py           # 5 adversarial sub-agent behaviors
│   └── rewards.py          # Raw reward computation
├── train_hf.py             # Training script (SFT + GRPO, Unsloth)
├── train_colab.ipynb       # Colab notebook for judges to re-run
├── job_runner.py           # HF Jobs entry point (bootstraps deps)
├── submit_job.py           # Submit training job to HF (4×L4)
├── openenv_app.py          # OpenEnv HTTP server (create_app factory)
├── server.py               # FastAPI demo dashboard
├── make_plots.py           # Generate reward / loss plots from history
├── openenv.yaml            # OpenEnv manifest
├── deploy_env_space.py     # Deploy environment Space
└── deploy_training_space.py # Deploy training Space
```

---

## Use the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model = AutoModelForCausalLM.from_pretrained("Laksh718/daedalus-designer")
tok   = AutoTokenizer.from_pretrained("Laksh718/daedalus-designer")

prompt = """You are a mechanism designer for a market auction system.
Analyze the current market state and design an optimal mechanism.

Round: 10 / 50
Curriculum Stage: 2

Your goal is to maximize the composite reward R = W x F x P x S

Respond with ONLY a JSON mechanism configuration."""

inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

*DAEDALUS — Designing the Institution, not just playing the game.*
