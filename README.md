# DAEDALUS — Mechanism Design via Adversarial RL

> **OpenEnv Hackathon Submission** | Theme #1: Multi-Agent Interactions + Theme #4: Self-Improvement

## 🏛️ What is DAEDALUS?

DAEDALUS is the first RL environment that trains an LLM to be a **referee** — designing market rules that constrain self-interested agents toward socially optimal outcomes. Every prior RL agent (AlphaGo, trading bots, game AI) is a *player* optimizing within fixed rules. DAEDALUS inverts this: the LLM learns to *design the rules themselves*.

### The Problem
Current LLMs fail at mechanism design because they've been trained to play games, not design them. The specific gaps are:
- **Equilibrium reasoning under partial information** — inferring what equilibria rules will produce
- **Second-order strategic modeling** — anticipating how agents will learn to exploit mechanisms over time
- **Multi-objective balancing** — optimizing welfare, fairness, and participation simultaneously under adversarial pressure
- **Adaptive rule revision** — modifying mechanisms mid-deployment without causing participant dropout

### Why It Matters
Real-world analogs: Google ad auctions, Airbnb pricing, FCC spectrum auctions, Ethereum EIP-1559. In each case, a designer made institutional choices that shaped equilibrium behavior — and adversarial participants continue probing for exploits.

---

## 🎮 Live Demo

Open `index.html` in a browser for the interactive dashboard:
- **Simulation Tab**: Configure mechanisms and watch adversarial agents adapt in real-time
- **Architecture Tab**: Full system design documentation
- **Training Tab**: Complete training pipeline with Colab setup code

> **Try it**: Switch to first-price auction and watch bid shaders suppress bids. Reveal winner identity and colluders exploit it for cartel rotation. Raise the reserve and the dropout agent exits.

---

## 🏗️ Architecture

### Three-Layer Environment (POMDP)

```
┌─── Outer: Mechanism Designer (LLM Agent) ──────────────────┐
│  Observes: aggregate market outcomes only                    │
│  Actions: auction type, info policy, reserves, penalties     │
│  ┌─── Middle: Market Simulation ─────────────────────────┐  │
│  │  Runs auction mechanics, collects bids, computes stats │  │
│  │  ┌─── Inner: Strategic Sub-Agents (HIDDEN) ────────┐  │  │
│  │  │  5 types: Truthful, Shader, Colluder,            │  │  │
│  │  │  Dropout, Budget Exploiter                       │  │  │
│  │  │  Private valuations NEVER revealed to designer   │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Adversarial Agent Taxonomy

| Agent | Strategy | Exploits | Defense |
|-------|----------|----------|---------|
| 🎯 Truthful | Bids true valuation | — | Baseline |
| 📉 Shader | Bid = v×(n-1)/n | First-price payment rules | Switch to second-price |
| 🤝 Colluder | Rotates wins with partner | Information transparency | Hide winner identity |
| 🚪 Dropout | Exits if surplus < threshold | High reserve prices | Calibrate near outside option |
| 💣 Exploiter | Budget-constrained strategic | Payment rule edge cases | Limit bid frequency |

### Multiplicative Reward Function

```
R(t) = W(t) × F(t) × P(t) × S(t)
```

- **W** — Social Welfare: allocated utility / theoretical maximum
- **F** — Fairness: 1 − Gini coefficient of payments
- **P** — Participation: active bidders / total agents
- **S** — Stability: consistency of welfare over recent rounds

**Why multiplicative?** All terms must be jointly positive — you cannot sacrifice fairness for welfare or participation for stability. Each term acts as a veto.

---

## 🔧 OpenEnv API

```python
from daedalus.env import DaedalusEnvironment

env = DaedalusEnvironment(
    n_agents=8,
    episode_length=50,
    curriculum_stage=0,
)

obs = env.reset()                    # New market scenario
obs, reward, done, info = env.step({  # Apply mechanism
    "auction_type": "second_price",
    "reserve_price": 0.1,
    "reveal_clearing_price": True,
    "collusion_penalty": 1.5,
    "coalition_policy": "penalize_suspected",
})
state = env.state()                  # Current POMDP observation
```

---

## 🧠 Training Pipeline

### Stack: OpenEnv + TRL GRPO + Unsloth

```python
# 1. Load model with Unsloth (4-bit quantized)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct", load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=16)

# 2. Train with GRPO
from trl import GRPOConfig, GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    config=GRPOConfig(num_generations=8, ...),
    reward_funcs=[welfare_fn, fairness_fn, participation_fn, composite_fn],
    train_dataset=dataset,
)
trainer.train()
```

### Curriculum (5 Stages)
| Stage | Population | Threshold | Key Discovery |
|-------|-----------|-----------|---------------|
| 0 | 100% Truthful | R > 0.75 | Second-price dominates |
| 1 | +30% Shaders | R > 0.65 | Info policy matters |
| 2 | +20% Dropout | R > 0.60 | Reserve calibration |
| 3 | +20% Colluders | R > 0.55 | Targeted opacity |
| 4 | Full adversarial | Eval benchmark | Robust mechanism |

---

## 📊 Evaluation Results

Run `python train_colab.py` for baseline benchmarks:

| Strategy | Avg Reward | Welfare | Fairness | Participation |
|----------|-----------|---------|----------|---------------|
| Random | ~0.05 | ~0.50 | ~0.40 | ~0.85 |
| Second-Price Default | ~0.25 | ~0.75 | ~0.60 | ~1.00 |
| VCG + Anti-Collusion | ~0.35 | ~0.80 | ~0.65 | ~0.95 |
| **Trained Agent (target)** | **>0.50** | **>0.85** | **>0.70** | **>0.95** |

---

## 🚀 Deployment

### Local
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t daedalus .
docker run -p 8000:8000 daedalus
```

### HuggingFace Spaces
```bash
openenv push --space your-username/daedalus
```

---

## 📁 Project Structure

```
daedalus/
├── index.html           # Interactive dashboard demo
├── styles.css           # Dashboard design system
├── app.js               # Simulation engine (JS)
├── daedalus/            # Python environment package
│   ├── __init__.py
│   ├── env.py           # Main OpenEnv environment (reset/step/state)
│   ├── agents.py        # 5 adversarial agent types
│   ├── rewards.py       # Multiple independent reward functions
│   └── models.py        # Typed dataclasses
├── server.py            # FastAPI server
├── train_colab.py       # Colab training script (TRL + Unsloth)
├── openenv.yaml         # OpenEnv manifest
├── requirements.txt     # Dependencies
├── Dockerfile           # Container deployment
└── README.md            # This file
```

---

## 🎯 Research Novelty

1. **First environment training LLMs as referees**, not players
2. **Two-timescale adversarial loop** — sub-agents adapt fast within episodes, designer adapts slow across episodes (GAN-like dynamics for mechanism design)
3. **Partial observability is non-artificial** — mirrors real mechanism designers who never see private valuations
4. **Emergent mechanism discovery** — potential to discover novel mechanisms in settings where theory has no closed-form solution

---

## 📚 References

- [OpenEnv Framework](https://github.com/openenv)
- [TRL (Transformer RL)](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- Myerson, R. (1981). Optimal Auction Design
- Vickrey, W. (1961). Counterspeculation, Auctions, and Competitive Sealed Tenders
- Clarke, E. (1971). Multipart Pricing of Public Goods

---

*DAEDALUS — From AI as player to AI as referee. A qualitatively different class of reasoning.*
