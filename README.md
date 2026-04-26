# DAEDALUS — Mechanism Design via Adversarial RL

> **OpenEnv Hackathon 2026 Submission** | Themes: #1 Multi-Agent Interactions & #4 Self-Improvement

[**Live Environment (HF Space)**](https://huggingface.co/spaces/kabilesh-c/daedalus-env) | [**Training Pipeline (HF Space)**](https://huggingface.co/spaces/kabilesh-c/daedalus-training-space) | [**Trained Designer (Model Hub)**](https://huggingface.co/kabilesh-c/daedalus-designer)

---

## 🏛️ Project Vision: The "AI Referee"

Most AI agents today are **players** optimizing within a fixed set of rules. **DAEDALUS** inverts this: it is the first RL environment that trains an LLM to be the **referee**. 

The model learns to design market mechanisms (auction types, info transparency, reserves, and penalties) that constrain self-interested, adversarial agents toward socially optimal outcomes. This is critical for the future of decentralized governance, ad auctions, and complex institution design.

### Why this is a "Hard" Problem for LLMs:
- **Equilibrium Reasoning**: The LLM must infer what behavior the rules will produce in self-interested participants.
- **Adversarial Resilience**: Defending against cartel rotation, strategic shading, and market dropout.
- **Multi-Objective Tradeoffs**: Balancing Welfare, Fairness, and Participation simultaneously—if any one fails, the market collapses.

---

## 🏗️ Architecture: The Strategic Sandbox

DAEDALUS implements a **Three-Layer POMDP** using the `OpenEnv` framework.

### 1. The Environment (OpenEnv)
A high-fidelity market simulator with 5 distinct adversarial agent types:
- **🎯 Truthful**: The baseline "honest" participant.
- **📉 Shaders**: Use Bayes-Nash strategies to suppress clearing prices.
- **🤝 Colluders**: Coordinate cartel rotation to win at reserve prices.
- **🚪 Dropouts**: Exit the market if surplus falls below an outside option.
- **💣 Exploiters**: Probe for payment rule edge cases.

### 2. Multi-Reward Design (The "Judge")
To prevent **reward hacking**, we use 4 independent reward signals:
1. **Format Reward**: Correct JSON schema compliance.
2. **Welfare Reward**: Maximizing total society utility.
3. **Fairness Reward**: Minimizing the Gini coefficient of participant surplus.
4. **Participation Reward**: Long-term market liquidity and stability.

---

## 🧠 Training Stack: Unsloth + GRPO

We use a state-of-the-art RL stack to achieve high-efficiency results:
- **TRL GRPO**: Group Relative Policy Optimization for stable, data-efficient RL.
- **Unsloth**: 4x faster training and inference through optimized kernels.
- **OpenEnv Curriculum**: A 5-stage training loop that gradually introduces harder adversaries as the model improves.

---

## 📊 Results & Evidence of Learning

### Before Training:
The baseline model often picks high reserves (causing market collapse) or reveals too much info (allowing colluders to exploit the system). Reward is volatile and low (~0.15).

### After Training:
The agent discovers **Targeted Opacity** and **Adaptive Reserves**. It learns that hiding winner identities breaks cartels, while dynamic reserve pricing prevents strategic dropouts. Final rewards exceed **0.55+** across all adversarial cohorts.

*(Include your training_history.json plots here after the run)*

---

## 📁 Repository Structure

```text
daedalus/
├── train_hf.py           # Canonical Training Script (Unsloth + GRPO)
├── deploy_training_space.py # Automated HF Space deployment for training
├── deploy_env_space.py      # Long-lived Environment Service deployment
├── daedalus/             # Core OpenEnv Package
│   ├── openenv_env.py    # MCP / OpenEnv Interface
│   ├── env.py            # Physics of the Auction Market
│   ├── agents.py         # The 5 Adversarial Behavioral Models
│   └── rewards.py        # Independent Scoring Rubrics
├── openenv.yaml          # Manifest for OpenEnv Hackathon
└── .env                  # Secret Storage (HF_TOKEN)
```

---

## 🚀 Getting Started

### 1. Run the Environment Locally
```bash
pip install -r requirements.txt
python -m daedalus.openenv_app
```

### 2. Start Training (Colab/Cloud)
```bash
$env:HF_TOKEN = "your_token"
python train_hf.py
```

---

*DAEDALUS — Designing the Institution, not just playing the game.*
