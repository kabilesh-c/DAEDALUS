"""
DAEDALUS Training Script — Colab-compatible
============================================
Train an LLM to design market mechanisms using GRPO via TRL + Unsloth.

This script is designed to run in Google Colab with a T4 GPU.

Usage:
    1. Open in Colab
    2. Run all cells
    3. Results saved to ./daedalus-checkpoints/

Themes: #1 Multi-Agent Interactions, #4 Self-Improvement (curriculum)
"""

# ═══════════════════════════════════════════════════════
# Step 0: Install Dependencies
# ═══════════════════════════════════════════════════════
# !pip install -q openenv trl unsloth transformers datasets accelerate
# !pip install -q torch --index-url https://download.pytorch.org/whl/cu121
# !pip install -q fastapi uvicorn pydantic

import json
import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════
# Step 1: Import DAEDALUS Environment
# ═══════════════════════════════════════════════════════
import sys
sys.path.insert(0, '.')

from daedalus.env import DaedalusEnvironment
from daedalus.models import MechanismConfig, Observation
from daedalus.rewards import (
    reward_welfare, reward_fairness, reward_participation,
    reward_stability, reward_composite, REWARD_FUNCTIONS,
)


# ═══════════════════════════════════════════════════════
# Step 2: Verify Environment Works
# ═══════════════════════════════════════════════════════
def test_environment():
    """Quick sanity check that the environment runs correctly."""
    print("=" * 60)
    print("DAEDALUS Environment Test")
    print("=" * 60)

    env = DaedalusEnvironment(episode_length=10)
    obs = env.reset()
    print(f"Reset OK. Round: {obs['round_number']}")
    print(f"Active agents: {obs['population_proxies']['active_count']}")

    total_reward = 0
    for i in range(10):
        # Test with different mechanism configs
        actions = [
            {"auction_type": "second_price", "reserve_price": 0.1},
            {"auction_type": "first_price", "reserve_price": 0.15},
            {"auction_type": "vcg", "reserve_price": 0.1, "collusion_penalty": 1.5},
        ]
        action = actions[i % len(actions)]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: R={reward:.4f} W={info['welfare_ratio']:.3f} "
              f"F={1-info['gini_coefficient']:.3f} P={info['participation_rate']:.3f} "
              f"Active={info['active_agents']}")
        if done:
            print("  Episode terminated.")
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Average reward: {total_reward/(i+1):.4f}")
    print("Environment test PASSED ✓")
    return True


# ═══════════════════════════════════════════════════════
# Step 3: Build Training Dataset (Prompts)
# ═══════════════════════════════════════════════════════
def generate_training_prompts(n_prompts: int = 500) -> List[Dict[str, str]]:
    """
    Generate training prompts by running the environment
    and converting observations to natural language prompts.
    """
    print(f"\nGenerating {n_prompts} training prompts...")
    prompts = []
    env = DaedalusEnvironment()

    for i in range(n_prompts):
        obs_dict = env.reset()
        obs = Observation(**obs_dict) if isinstance(obs_dict, dict) else obs_dict

        # Convert observation to natural language prompt
        if hasattr(obs, 'to_prompt'):
            prompt_text = obs.to_prompt()
        else:
            prompt_text = _format_prompt(obs_dict)

        prompts.append({
            "prompt": prompt_text,
            "obs": obs_dict,
        })

        # Also generate mid-episode prompts (more diverse states)
        for step in range(random.randint(1, 5)):
            action = _random_mechanism()
            obs_dict, _, done, _ = env.step(action)
            if done:
                break

            if hasattr(obs_dict, 'to_prompt'):
                prompt_text = obs_dict.to_prompt()
            else:
                prompt_text = _format_prompt(obs_dict)

            prompts.append({
                "prompt": prompt_text,
                "obs": obs_dict,
            })

    print(f"Generated {len(prompts)} prompts")
    return prompts


def _format_prompt(obs: dict) -> str:
    """Format observation dict into a natural language prompt."""
    lines = [
        "You are a mechanism designer for a market auction system.",
        "Analyze the current market state and design an optimal mechanism.",
        "",
        f"Round: {obs.get('round_number', 0)} / {obs.get('episode_length', 50)}",
        "",
        "Your goal is to maximize the composite reward R = W × F × P × S where:",
        "  W = Social Welfare (allocative efficiency)",
        "  F = Fairness (1 - Gini coefficient)",
        "  P = Participation Rate",
        "  S = Stability (consistency over time)",
        "",
    ]

    # Add market outcomes
    outcomes = obs.get('market_outcomes', [])
    if outcomes:
        lines.append("Recent Market Outcomes:")
        for o in outcomes[-5:]:
            lines.append(
                f"  W={o.get('welfare_ratio', 0):.3f} "
                f"F={1-o.get('gini_coefficient', 0):.3f} "
                f"P={o.get('participation_rate', 1):.3f} "
                f"R={o.get('composite_reward', 0):.3f}"
            )

    # Add population signals
    proxies = obs.get('population_proxies', {})
    if proxies:
        lines.extend([
            "",
            f"Active Bidders: {proxies.get('active_count', 8)}/{proxies.get('total_agents', 8)}",
            f"Bid Correlation (collusion signal): {proxies.get('bid_correlation', 0):.3f}",
            f"Dropout Rate: {proxies.get('dropout_rate', 0):.3f}",
        ])

    lines.extend([
        "",
        "Respond with ONLY a JSON mechanism configuration, no explanation:",
        '{"auction_type": "first_price|second_price|vcg", "reserve_price": float(0-0.9), '
        '"reveal_reserve": bool, "reveal_competing_bids": bool, "reveal_winner_identity": bool, '
        '"reveal_clearing_price": bool, "reveal_bid_distribution": bool, '
        '"shill_penalty": float(0-3), "withdrawal_penalty": float(0-3), "collusion_penalty": float(0-3), '
        '"coalition_policy": "allow|restrict|penalize_suspected|penalize_confirmed"}',
    ])

    return "\n".join(lines)


def _random_mechanism() -> dict:
    """Generate a random mechanism configuration."""
    return {
        "auction_type": random.choice(["first_price", "second_price", "vcg"]),
        "reserve_price": random.uniform(0.05, 0.5),
        "reveal_reserve": random.random() > 0.5,
        "reveal_competing_bids": random.random() > 0.7,
        "reveal_winner_identity": random.random() > 0.5,
        "reveal_clearing_price": random.random() > 0.3,
        "reveal_bid_distribution": random.random() > 0.7,
        "shill_penalty": random.uniform(0, 2),
        "withdrawal_penalty": random.uniform(0, 1),
        "collusion_penalty": random.uniform(0, 2),
        "coalition_policy": random.choice(["allow", "restrict", "penalize_suspected"]),
    }


# ═══════════════════════════════════════════════════════
# Step 4: Define Reward Functions for TRL
# ═══════════════════════════════════════════════════════
def evaluate_mechanism(generated_text: str, env: DaedalusEnvironment = None) -> Dict[str, float]:
    """
    Evaluate a generated mechanism configuration against the environment.
    Returns individual reward components.
    """
    if env is None:
        env = DaedalusEnvironment()
        env.reset()

    # Parse generated mechanism JSON
    try:
        # Extract JSON from model output
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            mechanism_json = generated_text[json_start:json_end]
            mechanism = json.loads(mechanism_json)
        else:
            return {"welfare": 0.0, "fairness": 0.0, "participation": 0.0,
                    "stability": 0.0, "composite": 0.0, "format_penalty": -0.5}
    except (json.JSONDecodeError, KeyError):
        return {"welfare": 0.0, "fairness": 0.0, "participation": 0.0,
                "stability": 0.0, "composite": 0.0, "format_penalty": -0.5}

    # Run in environment
    obs, reward, done, info = env.step(mechanism)

    return {
        "welfare": info.get("welfare_ratio", 0.0),
        "fairness": 1.0 - info.get("gini_coefficient", 0.0),
        "participation": info.get("participation_rate", 1.0),
        "stability": info.get("stability_score", 0.8),
        "composite": reward,
        "format_penalty": 0.0,  # Valid JSON = no penalty
    }


# ═══════════════════════════════════════════════════════
# Step 5: GRPO Training Setup
# ═══════════════════════════════════════════════════════
def setup_training():
    """
    Set up and run GRPO training using TRL + Unsloth.
    Requires GPU (T4 or better).
    """
    print("\n" + "=" * 60)
    print("DAEDALUS GRPO Training Setup")
    print("=" * 60)

    # ── Load Model with Unsloth ──
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Qwen2.5-7B-Instruct",
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        print("✓ Model loaded: Qwen2.5-7B-Instruct (4-bit)")

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print("✓ LoRA applied: rank=16, alpha=16")

    except ImportError:
        print("⚠ Unsloth not available. Using transformers directly.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        print("✓ Model loaded: Qwen2.5-7B-Instruct (auto)")

    # ── Generate Training Prompts ──
    prompts = generate_training_prompts(n_prompts=200)
    print(f"✓ Training prompts generated: {len(prompts)}")

    # ── Create Dataset ──
    from datasets import Dataset
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in prompts])
    print(f"✓ Dataset created: {len(dataset)} examples")

    # ── Define Reward Functions ──
    env = DaedalusEnvironment()

    def grpo_reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """GRPO reward function — evaluates mechanism configs against environment."""
        rewards = []
        for completion in completions:
            env.reset()
            result = evaluate_mechanism(completion, env)
            # Composite reward with format bonus
            reward = result["composite"] + result["format_penalty"]
            rewards.append(reward)
        return rewards

    # ── Configure GRPO ──
    try:
        from trl import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            output_dir="./daedalus-checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_generations=8,
            max_completion_length=512,
            learning_rate=5e-6,
            logging_steps=10,
            save_steps=200,
            bf16=True,
            report_to="none",
        )
        print("✓ GRPO config created")

        trainer = GRPOTrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            reward_funcs=[grpo_reward_fn],
            train_dataset=dataset,
        )
        print("✓ GRPO trainer initialized")

        # ── Train ──
        print("\n🚀 Starting GRPO training...")
        trainer.train()
        print("✓ Training complete!")

        # ── Save ──
        try:
            model.save_pretrained_merged(
                "daedalus-trained",
                tokenizer,
                save_method="merged_16bit",
            )
            print("✓ Model saved: daedalus-trained/")
        except AttributeError:
            model.save_pretrained("daedalus-trained")
            tokenizer.save_pretrained("daedalus-trained")
            print("✓ Model saved (adapter only): daedalus-trained/")

    except ImportError:
        print("⚠ TRL not available. Install with: pip install trl")
        print("  Training setup is ready — install TRL and re-run.")


# ═══════════════════════════════════════════════════════
# Step 6: Evaluation & Benchmarking
# ═══════════════════════════════════════════════════════
def run_evaluation():
    """
    Evaluate the environment with different mechanism strategies.
    Produces baseline metrics for before/after comparison.
    """
    print("\n" + "=" * 60)
    print("DAEDALUS Evaluation Benchmark")
    print("=" * 60)

    strategies = {
        "Random": lambda: _random_mechanism(),
        "Second-Price Default": lambda: {
            "auction_type": "second_price", "reserve_price": 0.1,
            "reveal_clearing_price": True, "coalition_policy": "allow",
        },
        "VCG + Anti-Collusion": lambda: {
            "auction_type": "vcg", "reserve_price": 0.1,
            "reveal_clearing_price": True, "reveal_winner_identity": False,
            "collusion_penalty": 2.0, "coalition_policy": "penalize_suspected",
        },
        "Aggressive Reserve": lambda: {
            "auction_type": "second_price", "reserve_price": 0.4,
            "reveal_clearing_price": True,
        },
    }

    results = {}
    n_episodes = 20

    for name, strategy_fn in strategies.items():
        episode_rewards = []
        episode_welfare = []
        episode_fairness = []
        episode_participation = []

        for ep in range(n_episodes):
            env = DaedalusEnvironment(episode_length=20)
            env.reset()

            ep_reward = 0
            last_info = {}
            for step in range(20):
                action = strategy_fn()
                _, reward, done, info = env.step(action)
                ep_reward += reward
                last_info = info
                if done:
                    break

            episode_rewards.append(ep_reward / (step + 1))
            episode_welfare.append(last_info.get("welfare_ratio", 0))
            episode_fairness.append(1 - last_info.get("gini_coefficient", 0))
            episode_participation.append(last_info.get("participation_rate", 1))

        avg_r = np.mean(episode_rewards)
        min_r = np.min(episode_rewards)
        results[name] = {
            "avg_reward": avg_r,
            "min_reward": min_r,
            "avg_welfare": np.mean(episode_welfare),
            "avg_fairness": np.mean(episode_fairness),
            "avg_participation": np.mean(episode_participation),
        }

        print(f"\n  {name}:")
        print(f"    Avg Reward: {avg_r:.4f}  (min: {min_r:.4f})")
        print(f"    Welfare:    {results[name]['avg_welfare']:.4f}")
        print(f"    Fairness:   {results[name]['avg_fairness']:.4f}")
        print(f"    Participation: {results[name]['avg_participation']:.4f}")

    return results


# ═══════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Test environment
    test_environment()

    # 2. Run baseline evaluation
    baseline_results = run_evaluation()

    # 3. Train (uncomment when GPU available)
    # setup_training()

    print("\n" + "=" * 60)
    print("DAEDALUS Pipeline Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run setup_training() in Colab with GPU")
    print("  2. Compare trained model against baseline results")
    print("  3. Deploy to HuggingFace Space with: openenv push")
