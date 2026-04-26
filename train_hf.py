"""
DAEDALUS Training - Qwen2.5-0.5B + Unsloth + GRPO Multi-Reward (v4)
=====================================================================
Enhanced pipeline for the OpenEnv Hackathon:
    - Unsloth for 2x-4x faster RL training.
    - Decomposed Rewards (Format, Welfare, Fairness, Participation).
    - OpenEnv-compliant environment interaction.
    - Stage-based Curriculum.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import sys
import traceback
import functools
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# -------------------------------------------------------------------
# Auth
# -------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_ID = os.environ.get("SPACE_ID")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[auth] logged into Hugging Face from HF_TOKEN env var")
    except Exception as e:
        print(f"[auth] login failed: {e}")
else:
    print("[auth] WARNING: HF_TOKEN env var is not set; push_to_hub will fail")


# -------------------------------------------------------------------
# Unsloth & TRL Patching
# -------------------------------------------------------------------
# We patch before importing TRL to ensure Unsloth speedups are applied.
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("grpo", "unsloth/Qwen2.5-0.5B-Instruct")


# -------------------------------------------------------------------
# Make daedalus/ importable regardless of cwd
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (SCRIPT_DIR, "/workspace", "/app"):
    if os.path.isdir(os.path.join(candidate, "daedalus")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from daedalus import DaedalusOpenEnv, DaedalusAction  # noqa: E402
    USE_OPENENV = True
except Exception as _exc:  # noqa: BLE001
    print(f"[env] error loading DaedalusOpenEnv: {_exc}")
    USE_OPENENV = False


def _to_dict(obs: Any) -> dict:
    if isinstance(obs, dict): return obs
    if hasattr(obs, "model_dump"): return obs.model_dump()
    if hasattr(obs, "to_dict"): return obs.to_dict()
    return dict(obs)


def make_env(stage: int = 0) -> Any:
    return DaedalusOpenEnv(curriculum_stage=stage)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
TRAIN_MODE = os.environ.get("TRAIN_MODE", "short").lower()
MODEL_ID = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")
HUB_REPO = os.environ.get("HUB_MODEL_ID", "kabilesh-c/daedalus-designer")

if TRAIN_MODE == "long":
    N_SFT_EXAMPLES = 300
    SFT_EPOCHS = 2
    N_GRPO_PROMPTS = 200
    GRPO_STEPS = 120
else:
    N_SFT_EXAMPLES = 120
    SFT_EPOCHS = 1
    N_GRPO_PROMPTS = 80
    GRPO_STEPS = 40

SFT_OUT = "./sft-warmup"
SFT_MERGED = "./sft-merged"
GRPO_OUT = "./grpo-refined"

REQUIRED_KEYS = {
    "auction_type", "reserve_price", "reveal_reserve",
    "reveal_competing_bids", "reveal_winner_identity",
    "reveal_clearing_price", "reveal_bid_distribution",
    "shill_penalty", "withdrawal_penalty", "collusion_penalty",
    "coalition_policy"
}


# -------------------------------------------------------------------
# Prompt and synthetic data
# -------------------------------------------------------------------
def format_prompt(obs: dict) -> str:
    lines = [
        "You are a mechanism designer for a market auction system.",
        "Analyze the current market state and design an optimal mechanism.",
        "",
        f"Round: {obs.get('round_number', 0)} / {obs.get('episode_length', 50)}",
        "Curriculum Stage: " + str(obs.get('curriculum_stage', 0)),
        "",
        "Your goal is to maximize the composite reward R = W x F x P x S",
    ]
    outcomes = obs.get("market_outcomes", [])
    if outcomes:
        lines.append("Recent Market Outcomes:")
        for o in outcomes[-5:]:
            lines.append(
                f"  W={o.get('welfare_ratio', 0):.3f} "
                f"F={1 - o.get('gini_coefficient', 0):.3f} "
                f"P={o.get('participation_rate', 1):.3f} "
                f"R={o.get('composite_reward', 0):.3f}"
            )
    lines.extend([
        "",
        "Respond with ONLY a JSON mechanism configuration with these keys:",
        "  auction_type: \"first_price\" | \"second_price\" | \"vcg\"",
        "  reserve_price: float [0.0, 0.9]",
        "  reveal_reserve, reveal_competing_bids, reveal_winner_identity, reveal_clearing_price, reveal_bid_distribution: bool",
        "  shill_penalty, withdrawal_penalty, collusion_penalty: float [0.0, 3.0]",
        "  coalition_policy: \"allow\" | \"restrict\" | \"penalize_suspected\" | \"penalize_confirmed\"",
        "",
        "Output strictly a single JSON object, no commentary.",
    ])
    return "\n".join(lines)


def random_valid_mechanism() -> dict:
    return {
        "auction_type": random.choice(["first_price", "second_price", "vcg"]),
        "reserve_price": round(random.uniform(0.05, 0.5), 3),
        "reveal_reserve": random.choice([True, False]),
        "reveal_competing_bids": random.random() < 0.3,
        "reveal_winner_identity": random.choice([True, False]),
        "reveal_clearing_price": random.random() < 0.7,
        "reveal_bid_distribution": random.random() < 0.3,
        "shill_penalty": round(random.uniform(0.0, 2.0), 3),
        "withdrawal_penalty": round(random.uniform(0.0, 1.0), 3),
        "collusion_penalty": round(random.uniform(0.0, 2.0), 3),
        "coalition_policy": random.choice(["allow", "restrict", "penalize_suspected", "penalize_confirmed"]),
    }


def generate_sft_examples(n: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    # Mix stages in SFT
    for stage in range(5):
        n_stage = n // 5
        env = make_env(stage)
        while len(pairs) < (stage + 1) * n_stage:
            obs_dict = _to_dict(env.reset())
            for _ in range(3):
                mech = random_valid_mechanism()
                pairs.append({
                    "messages": [
                        {"role": "user", "content": format_prompt(obs_dict)},
                        {"role": "assistant", "content": json.dumps(mech)},
                    ]
                })
                obs = env.step(DaedalusAction(**mech))
                obs_dict = _to_dict(obs)
                if obs.done: break
    return pairs[:n]


def generate_grpo_prompts(n: int) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    for stage in range(5):
        n_stage = n // 5
        env = make_env(stage)
        while len(prompts) < (stage + 1) * n_stage:
            obs_dict = _to_dict(env.reset())
            prompts.append({"prompt": format_prompt(obs_dict)})
            for _ in range(3):
                obs = env.step(DaedalusAction(**random_valid_mechanism()))
                prompts.append({"prompt": format_prompt(_to_dict(obs))})
                if obs.done: break
    return prompts[:n]


# -------------------------------------------------------------------
# Decomposed Reward Functions
# -------------------------------------------------------------------
_reward_env = make_env()

@functools.lru_cache(maxsize=128)
def _get_env_outcome(text: str) -> Optional[dict]:
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start < 0 or j_end <= j_start: return None
    try:
        mech = json.loads(text[j_start:j_end])
        if not isinstance(mech, dict): return None
        _reward_env.reset()
        obs = _reward_env.step(DaedalusAction(**mech))
        return _to_dict(obs)
    except Exception:
        return None

def reward_format(completions, **kwargs) -> List[float]:
    rewards = []
    for content in completions:
        j_start = content.find("{")
        j_end = content.rfind("}") + 1
        if j_start < 0 or j_end <= j_start:
            rewards.append(-1.0)
            continue
        try:
            mech = json.loads(content[j_start:j_end])
            coverage = len(set(mech.keys()) & REQUIRED_KEYS) / len(REQUIRED_KEYS)
            rewards.append(0.5 + 0.5 * coverage)
        except Exception:
            rewards.append(-0.5)
    return rewards

def reward_welfare(completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome and "metadata" in outcome:
            rewards.append(float(outcome["metadata"].get("welfare_ratio", 0.0)))
        else:
            rewards.append(0.0)
    return rewards

def reward_fairness(completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome and "metadata" in outcome:
            rewards.append(float(outcome["metadata"].get("stability_score", 0.0)))
        else:
            rewards.append(0.0)
    return rewards

def reward_composite(completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome:
            rewards.append(float(outcome.get("reward", 0.0)))
        else:
            rewards.append(0.0)
    return rewards


# -------------------------------------------------------------------
# Phase 1: SFT warmup (Unsloth)
# -------------------------------------------------------------------
def run_sft_and_merge() -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print("\n[sft] loading model for SFT warmup...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    dataset = Dataset.from_list(generate_sft_examples(N_SFT_EXAMPLES))

    cfg = SFTConfig(
        output_dir=SFT_OUT,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_seq_length=2048,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=cfg)
    trainer.train()
    
    print("\n[sft] merging LoRA...")
    model.save_pretrained_merged(SFT_MERGED, tokenizer, save_method="merged_16bit")
    del trainer, model
    gc.collect()
    return SFT_MERGED


# -------------------------------------------------------------------
# Phase 2: GRPO refinement (Unsloth)
# -------------------------------------------------------------------
def run_grpo(merged_model_dir: str):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    print("\n[grpo] loading merged model for GRPO refinement...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_model_dir,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    dataset = Dataset.from_list(generate_grpo_prompts(N_GRPO_PROMPTS))

    cfg = GRPOConfig(
        output_dir=GRPO_OUT,
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=160,
        learning_rate=1e-5,
        logging_steps=2,
        bf16=True,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_REPO,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_format, reward_welfare, reward_fairness, reward_composite],
        args=cfg,
        train_dataset=dataset,
    )

    trainer.train()
    
    # Save training history
    history_path = os.path.join(GRPO_OUT, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"history": trainer.state.log_history}, f, indent=2)

    trainer.push_to_hub()
    print(f"[done] model live at {HUB_REPO}")


def main():
    if os.path.exists(SFT_MERGED): shutil.rmtree(SFT_MERGED)
    merged_path = run_sft_and_merge()
    run_grpo(merged_path)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        pause_self()
