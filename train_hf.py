"""
DAEDALUS Training v4 - Unsloth + Qwen2.5-0.5B + GRPO (single-adapter pipeline)
==============================================================================

What changed vs v3:
    * Single LoRA on the original Qwen base (no SFT-merge step). The exported
      adapter sits cleanly on top of `Qwen/Qwen2.5-0.5B-Instruct` so any
      consumer (e.g. server.py) can load it without juggling intermediate
      checkpoints.
    * Uses unsloth's pre-quantized `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`
      to skip on-the-fly 4-bit quantization (~2x faster startup, lower RAM).
    * After GRPO completes, the LoRA is merged and the full 16-bit model is
      pushed to the hub. server.py can now do a one-liner load.
    * Adds `pause_self()` so the script no longer NameErrors on exit.
    * Smaller `max_seq_length` (1024) and bigger micro-batch since the model
      is tiny and the prompts are short.
    * Sentinel log line: `[grpo v4] single-adapter on base + push merged`.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

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
# Unsloth & TRL Patching (must happen before TRL import)
# -------------------------------------------------------------------
from unsloth import FastLanguageModel, PatchFastRL  # noqa: E402

PatchFastRL("grpo", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")


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
# We load the unsloth pre-quantized variant for 2x faster startup, but the
# resulting LoRA is fully compatible with the upstream `Qwen/Qwen2.5-0.5B-Instruct`.
MODEL_ID = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
HUB_REPO = os.environ.get("HUB_MODEL_ID", "kabilesh-c/daedalus-designer")
PUSH_MERGED = os.environ.get("PUSH_MERGED", "1") not in ("0", "false", "False")

if TRAIN_MODE == "long":
    N_SFT_EXAMPLES = 400
    SFT_EPOCHS = 2
    N_GRPO_PROMPTS = 240
    GRPO_STEPS = 160
else:
    N_SFT_EXAMPLES = 160
    SFT_EPOCHS = 1
    N_GRPO_PROMPTS = 96
    GRPO_STEPS = 60

OUT_DIR = "./daedalus-lora"
MAX_SEQ_LEN = 1024  # mechanism prompts + JSON completions easily fit in 1k

REQUIRED_KEYS = {
    "auction_type", "reserve_price", "reveal_reserve",
    "reveal_competing_bids", "reveal_winner_identity",
    "reveal_clearing_price", "reveal_bid_distribution",
    "shill_penalty", "withdrawal_penalty", "collusion_penalty",
    "coalition_policy",
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
    for stage in range(5):
        n_stage = max(1, n // 5)
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
                if obs.done:
                    break
    return pairs[:n]


def generate_grpo_prompts(n: int) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    for stage in range(5):
        n_stage = max(1, n // 5)
        env = make_env(stage)
        while len(prompts) < (stage + 1) * n_stage:
            obs_dict = _to_dict(env.reset())
            prompts.append({"prompt": format_prompt(obs_dict)})
            for _ in range(3):
                obs = env.step(DaedalusAction(**random_valid_mechanism()))
                prompts.append({"prompt": format_prompt(_to_dict(obs))})
                if obs.done:
                    break
    return prompts[:n]


# -------------------------------------------------------------------
# Decomposed Reward Functions
# -------------------------------------------------------------------
_reward_env = make_env()


@functools.lru_cache(maxsize=256)
def _get_env_outcome(text: str) -> Optional[dict]:
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start < 0 or j_end <= j_start:
        return None
    try:
        mech = json.loads(text[j_start:j_end])
        if not isinstance(mech, dict):
            return None
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
    out = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome and "metadata" in outcome:
            out.append(float(outcome["metadata"].get("welfare_ratio", 0.0)))
        else:
            out.append(0.0)
    return out


def reward_fairness(completions, **kwargs) -> List[float]:
    out = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome and "metadata" in outcome:
            out.append(float(outcome["metadata"].get("stability_score", 0.0)))
        else:
            out.append(0.0)
    return out


def reward_composite(completions, **kwargs) -> List[float]:
    out = []
    for c in completions:
        outcome = _get_env_outcome(c)
        if outcome:
            out.append(float(outcome.get("reward", 0.0)))
        else:
            out.append(0.0)
    return out


# -------------------------------------------------------------------
# Single-adapter training pipeline (SFT then GRPO on the SAME LoRA)
# -------------------------------------------------------------------
def build_model_and_tokenizer():
    """Load Qwen 0.5B (4-bit) and attach a single LoRA used for both SFT and GRPO."""
    print("[model] loading base via Unsloth (pre-quantized 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # let unsloth choose bf16/fp16
    )
    print("[model] attaching LoRA (r=16, all attn + MLP)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def run_sft(model, tokenizer):
    """Phase 1: teach the JSON output format via supervised fine-tuning."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[sft] generating {N_SFT_EXAMPLES} synthetic (prompt, mechanism) pairs ...")
    dataset = Dataset.from_list(generate_sft_examples(N_SFT_EXAMPLES))

    cfg = SFTConfig(
        output_dir=OUT_DIR + "-sft",
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=cfg,
    )
    print("[sft] training ...")
    trainer.train()

    del trainer
    gc.collect()
    return model


def run_grpo(model, tokenizer):
    """Phase 2: GRPO refinement on the SAME LoRA we just SFT'd."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    print("[grpo v4] using single-adapter (no merge) approach")
    print(f"[grpo] generating {N_GRPO_PROMPTS} prompts ...")
    dataset = Dataset.from_list(generate_grpo_prompts(N_GRPO_PROMPTS))

    cfg = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=192,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=2,
        bf16=True,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_format, reward_welfare, reward_fairness, reward_composite],
        args=cfg,
        train_dataset=dataset,
    )
    print("[grpo] training ...")
    trainer.train()

    history_path = os.path.join(OUT_DIR, "training_history.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({"history": trainer.state.log_history}, f, indent=2)
    print(f"[grpo] saved log history to {history_path}")

    return trainer.model


def push_to_hub(model, tokenizer):
    """Push merged 16-bit weights so consumers can load with one line."""
    if not HF_TOKEN:
        print("[push] skipped: HF_TOKEN not set")
        return

    if PUSH_MERGED:
        print(f"[push] merging LoRA + uploading FULL 16-bit model to {HUB_REPO} ...")
        try:
            model.push_to_hub_merged(
                HUB_REPO,
                tokenizer,
                save_method="merged_16bit",
                token=HF_TOKEN,
            )
            print(f"[done] merged model live at https://huggingface.co/{HUB_REPO}")
            return
        except Exception as e:
            print(f"[push] merged upload failed ({e}); falling back to LoRA-only push")

    print(f"[push] uploading LoRA adapter to {HUB_REPO} ...")
    model.push_to_hub(HUB_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print(f"[done] adapter live at https://huggingface.co/{HUB_REPO}")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def pause_self() -> None:
    """When running inside a HF Space, pause the Space so it stops billing.

    Outside of a Space (or on auth failure) this is a harmless no-op.
    """
    if not SPACE_ID or not HF_TOKEN:
        print("[pause] not in a Space (or no token); skipping pause_space")
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=HF_TOKEN).pause_space(SPACE_ID)
        print(f"[pause] paused Space {SPACE_ID}")
    except Exception as e:
        print(f"[pause] pause_space failed: {e}")


def main():
    t0 = time.time()
    model, tokenizer = build_model_and_tokenizer()
    model = run_sft(model, tokenizer)
    model = run_grpo(model, tokenizer)
    push_to_hub(model, tokenizer)
    print(f"[done] total wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        pause_self()
