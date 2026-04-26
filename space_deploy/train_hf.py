"""
DAEDALUS Training - Qwen2.5-0.5B + SFT warmup + GRPO refinement (v3)
=====================================================================
Two-phase pipeline that produces a reliable mechanism-designer LoRA:

    Phase 1 (SFT, ~2-4 min in short mode):
        Teach the model the JSON output format on synthetic
        (prompt, valid mechanism) pairs.

    Phase 1.5 (merge):
        Merge the SFT LoRA into the base model -> ./sft-merged
        This avoids the AutoPeftModel.from_pretrained "frozen adapter"
        trap that broke GRPO at step 0 in v1/v2.

    Phase 2 (GRPO, ~6-12 min in short mode):
        Reload the merged model fresh, attach a brand-new LoRA via
        peft_config (so TRL handles requires_grad correctly by
        construction), then refine with a format-shaped reward.

    Phase 3 (push):
        Upload the GRPO adapter and a training_history.json (loss/
        reward curves) to the Hub for the hackathon demo plots.

Auto-pauses the host HF Space on completion so we never re-enter the
Docker restart loop.

Authentication: HF_TOKEN env var (NEVER hardcoded). On HF Spaces this
is supplied as a Space Secret.

Short / long mode is controlled by env vars:
    TRAIN_MODE=short  (default) -> tight defaults, ~10-15 min total
    TRAIN_MODE=long              -> wider defaults, ~45-60 min total
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import sys
import traceback
from typing import Any, Dict, List


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
# Make daedalus/ importable regardless of cwd
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (SCRIPT_DIR, "/workspace", "/app"):
    if os.path.isdir(os.path.join(candidate, "daedalus")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from daedalus.env import DaedalusEnvironment  # noqa: E402


# -------------------------------------------------------------------
# Config (env-overridable)
# -------------------------------------------------------------------
TRAIN_MODE = os.environ.get("TRAIN_MODE", "short").lower()

MODEL_ID = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HUB_REPO = os.environ.get("HUB_MODEL_ID", "kabilesh-c/daedalus-designer")

if TRAIN_MODE == "long":
    N_SFT_EXAMPLES_DEFAULT = "300"
    SFT_EPOCHS_DEFAULT = "2"
    N_GRPO_PROMPTS_DEFAULT = "200"
    GRPO_STEPS_DEFAULT = "120"
else:
    # short = aggressive but still moves the reward curve
    N_SFT_EXAMPLES_DEFAULT = "120"
    SFT_EPOCHS_DEFAULT = "1"
    N_GRPO_PROMPTS_DEFAULT = "80"
    GRPO_STEPS_DEFAULT = "40"

N_SFT_EXAMPLES = int(os.environ.get("N_SFT_EXAMPLES", N_SFT_EXAMPLES_DEFAULT))
SFT_EPOCHS = int(os.environ.get("SFT_EPOCHS", SFT_EPOCHS_DEFAULT))
N_GRPO_PROMPTS = int(os.environ.get("N_GRPO_PROMPTS", N_GRPO_PROMPTS_DEFAULT))
GRPO_STEPS = int(os.environ.get("GRPO_STEPS", GRPO_STEPS_DEFAULT))

SFT_OUT = "./sft-warmup"
SFT_MERGED = "./sft-merged"
GRPO_OUT = "./grpo-refined"

REQUIRED_KEYS = {
    "auction_type",
    "reserve_price",
    "reveal_reserve",
    "reveal_competing_bids",
    "reveal_winner_identity",
    "reveal_clearing_price",
    "reveal_bid_distribution",
    "shill_penalty",
    "withdrawal_penalty",
    "collusion_penalty",
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
        "  auction_type        : one of \"first_price\" | \"second_price\" | \"vcg\"",
        "  reserve_price       : float in [0.0, 0.9]",
        "  reveal_reserve      : bool",
        "  reveal_competing_bids   : bool",
        "  reveal_winner_identity  : bool",
        "  reveal_clearing_price   : bool",
        "  reveal_bid_distribution : bool",
        "  shill_penalty       : float in [0.0, 3.0]",
        "  withdrawal_penalty  : float in [0.0, 3.0]",
        "  collusion_penalty   : float in [0.0, 3.0]",
        "  coalition_policy    : one of \"allow\" | \"restrict\" | \"penalize_suspected\" | \"penalize_confirmed\"",
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
        "coalition_policy": random.choice(
            ["allow", "restrict", "penalize_suspected", "penalize_confirmed"]
        ),
    }


def generate_sft_examples(n: int) -> List[Dict[str, Any]]:
    print(f"[data] generating {n} SFT examples (prompt -> valid JSON) ...")
    pairs: List[Dict[str, Any]] = []
    env = DaedalusEnvironment()
    while len(pairs) < n:
        obs = env.reset()
        for _ in range(random.randint(1, 5)):
            mech = random_valid_mechanism()
            answer = json.dumps(mech)
            pairs.append({
                "messages": [
                    {"role": "user", "content": format_prompt(obs)},
                    {"role": "assistant", "content": answer},
                ]
            })
            obs, _, done, _ = env.step(mech)
            if done or len(pairs) >= n:
                break
    print(f"[data] SFT examples: {len(pairs)}")
    return pairs[:n]


def generate_grpo_prompts(n: int) -> List[Dict[str, str]]:
    print(f"[data] generating {n} GRPO prompts via env rollout ...")
    prompts: List[Dict[str, str]] = []
    env = DaedalusEnvironment()
    while len(prompts) < n:
        obs = env.reset()
        prompts.append({"prompt": format_prompt(obs)})
        for _ in range(random.randint(1, 5)):
            obs, _, done, _ = env.step(random_valid_mechanism())
            if done:
                break
            prompts.append({"prompt": format_prompt(obs)})
            if len(prompts) >= n:
                break
    print(f"[data] GRPO prompts: {len(prompts[:n])}")
    return prompts[:n]


# -------------------------------------------------------------------
# Reward shaping
# -------------------------------------------------------------------
_eval_env = DaedalusEnvironment()


def shaped_reward(text: str) -> float:
    """
    Continuous reward landscape with stepping stones:
      - no JSON braces                  -> -1.0
      - braces but JSON parse fails     -> -0.5
      - parses but not a dict           -> -0.3
      - dict, partial keys              ->  0.2 + 0.3*coverage   (max 0.5)
      - dict, env step succeeds         -> +format_bonus + env reward
    """
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start < 0 or j_end <= j_start:
        return -1.0
    try:
        mech = json.loads(text[j_start:j_end])
    except Exception:
        return -0.5
    if not isinstance(mech, dict):
        return -0.3

    present = set(mech.keys()) & REQUIRED_KEYS
    coverage = len(present) / len(REQUIRED_KEYS)
    format_bonus = 0.2 + 0.3 * coverage  # 0.2 to 0.5

    try:
        _eval_env.reset()
        _, env_r, _, _ = _eval_env.step(mech)
        return float(format_bonus + env_r)
    except Exception:
        return float(format_bonus - 0.2)


def grpo_reward_fn(completions, prompts=None, **kwargs) -> List[float]:
    return [shaped_reward(c) for c in completions]


# -------------------------------------------------------------------
# Phase 1: SFT warmup  ->  Phase 1.5: merge into base
# -------------------------------------------------------------------
def run_sft_and_merge() -> str:
    """Run SFT, merge the LoRA into the base, save merged model to SFT_MERGED."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoTokenizer

    print("\n" + "=" * 64)
    print(f"PHASE 1 / SFT warmup  (target ~2-4 min in short mode)")
    print("=" * 64)

    for d in (SFT_OUT, SFT_MERGED):
        if os.path.isdir(d):
            shutil.rmtree(d)

    examples = generate_sft_examples(N_SFT_EXAMPLES)
    dataset = Dataset.from_list(examples)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    cfg = SFTConfig(
        output_dir=SFT_OUT,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        max_length=1500,
        report_to="none",
        push_to_hub=False,
        seed=42,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(SFT_OUT)
    print(f"[sft] adapter saved to {SFT_OUT}")

    # ---- Phase 1.5: merge SFT LoRA into base, save as a regular HF model ----
    print("\n" + "-" * 64)
    print("PHASE 1.5 / merging SFT adapter into base weights ...")
    print("-" * 64)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(SFT_MERGED)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(SFT_MERGED)
    print(f"[sft] merged model saved to {SFT_MERGED}")

    # Free SFT trainer GPU memory before GRPO phase
    del trainer, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return SFT_MERGED


# -------------------------------------------------------------------
# Phase 2: GRPO refinement on the merged model with a FRESH adapter
# -------------------------------------------------------------------
def run_grpo(merged_model_dir: str) -> Dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    print("\n" + "=" * 64)
    print("PHASE 2 / GRPO refinement  (target ~6-12 min in short mode)")
    print(f"          base = merged SFT model at {merged_model_dir}")
    print("=" * 64)

    # >>> SENTINEL: if you don't see this line in container logs, the new
    # >>> code did NOT reach the Space (only restart_space was called).
    print("[grpo v3] using merge+fresh-adapter approach (peft_config -> GRPOTrainer)")

    dataset = Dataset.from_list(generate_grpo_prompts(N_GRPO_PROMPTS))

    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(merged_model_dir, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(merged_model_dir, torch_dtype=dtype)
    model.config.use_cache = False  # critical for training stability

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir=GRPO_OUT,
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=140,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=2,
        save_steps=max(GRPO_STEPS, 1),
        save_total_limit=1,
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        gradient_checkpointing=False,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_REPO,
        hub_strategy="end",
        hub_private_repo=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[grpo_reward_fn],
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Print trainable param count AFTER trainer wrapped the model in PEFT.
    # If this is 0, GRPO will fail at step 0 - we want to see it before training.
    inner = trainer.model
    trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    total = sum(p.numel() for p in inner.parameters())
    pct = 100 * trainable / total if total else 0.0
    print(f"[grpo] trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")
    if trainable == 0:
        raise RuntimeError(
            "[grpo] no trainable parameters - peft_config did not wire up. "
            "Aborting before backward pass crash."
        )

    trainer.train()

    # Capture training history for hackathon plots
    history = trainer.state.log_history
    history_path = os.path.join(GRPO_OUT, "training_history.json")
    os.makedirs(GRPO_OUT, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "config": {
            "model_id": MODEL_ID,
            "n_sft_examples": N_SFT_EXAMPLES,
            "sft_epochs": SFT_EPOCHS,
            "n_grpo_prompts": N_GRPO_PROMPTS,
            "grpo_steps": GRPO_STEPS,
            "train_mode": TRAIN_MODE,
        }}, f, indent=2, default=str)
    print(f"[grpo] training_history.json written: {len(history)} log entries")

    print("[grpo] uploading final adapter + history to Hub ...")
    trainer.push_to_hub()

    # Also push the training_history.json directly so the README plot script can find it.
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=history_path,
            path_in_repo="training_history.json",
            repo_id=HUB_REPO,
            repo_type="model",
        )
        print(f"[grpo] training_history.json pushed to {HUB_REPO}")
    except Exception as e:
        print(f"[grpo] WARNING: history upload failed: {e}")

    print(f"[grpo] adapter live at https://huggingface.co/{HUB_REPO}")
    return {"history": history}


# -------------------------------------------------------------------
# Auto-pause Space so the Docker restart loop never re-bills us
# -------------------------------------------------------------------
def pause_self() -> None:
    if not (SPACE_ID and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        print(f"[shutdown] pausing Space {SPACE_ID} to prevent restart loop ...")
        api.pause_space(SPACE_ID)
        print(f"[shutdown] Space paused")
    except Exception as e:
        print(f"[shutdown] pause failed: {e}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    import torch
    print("=" * 64)
    print("DAEDALUS two-phase training (v3 / merge+fresh-adapter)")
    print(f"  train_mode      : {TRAIN_MODE}")
    print(f"  base model      : {MODEL_ID}")
    print(f"  hub destination : {HUB_REPO}")
    print(f"  sft examples    : {N_SFT_EXAMPLES} x {SFT_EPOCHS} epochs")
    print(f"  grpo steps      : {GRPO_STEPS}  ({N_GRPO_PROMPTS} prompts)")
    print(f"  cuda            : {torch.cuda.is_available()}  "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print("=" * 64)

    merged_dir = run_sft_and_merge()
    run_grpo(merged_dir)


if __name__ == "__main__":
    try:
        main()
        print("[done] training pipeline completed successfully")
    except Exception:
        traceback.print_exc()
    finally:
        pause_self()
