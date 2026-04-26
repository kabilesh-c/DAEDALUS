"""
DAEDALUS Training v5 — Unsloth + Qwen2.5-0.5B + GRPO (single-adapter pipeline)
================================================================================

Changes vs v4:
    * reward_fairness now correctly computes 1 − gini_coefficient (never stability).
    * reward_stability added as a separate fifth reward signal.
    * GRPO reward_funcs updated to [format, welfare, fairness, stability, composite].
    * Training scales up automatically when A100-class GPU is detected:
        - 'long' mode doubles SFT/GRPO counts and uses larger batches.
        - 'full' mode is the maximum A100 training budget.
    * HUB_REPO defaults to HUB_MODEL_ID env var (set by deploy script).
    * format_prompt includes Curriculum Stage line to match server.py inference.
    * Sentinel: [grpo v5] five-reward single-adapter
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


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Unsloth import (must happen before TRL so unsloth can patch).
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402

# Ampere+ (A100/A10G/3090+) supports bfloat16; T4 (Turing) only fp16.
USE_BF16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
USE_FP16 = bool(torch.cuda.is_available() and not USE_BF16)
print(f"[precision] bf16={USE_BF16}  fp16={USE_FP16}  cuda={torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] {gpu_name}  VRAM={gpu_mem_gb:.0f} GB")
else:
    gpu_name = "CPU"
    gpu_mem_gb = 0.0

# Detect A100-class GPU (>=40 GB VRAM) for automatic scaling.
IS_HIGH_VRAM = gpu_mem_gb >= 40.0


# ---------------------------------------------------------------------------
# Make daedalus/ importable regardless of cwd.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (SCRIPT_DIR, "/workspace", "/app"):
    if os.path.isdir(os.path.join(candidate, "daedalus")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

# Training uses the legacy env directly — no openenv-core dep on the Space.
from daedalus.env import DaedalusEnvironment  # noqa: E402


def _to_dict(obs: Any) -> dict:
    if obs is None:
        return {}
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    return dict(obs)


def make_env(stage: int = 0) -> DaedalusEnvironment:
    return DaedalusEnvironment(curriculum_stage=stage)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_MODE = os.environ.get("TRAIN_MODE", "short").lower()

MODEL_ID = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
HUB_REPO = os.environ.get("HUB_MODEL_ID", "Laksh718/daedalus-designer")
PUSH_MERGED = os.environ.get("PUSH_MERGED", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Training scale: auto-boost when a high-VRAM GPU is detected.
# ---------------------------------------------------------------------------
if TRAIN_MODE == "full":
    # Maximum quality — designed for A100 80 GB. ~60-90 min.
    N_SFT_EXAMPLES = 2000
    SFT_EPOCHS = 3
    N_GRPO_PROMPTS = 1200
    GRPO_STEPS = 500
    SFT_BATCH = 16 if IS_HIGH_VRAM else 8
    GRPO_BATCH = 8  if IS_HIGH_VRAM else 4
    GRPO_GENERATIONS = 8 if IS_HIGH_VRAM else 4
elif TRAIN_MODE == "long":
    # Good quality run — 4×A100 ~30-45 min, single A100 ~25-35 min, T4 ~35-50 min.
    N_SFT_EXAMPLES = 1200 if IS_HIGH_VRAM else 400
    SFT_EPOCHS = 2
    N_GRPO_PROMPTS = 700 if IS_HIGH_VRAM else 240
    GRPO_STEPS = 400 if IS_HIGH_VRAM else 160
    SFT_BATCH = 16 if IS_HIGH_VRAM else 8
    GRPO_BATCH = 8  if IS_HIGH_VRAM else 4
    GRPO_GENERATIONS = 8 if IS_HIGH_VRAM else 4
elif TRAIN_MODE == "smoke":
    # CI smoke test — exercises every code path in ~3-5 min.
    N_SFT_EXAMPLES = 24
    SFT_EPOCHS = 1
    N_GRPO_PROMPTS = 16
    GRPO_STEPS = 4
    SFT_BATCH = 4
    GRPO_BATCH = 2
    GRPO_GENERATIONS = 4
else:
    # "short" — quick but useful run (~10-15 min on A100, ~15-25 min on T4).
    N_SFT_EXAMPLES = 320 if IS_HIGH_VRAM else 160
    SFT_EPOCHS = 1
    N_GRPO_PROMPTS = 200 if IS_HIGH_VRAM else 96
    GRPO_STEPS = 120 if IS_HIGH_VRAM else 60
    SFT_BATCH = 10 if IS_HIGH_VRAM else 8
    GRPO_BATCH = 5  if IS_HIGH_VRAM else 4
    GRPO_GENERATIONS = 8 if IS_HIGH_VRAM else 4

OUT_DIR = "./daedalus-lora"
MAX_SEQ_LEN = 1024

REQUIRED_KEYS = {
    "auction_type", "reserve_price", "reveal_reserve",
    "reveal_competing_bids", "reveal_winner_identity",
    "reveal_clearing_price", "reveal_bid_distribution",
    "shill_penalty", "withdrawal_penalty", "collusion_penalty",
    "coalition_policy",
}


# ---------------------------------------------------------------------------
# Prompt — MUST stay identical to server.py::_build_prompt
# ---------------------------------------------------------------------------
def format_prompt(obs: dict) -> str:
    lines = [
        "You are a mechanism designer for a market auction system.",
        "Analyze the current market state and design an optimal mechanism.",
        "",
        f"Round: {obs.get('round_number', 0)} / {obs.get('episode_length', 50)}",
        f"Curriculum Stage: {obs.get('curriculum_stage', 0)}",
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
                f"S={o.get('stability_score', 1):.3f} "
                f"R={o.get('composite_reward', 0):.3f}"
            )
    proxies = obs.get("population_proxies", {})
    if proxies:
        lines.extend([
            "",
            "Population Signals:",
            f"  Active Bidders: {proxies.get('active_count', 8)} / {proxies.get('total_agents', 8)}",
            f"  Bid Correlation (collusion proxy): {proxies.get('bid_correlation', 0):.3f}",
            f"  Winner Rotation Entropy: {proxies.get('rotation_entropy', 1):.3f}",
            f"  Dropout Rate: {proxies.get('dropout_rate', 0):.3f}",
        ])
    lines.extend([
        "",
        "Respond with ONLY a JSON mechanism configuration with these exact keys:",
        "  auction_type: \"first_price\" | \"second_price\" | \"vcg\"",
        "  reserve_price: float [0.0, 0.9]",
        "  reveal_reserve: bool",
        "  reveal_competing_bids: bool",
        "  reveal_winner_identity: bool",
        "  reveal_clearing_price: bool",
        "  reveal_bid_distribution: bool",
        "  shill_penalty: float [0.0, 3.0]",
        "  withdrawal_penalty: float [0.0, 3.0]",
        "  collusion_penalty: float [0.0, 3.0]",
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
        "coalition_policy": random.choice(
            ["allow", "restrict", "penalize_suspected", "penalize_confirmed"]
        ),
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
                obs_dict, _, done, _ = env.step(mech)
                obs_dict = _to_dict(obs_dict)
                if done:
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
                obs_dict, _, done, _ = env.step(random_valid_mechanism())
                prompts.append({"prompt": format_prompt(_to_dict(obs_dict))})
                if done:
                    break
    return prompts[:n]


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
_reward_env: Optional[DaedalusEnvironment] = None


def _get_reward_env() -> DaedalusEnvironment:
    global _reward_env
    if _reward_env is None:
        _reward_env = make_env(stage=0)
    return _reward_env


def _completion_text(c: Any) -> str:
    """Normalize a TRL completion to plain string regardless of its format."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        last = c[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(c, dict):
        return str(c.get("content", c))
    return str(c)


@functools.lru_cache(maxsize=4096)
def _get_env_outcome(text: str) -> Optional[dict]:
    """Run one env step for the JSON inside `text`. Cached to avoid re-simulation."""
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start < 0 or j_end <= j_start:
        return None
    try:
        mech = json.loads(text[j_start:j_end])
        if not isinstance(mech, dict):
            return None
        env = _get_reward_env()
        env.reset()
        obs_dict, reward, done, info = env.step(mech)
        return {
            "obs": _to_dict(obs_dict),
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info or {}),
        }
    except Exception:
        return None


def reward_format(completions=None, **kwargs) -> List[float]:
    """Schema coverage: 0.5 + 0.5*coverage ∈ [−1, 1]."""
    rewards = []
    for raw in (completions or []):
        content = _completion_text(raw)
        j_start = content.find("{")
        j_end = content.rfind("}") + 1
        if j_start < 0 or j_end <= j_start:
            rewards.append(-1.0)
            continue
        try:
            mech = json.loads(content[j_start:j_end])
            if not isinstance(mech, dict):
                rewards.append(-0.5)
                continue
            coverage = len(set(mech.keys()) & REQUIRED_KEYS) / len(REQUIRED_KEYS)
            rewards.append(0.5 + 0.5 * coverage)
        except Exception:
            rewards.append(-0.5)
    return rewards


def reward_welfare(completions=None, **kwargs) -> List[float]:
    """Social welfare ratio W ∈ [0, 1]."""
    out = []
    for raw in (completions or []):
        outcome = _get_env_outcome(_completion_text(raw))
        if outcome:
            out.append(float(outcome["info"].get("welfare_ratio", 0.0)))
        else:
            out.append(0.0)
    return out


def reward_fairness(completions=None, **kwargs) -> List[float]:
    """Fairness = 1 − Gini(surplus) ∈ [0, 1].
    Always computed from gini_coefficient — never reads stability_score.
    """
    out = []
    for raw in (completions or []):
        outcome = _get_env_outcome(_completion_text(raw))
        if outcome:
            info = outcome["info"]
            gini = float(info.get("gini_coefficient", 0.0))
            out.append(max(0.0, 1.0 - gini))
        else:
            out.append(0.0)
    return out


def reward_stability(completions=None, **kwargs) -> List[float]:
    """Stability = 1 − 3·σ(recent welfare) ∈ [0, 1]."""
    out = []
    for raw in (completions or []):
        outcome = _get_env_outcome(_completion_text(raw))
        if outcome:
            out.append(float(outcome["info"].get("stability_score", 1.0)))
        else:
            out.append(1.0)  # Neutral default — don't punish parse failures twice
    return out


def reward_composite(completions=None, **kwargs) -> List[float]:
    """Full composite R = W × F × P × S × anti_collusion ∈ [0, 1]."""
    out = []
    for raw in (completions or []):
        outcome = _get_env_outcome(_completion_text(raw))
        if outcome:
            out.append(float(outcome.get("reward", 0.0)))
        else:
            out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------
def build_model_and_tokenizer():
    print("[model] loading base via Unsloth (pre-quantized 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # Unsloth picks bf16/fp16 automatically
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
        lora_dropout=0,      # Must be 0 for Unsloth's fast kernel path
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Phase 1: SFT — teach the JSON schema
# ---------------------------------------------------------------------------
def run_sft(model, tokenizer):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[sft] generating {N_SFT_EXAMPLES} synthetic (prompt, mechanism) pairs ...")
    dataset = Dataset.from_list(generate_sft_examples(N_SFT_EXAMPLES))

    cfg = SFTConfig(
        output_dir=OUT_DIR + "-sft",
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=max(5, N_SFT_EXAMPLES // 20),
        logging_steps=5,
        save_strategy="no",
        bf16=USE_BF16,
        fp16=USE_FP16,
        max_seq_length=MAX_SEQ_LEN,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=cfg,
    )
    print(f"[sft] training  (batch={SFT_BATCH}, epochs={SFT_EPOCHS}) ...")
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Phase 2: GRPO — reinforce with 5 reward signals on the SAME LoRA
# ---------------------------------------------------------------------------
def run_grpo(model, tokenizer):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    print("[grpo v5] five-reward single-adapter on base + push merged")
    print(f"[grpo] generating {N_GRPO_PROMPTS} prompts ...")
    dataset = Dataset.from_list(generate_grpo_prompts(N_GRPO_PROMPTS))

    grpo_warmup = max(1, GRPO_STEPS // 10)
    cfg = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=GRPO_BATCH,
        gradient_accumulation_steps=2,
        num_generations=GRPO_GENERATIONS,
        max_completion_length=400,   # Full mechanism JSON ~100-150 tokens; 400 is safe
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=grpo_warmup,
        logging_steps=2,
        save_steps=max(10, GRPO_STEPS // 10),
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,
            reward_welfare,
            reward_fairness,
            reward_stability,
            reward_composite,
        ],
        args=cfg,
        train_dataset=dataset,
    )
    print(
        f"[grpo] training  "
        f"(batch={GRPO_BATCH}, generations={GRPO_GENERATIONS}, steps={GRPO_STEPS}) ..."
    )
    trainer.train()

    # Save training history for make_plots.py
    os.makedirs(OUT_DIR, exist_ok=True)
    history_path = os.path.join(OUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"history": trainer.state.log_history}, f, indent=2)
    print(f"[grpo] saved log history → {history_path}")

    # Also save a copy in working directory root for easy access
    try:
        import shutil
        shutil.copy(history_path, "./training_history.json")
    except Exception:
        pass

    return trainer.model


# ---------------------------------------------------------------------------
# Push training_history.json to HF Hub (survives ephemeral job filesystem)
# ---------------------------------------------------------------------------
def push_history_to_hub(history_path: str) -> None:
    if not HF_TOKEN or not os.path.exists(history_path):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HUB_REPO, repo_type="model", exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=history_path,
            path_in_repo="training_history.json",
            repo_id=HUB_REPO,
            repo_type="model",
            commit_message="add training history",
        )
        print(f"[push] training_history.json → https://huggingface.co/{HUB_REPO}/blob/main/training_history.json")
    except Exception as e:
        print(f"[push] history upload failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Push merged 16-bit model to Hub
# ---------------------------------------------------------------------------
def push_to_hub(model, tokenizer):
    if not HF_TOKEN:
        print("[push] skipped — HF_TOKEN not set")
        return

    # Upload training history first (small file, survives even if model push fails)
    for candidate in (
        os.path.join(OUT_DIR, "training_history.json"),
        "./training_history.json",
    ):
        if os.path.exists(candidate):
            push_history_to_hub(candidate)
            break

    if PUSH_MERGED:
        print(f"[push] merging LoRA and uploading full 16-bit model → {HUB_REPO} ...")
        try:
            model.push_to_hub_merged(
                HUB_REPO,
                tokenizer,
                save_method="merged_16bit",
                token=HF_TOKEN,
                private=False,
            )
            print(f"[done] merged model live at https://huggingface.co/{HUB_REPO}")
            return
        except Exception as e:
            traceback.print_exc()
            print(f"[push] merged upload failed ({e}); falling back to LoRA-only push")

    print(f"[push] uploading LoRA adapter → {HUB_REPO} ...")
    model.push_to_hub(HUB_REPO, token=HF_TOKEN, private=False)
    tokenizer.push_to_hub(HUB_REPO, token=HF_TOKEN, private=False)
    print(f"[done] adapter live at https://huggingface.co/{HUB_REPO}")


# ---------------------------------------------------------------------------
# Space auto-pause (stops billing when training completes)
# ---------------------------------------------------------------------------
def pause_self() -> None:
    if not SPACE_ID or not HF_TOKEN:
        print("[pause] not in a Space (or no token) — skipping")
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).pause_space(SPACE_ID)
        print(f"[pause] Space {SPACE_ID} paused")
    except Exception as e:
        print(f"[pause] pause_space failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(
        f"[main] TRAIN_MODE={TRAIN_MODE}  gpu={gpu_name}  "
        f"SFT={N_SFT_EXAMPLES}×{SFT_EPOCHS}ep  "
        f"GRPO={GRPO_STEPS} steps × {GRPO_GENERATIONS} gen  "
        f"→ {HUB_REPO}"
    )
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
