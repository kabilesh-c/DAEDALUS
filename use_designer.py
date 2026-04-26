"""
Quick CLI sanity check for the trained DAEDALUS designer adapter.

Loads:
    base    : HuggingFaceTB/SmolLM2-135M-Instruct
    adapter : kabilesh-c/daedalus-designer  (LoRA, ~3.7 MB)

Resets the DAEDALUS env, asks the model to design a mechanism for the
initial observation, parses the JSON, runs ONE env.step with that mechanism,
and prints the reward.

Usage (PowerShell):
    python use_designer.py
    python use_designer.py --base HuggingFaceTB/SmolLM2-135M-Instruct \
                           --adapter kabilesh-c/daedalus-designer \
                           --temperature 0.7 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from daedalus.env import DaedalusEnvironment


def build_prompt(obs: dict) -> str:
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
        "Respond with ONLY a JSON mechanism configuration:",
        '{"auction_type": "first_price|second_price|vcg", '
        '"reserve_price": float(0-0.9), '
        '"reveal_reserve": bool, "reveal_competing_bids": bool, '
        '"reveal_winner_identity": bool, "reveal_clearing_price": bool, '
        '"reveal_bid_distribution": bool, '
        '"shill_penalty": float(0-3), "withdrawal_penalty": float(0-3), '
        '"collusion_penalty": float(0-3), '
        '"coalition_policy": "allow|restrict|penalize_suspected|penalize_confirmed"}',
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--adapter", default="kabilesh-c/daedalus-designer")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3, help="independent samples to draw")
    p.add_argument("--no-adapter", action="store_true",
                   help="run the base model only (useful as a baseline)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[load] device={device} dtype={dtype} base={args.base}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device)

    if args.no_adapter:
        model = base
        tag = "base-only"
    else:
        print(f"[load] applying LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter).to(device)
        tag = "with-adapter"
    model.eval()

    env = DaedalusEnvironment()
    obs = env.reset()
    user_prompt = build_prompt(obs)
    chat = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    print("\n" + "=" * 64)
    print(f"DAEDALUS designer sanity check  [{tag}]")
    print("=" * 64)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    successes = 0
    rewards: list[float] = []
    for i in range(args.runs):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        print(f"\n--- run {i + 1}/{args.runs} ---")
        print(f"raw : {completion[:240]!r}")

        j_start = completion.find("{")
        j_end = completion.rfind("}") + 1
        if j_start < 0 or j_end <= j_start:
            print("parse: NO JSON braces found")
            continue

        try:
            mech = json.loads(completion[j_start:j_end])
        except json.JSONDecodeError as e:
            print(f"parse: failed -> {e}")
            continue

        successes += 1
        env.reset()
        _, reward, _, info = env.step(mech)
        rewards.append(float(reward))
        print(f"parse: OK")
        print(f"mech : {mech}")
        print(f"reward = {reward:.4f}  (W={info['welfare_ratio']:.3f} "
              f"F={1 - info['gini_coefficient']:.3f} "
              f"P={info['participation_rate']:.3f})")

    print("\n" + "=" * 64)
    print(f"valid JSON: {successes}/{args.runs}"
          + (f"  | mean reward = {sum(rewards)/len(rewards):.4f}" if rewards else ""))
    print("=" * 64)


if __name__ == "__main__":
    main()
