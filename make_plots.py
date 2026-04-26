"""
DAEDALUS Plotting — generates hackathon-ready training evidence.

Usage:
    python make_plots.py                    # auto-fetches history from HF Hub
    python make_plots.py path/to/history.json
    python make_plots.py --hub Laksh718/daedalus-designer

Outputs (saved to ./plots/):
    reward_curve.png        — composite reward over GRPO steps
    component_curves.png    — welfare / fairness / participation / stability
    loss_curve.png          — GRPO optimizer loss (log scale)
    summary.png             — 2×2 panel combining all four signals
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Fetch / locate training_history.json
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_PATHS = [
    "training_history.json",
    "daedalus-lora/training_history.json",
    "grpo-refined/training_history.json",
]
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "Laksh718/daedalus-designer")
OUTPUT_DIR = "plots"


def _fetch_from_hub(repo_id: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="training_history.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[fetch] hub download failed ({e})")
        return None


def load_history(arg: str | None = None) -> dict:
    # 1. Explicit path or --hub flag from CLI
    if arg:
        if arg.startswith("--hub"):
            parts = arg.split()
            repo_id = parts[1] if len(parts) > 1 else HUB_MODEL_ID
            data = _fetch_from_hub(repo_id)
            if data:
                return data
            print("Could not fetch from hub; trying local paths...")
        else:
            with open(arg) as f:
                return json.load(f)

    # 2. Local files
    for p in DEFAULT_LOCAL_PATHS:
        if os.path.exists(p):
            print(f"[load] found local history at {p}")
            with open(p) as f:
                return json.load(f)

    # 3. HF Hub fallback
    print(f"[load] no local history found — fetching from {HUB_MODEL_ID} ...")
    data = _fetch_from_hub(HUB_MODEL_ID)
    if data:
        return data

    print("ERROR: training_history.json not found locally or on HF Hub.")
    print("Run training first, then re-run this script.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _smooth(values: list[float], window: int = 5) -> list[float]:
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), lo + window)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


COLORS = {
    "composite": "#6366f1",
    "welfare":   "#10b981",
    "fairness":  "#f59e0b",
    "participation": "#3b82f6",
    "stability": "#ec4899",
    "loss":      "#ef4444",
}


def generate_plots(data: dict, output_dir: str = OUTPUT_DIR) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    history = data.get("history", [])

    # ── Parse log entries ──────────────────────────────────────────────────
    reward_steps, rewards = [], []
    loss_steps, losses    = [], []
    component_data: dict[str, list] = {
        "welfare": [], "fairness": [], "participation": [], "stability": []
    }
    component_steps = []

    for entry in history:
        step = entry.get("step", entry.get("global_step", 0))
        if "reward" in entry:
            reward_steps.append(step)
            rewards.append(float(entry["reward"]))
        if "loss" in entry:
            loss_steps.append(step)
            losses.append(float(entry["loss"]))
        for key in component_data:
            field = f"rewards/{key}" if f"rewards/{key}" in entry else key
            if field in entry:
                if len(component_steps) < len(component_data["welfare"]) + 1:
                    component_steps.append(step)
                component_data[key].append(float(entry[field]))

    if not rewards:
        print("[plot] WARNING: no 'reward' entries found in history — may be TRL format")
        # TRL GRPO logs reward as 'train/reward' or in nested keys
        for entry in history:
            step = entry.get("step", entry.get("global_step", 0))
            for key in ("train/reward", "rewards/composite", "reward_composite"):
                if key in entry:
                    reward_steps.append(step)
                    rewards.append(float(entry[key]))
                    break

    if not rewards:
        print("[plot] No reward data found. Generating illustrative example plots.")
        _generate_example_plots(output_dir)
        return

    smooth_rewards = _smooth(rewards, window=max(1, len(rewards) // 10))

    # ── Plot 1: Composite Reward Curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_steps, rewards, alpha=0.3, color=COLORS["composite"], linewidth=1, label="raw")
    ax.plot(reward_steps, smooth_rewards, color=COLORS["composite"], linewidth=2.5, label="smoothed")
    ax.fill_between(reward_steps, 0, smooth_rewards, alpha=0.08, color=COLORS["composite"])
    ax.set_title("DAEDALUS GRPO Training: Composite Reward  (W × F × P × S)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Composite Reward  R = W × F × P × S")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/reward_curve.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_dir}/reward_curve.png")

    # ── Plot 2: Loss Curve ─────────────────────────────────────────────────
    if losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(loss_steps, losses, color=COLORS["loss"], linewidth=1.5, label="GRPO loss")
        ax.set_yscale("log")
        ax.set_title("DAEDALUS GRPO Training: Optimizer Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss (log scale)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"[plot] saved {output_dir}/loss_curve.png")

    # ── Plot 3: Per-component curves ───────────────────────────────────────
    has_components = any(len(v) > 0 for v in component_data.values())
    if has_components:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (key, vals) in zip(axes.flat, component_data.items()):
            if not vals:
                ax.set_visible(False)
                continue
            steps = list(range(len(vals)))
            smoothed = _smooth(vals, window=max(1, len(vals) // 10))
            ax.plot(steps, vals, alpha=0.25, color=COLORS[key], linewidth=1)
            ax.plot(steps, smoothed, color=COLORS[key], linewidth=2.5, label=key.capitalize())
            ax.fill_between(steps, 0, smoothed, alpha=0.07, color=COLORS[key])
            ax.set_title(key.capitalize())
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Step")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
        fig.suptitle("DAEDALUS: Per-Component Reward Signals", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/component_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved {output_dir}/component_curves.png")

    # ── Plot 4: 2×2 summary panel ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DAEDALUS Training Summary", fontsize=15, fontweight="bold")

    # Top-left: reward
    ax = axes[0, 0]
    ax.plot(reward_steps, smooth_rewards, color=COLORS["composite"], linewidth=2.5)
    ax.fill_between(reward_steps, 0, smooth_rewards, alpha=0.1, color=COLORS["composite"])
    ax.set_title("Composite Reward R = W×F×P×S")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_ylim(bottom=0); ax.grid(True, linestyle="--", alpha=0.4)

    # Top-right: loss
    ax = axes[0, 1]
    if losses:
        ax.plot(loss_steps, losses, color=COLORS["loss"], linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title("GRPO Loss (log scale)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.set_visible(False)

    # Bottom: component signals
    for ax, (key, vals) in zip(axes[1], list(component_data.items())[:2]):
        if vals:
            sm = _smooth(vals, window=max(1, len(vals) // 10))
            ax.plot(range(len(vals)), vals, alpha=0.25, color=COLORS[key], linewidth=1)
            ax.plot(range(len(sm)), sm, color=COLORS[key], linewidth=2.5, label=key.capitalize())
            ax.set_ylim(0, 1.05); ax.set_title(key.capitalize())
            ax.set_xlabel("Step"); ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/summary.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {output_dir}/summary.png")

    print(f"\n[done] all plots saved to ./{output_dir}/")
    print(f"  Embed in README:  ![Reward](plots/reward_curve.png)")


def _generate_example_plots(output_dir: str) -> None:
    """Generate realistic illustrative plots when no real data is available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    steps = np.arange(0, 300)
    base   = 0.12 + 0.38 * (1 - np.exp(-steps / 80))
    noise  = rng.normal(0, 0.04, len(steps))
    reward = np.clip(base + noise, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, reward, alpha=0.3, color=COLORS["composite"], linewidth=1, label="raw")
    ax.plot(steps, _smooth(list(reward), 15), color=COLORS["composite"], linewidth=2.5, label="smoothed")
    ax.fill_between(steps, 0, _smooth(list(reward), 15), alpha=0.08, color=COLORS["composite"])
    ax.set_title("DAEDALUS GRPO: Composite Reward  (illustrative — train to get real data)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Reward R = W×F×P×S")
    ax.set_ylim(0, 0.7); ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/reward_curve.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved example {output_dir}/reward_curve.png (run training to get real data)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_history(arg)
    generate_plots(data)
