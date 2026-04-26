"""
DAEDALUS Plotting — Generates Hackathon-ready visual evidence.
Pulls training_history.json and produces reward/loss curves.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(history_path="grpo-refined/training_history.json", output_dir="plots"):
    if not os.path.exists(history_path):
        print(f"ERROR: {history_path} not found. Run training first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_path, "r") as f:
        data = json.load(f)
    
    history = data.get("history", [])
    
    steps = [x["step"] for x in history if "reward" in x]
    rewards = [x["reward"] for x in history if "reward" in x]
    loss = [x["loss"] for x in history if "loss" in x]
    loss_steps = [x["step"] for x in history if "loss" in x]

    # --- Plot 1: Reward Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label="Composite Reward (W*F*P*S)", color="#6366f1", linewidth=2)
    plt.fill_between(steps, 0, rewards, alpha=0.1, color="#6366f1")
    plt.title("DAEDALUS Training: Reward Improvement", fontsize=14, fontweight='bold')
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Multiplicative Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=300)
    print(f"Saved {output_dir}/reward_curve.png")

    # --- Plot 2: Loss Curve ---
    if loss:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_steps, loss, label="GRPO Loss", color="#ec4899", linewidth=2)
        plt.yscale('log')
        plt.title("DAEDALUS Training: Optimizer Convergence", fontsize=14, fontweight='bold')
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Log Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
        print(f"Saved {output_dir}/loss_curve.png")

if __name__ == "__main__":
    generate_plots()
