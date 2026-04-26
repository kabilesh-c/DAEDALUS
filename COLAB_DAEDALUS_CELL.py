# =====================================================================
# DAEDALUS — Mechanism Design via Adversarial RL (Colab Submission)
# =====================================================================
# This cell performs a one-shot training run of the DAEDALUS designer.
# 1. Installs dependencies
# 2. Clones the project logic
# 3. Runs SFT Warmup -> Multi-Reward GRPO Refinement
# 4. Plots the results inline
# =====================================================================

# 1. INSTALL DEPENDENCIES
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q openenv-core trl peft transformers datasets accelerate huggingface_hub python-dotenv

# 2. SETUP PROJECT
import os
import shutil
if os.path.exists("DAEDALUS"): shutil.rmtree("DAEDALUS")
!git clone https://github.com/kabilesh-c/DAEDALUS.git
os.chdir("DAEDALUS")

# 3. RUN TRAINING (Fast Mode for Verification)
# We override the TRAIN_MODE to 'short' for judges so it finishes in ~15 mins.
os.environ["TRAIN_MODE"] = "short"
os.environ["BASE_MODEL"] = "unsloth/Qwen2.5-0.5B-Instruct"

from train_hf import main
import matplotlib.pyplot as plt
import json

print("\n🚀 Starting DAEDALUS Training Pipeline...")
try:
    main()
except Exception as e:
    print(f"Error during training: {e}")

# 4. PLOT RESULTS
print("\n📊 Generating Results...")
history_path = "grpo-refined/training_history.json"
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        data = json.load(f)
    
    logs = data.get("history", [])
    steps = [x["step"] for x in logs if "reward" in x]
    rewards = [x["reward"] for x in logs if "reward" in x]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, color="#6366f1", label="Composite Reward")
    plt.title("DAEDALUS: LLM Learning Curve (Mechanism Design)")
    plt.xlabel("Step")
    plt.ylabel("W*F*P*S Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("Training history not found. Check logs for errors.")
