---
title: DAEDALUS Training
emoji: 🏛
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
hardware: t4-medium
---

# DAEDALUS Training Space (v3)

One-shot Docker Space that trains the DAEDALUS mechanism designer
on a T4 GPU using a robust three-step pipeline:

1. **SFT warmup** on `Qwen/Qwen2.5-0.5B-Instruct` with synthetic
   `(prompt, valid mechanism JSON)` pairs - teaches the JSON output format.
2. **Merge step** - bake the SFT LoRA into the base weights via
   `merge_and_unload()`. This avoids the `AutoPeftModel` "frozen
   adapter" trap that broke GRPO at step 0 in earlier versions.
3. **GRPO refinement** with a fresh LoRA (passed to `GRPOTrainer` via
   `peft_config`, so TRL handles `requires_grad` correctly by
   construction) and a format-shaped reward.

Look for the sentinel line `[grpo v3] using merge+fresh-adapter approach`
in container logs to confirm the new code is live.

The trained LoRA adapter and `training_history.json` are pushed to
[`kabilesh-c/daedalus-designer`](https://huggingface.co/kabilesh-c/daedalus-designer)
and the Space auto-pauses on completion.

Mode is controlled by the `TRAIN_MODE` Space variable
(default: `short`, ~10-15 min training; `long` ~45-60 min).
