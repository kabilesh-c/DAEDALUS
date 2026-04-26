---
title: DAEDALUS Training
emoji: 🏛
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
hardware: t4-medium
---

# DAEDALUS Training Space (v4)

One-shot Docker Space that trains the DAEDALUS mechanism designer on a T4
GPU using a streamlined Unsloth + Qwen pipeline:

1. **Load** `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` (pre-quantized for
   ~2x faster startup vs on-the-fly bnb).
2. **Attach a single LoRA** (rank 16, all attention + MLP modules,
   `use_gradient_checkpointing="unsloth"`).
3. **SFT phase** - teach the JSON output format on synthetic
   `(prompt, valid mechanism)` pairs.
4. **GRPO phase** - reinforce with the same LoRA using a multiplicative
   reward (format + welfare + fairness + composite).
5. **Merge & push** - the LoRA is merged into the base and the **full
   16-bit model** is pushed to `kabilesh-c/daedalus-designer` so any
   consumer can load it with one line:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("kabilesh-c/daedalus-designer")
t = AutoTokenizer.from_pretrained("kabilesh-c/daedalus-designer")
```

Look for the sentinel line `[grpo v4] using single-adapter (no merge) approach`
in container logs to confirm the new code is live.

Mode is controlled by the `TRAIN_MODE` Space variable:

| `TRAIN_MODE` | SFT examples | GRPO steps | Wall time (T4) | Use case                          |
|--------------|--------------|------------|----------------|-----------------------------------|
| `smoke`      | 24           | 4          | ~3-5 min       | shake out build/runtime errors    |
| `short`      | 160          | 60         | ~8-12 min      | usable model                      |
| `long`       | 400          | 160        | ~30-45 min     | best results                      |
