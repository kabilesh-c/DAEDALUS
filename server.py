"""
DAEDALUS Server - FastAPI wrapper for the OpenEnv environment.
Hybrid Designer: Uses local GPU if available, falls back to HF API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import InferenceClient

from daedalus.env import DaedalusEnvironment
from daedalus.models import MechanismConfig

# Optional HF login
_HF_TOKEN = os.environ.get("HF_TOKEN")
if _HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=_HF_TOKEN, add_to_git_credential=False)
    except: pass

app = FastAPI(title="DAEDALUS Environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

environments: Dict[str, DaedalusEnvironment] = {}

# --- AI Designer Configuration ---
ADAPTER_ID = os.environ.get("DAEDALUS_ADAPTER", "kabilesh-c/daedalus-designer")
DESIGNER_MODEL = None
DESIGNER_TOKENIZER = None

DESIGNER_STATUS: Dict[str, Optional[str]] = {
    "status": "ready",
    "adapter": ADAPTER_ID,
    "device": "cpu",
    "error": None,
}

def _load_if_gpu():
    """Singleton loader for local GPU inference."""
    global DESIGNER_MODEL, DESIGNER_TOKENIZER
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.cuda.is_available():
            print(f"[designer] GPU detected! Loading {ADAPTER_ID} locally...")
            DESIGNER_TOKENIZER = AutoTokenizer.from_pretrained(ADAPTER_ID)
            DESIGNER_MODEL = AutoModelForCausalLM.from_pretrained(
                ADAPTER_ID, torch_dtype=torch.float16, device_map="auto"
            )
            DESIGNER_STATUS["device"] = "cuda (local)"
        else:
            DESIGNER_STATUS["device"] = "huggingface-inference-api"
    except Exception as e:
        DESIGNER_STATUS["device"] = "huggingface-inference-api"

@app.on_event("startup")
async def startup():
    _load_if_gpu()

def _get_inference_client():
    return InferenceClient(model=ADAPTER_ID, token=_HF_TOKEN)

@app.post("/api/design")
async def design_mechanism(observation: dict):
    try:
        user_prompt = _build_prompt(observation)
        
        # Scenario A: Local GPU Inference
        if DESIGNER_MODEL is not None:
            prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = DESIGNER_TOKENIZER(prompt, return_tensors="pt").to("cuda")
            out = DESIGNER_MODEL.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
            text = DESIGNER_TOKENIZER.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # Scenario B: Remote API Inference (using text_generation for better support)
        else:
            client = _get_inference_client()
            prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            text = client.text_generation(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                stop_sequences=["<|im_end|>", "}"]
            )
            # Ensure JSON integrity
            if "}" in text and not text.endswith("}"):
                text = text[:text.rfind("}")+1]

        j_start = text.find("{")
        j_end = text.rfind("}") + 1
        if j_start < 0: raise ValueError(f"AI returned no JSON: {text[:100]}")
        
        mech = json.loads(text[j_start:j_end])
        return {"mechanism": mech, "source": "ai", "status": "ready", "adapter": ADAPTER_ID}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e), "status": "error"})

def _build_prompt(observation: dict) -> str:
    lines = [
        "You are a mechanism designer. Respond with ONLY JSON.",
        f"Round: {observation.get('round_number', 0)}",
        "Recent Outcomes:"
    ]
    for o in observation.get("market_outcomes", [])[-3:]:
        lines.append(f"  W={o.get('welfare_ratio', 0):.2f} R={o.get('composite_reward', 0):.2f}")
    lines.append("Return JSON with: auction_type, reserve_price, shill_penalty, withdrawal_penalty, collusion_penalty, coalition_policy.")
    return "\n".join(lines)

@app.get("/api/designer/status")
async def status(): return DESIGNER_STATUS

class ActionRequest(BaseModel):
    action: dict

@app.post("/reset")
async def reset():
    env = DaedalusEnvironment()
    environments["default"] = env
    return {"observation": env.reset()}

@app.post("/step")
async def step(req: ActionRequest):
    env = environments.get("default")
    if not env: raise HTTPException(404, "Reset first")
    obs, rew, done, info = env.step(req.action)
    return {"observation": obs, "reward": rew, "done": done, "info": info}

# Static asset serving
static_dir = os.path.dirname(os.path.abspath(__file__))
@app.get("/")
async def index(): return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/{f}")
async def assets(f: str):
    p = os.path.join(static_dir, f)
    if os.path.isfile(p): return FileResponse(p)
    raise HTTPException(404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
