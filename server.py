"""
DAEDALUS Server - FastAPI wrapper for the OpenEnv environment.
Deploy as a Hugging Face Space or run locally with uvicorn.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from daedalus.env import DaedalusEnvironment
from daedalus.models import MechanismConfig

# Optional HF login when a token is present (private repos / rate limits)
_HF_TOKEN = os.environ.get("HF_TOKEN")
if _HF_TOKEN:
    try:
        from huggingface_hub import login

        login(token=_HF_TOKEN, add_to_git_credential=False)
        print("[auth] logged into Hugging Face from HF_TOKEN env var")
    except Exception as e:
        print(f"[auth] login failed (continuing anonymously): {e}")


app = FastAPI(
    title="DAEDALUS Environment",
    description="Mechanism Design via Adversarial RL - OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

environments: Dict[str, DaedalusEnvironment] = {}

# --- AI Designer Loading ---
BASE_MODEL_ID = os.environ.get("DAEDALUS_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_ID = os.environ.get("DAEDALUS_ADAPTER", "kabilesh-c/daedalus-designer")

DESIGNER_MODEL = None
DESIGNER_TOKENIZER = None

# status: "idle" | "loading" | "ready" | "error"
DESIGNER_STATUS: Dict[str, Optional[str]] = {
    "status": "idle",
    "base_model": BASE_MODEL_ID,
    "adapter": ADAPTER_ID,
    "device": None,
    "error": None,
}
_DESIGNER_LOCK = threading.Lock()


def _from_pretrained_compat(cls, model_id: str, **kwargs):
    """transformers >=4.45 prefers `dtype=`, older builds want `torch_dtype=`."""
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return cls.from_pretrained(model_id, **kwargs)


def _load_designer_blocking() -> None:
    """Load `kabilesh-c/daedalus-designer`.

    The repo may be either:
      (a) a full merged model (v4 trainer output)  -> single from_pretrained
      (b) a LoRA adapter sitting on Qwen2.5-0.5B-Instruct (v3 trainer output)
          -> base + PeftModel.from_pretrained

    We try (a) first, fall back to (b) if it looks like an adapter repo.
    """
    global DESIGNER_MODEL, DESIGNER_TOKENIZER

    with _DESIGNER_LOCK:
        if DESIGNER_MODEL is not None:
            return
        DESIGNER_STATUS["status"] = "loading"
        DESIGNER_STATUS["error"] = None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        # ---- Path A: full merged model in ADAPTER_ID itself ---------------
        try:
            print(f"[designer] trying full-model load from {ADAPTER_ID} ...")
            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = _from_pretrained_compat(
                AutoModelForCausalLM,
                ADAPTER_ID,
                dtype=dtype,
                device_map=device_map,
            )
            model.eval()
            DESIGNER_TOKENIZER = tokenizer
            DESIGNER_MODEL = model
            DESIGNER_STATUS["status"] = "ready"
            DESIGNER_STATUS["device"] = str(next(model.parameters()).device)
            print(f"[designer] ready (full model) on {DESIGNER_STATUS['device']}")
            return
        except Exception as e_full:
            print(f"[designer] full-model load failed ({e_full}); trying LoRA adapter on {BASE_MODEL_ID} ...")

        # ---- Path B: LoRA adapter on top of BASE_MODEL_ID -----------------
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = _from_pretrained_compat(
                AutoModelForCausalLM,
                BASE_MODEL_ID,
                dtype=dtype,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
            model.eval()
            DESIGNER_TOKENIZER = tokenizer
            DESIGNER_MODEL = model
            DESIGNER_STATUS["status"] = "ready"
            DESIGNER_STATUS["device"] = str(next(model.parameters()).device)
            print(f"[designer] ready (base + LoRA) on {DESIGNER_STATUS['device']}")
        except Exception as e:
            DESIGNER_STATUS["status"] = "error"
            DESIGNER_STATUS["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(f"[designer] LOAD FAILED: {DESIGNER_STATUS['error']}")


def _kickoff_designer_load() -> None:
    """Start loading in a background thread so /health works immediately."""
    if DESIGNER_STATUS["status"] in ("loading", "ready"):
        return
    t = threading.Thread(target=_load_designer_blocking, daemon=True)
    t.start()


@app.on_event("startup")
async def _startup_warmup() -> None:
    """Begin downloading + loading the designer the moment the server boots."""
    _kickoff_designer_load()


def _build_prompt(observation: dict) -> str:
    """Mirror train_hf.py::format_prompt so inference matches training."""
    lines = [
        "You are a mechanism designer for a market auction system.",
        "Analyze the current market state and design an optimal mechanism.",
        "",
        f"Round: {observation.get('round_number', 0)} / "
        f"{observation.get('episode_length', 50)}",
        "",
        "Your goal is to maximize the composite reward R = W x F x P x S",
    ]
    outcomes = observation.get("market_outcomes", [])
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


_DEFAULT_MECHANISM = {
    "auction_type": "second_price",
    "reserve_price": 0.10,
    "reveal_reserve": True,
    "reveal_competing_bids": False,
    "reveal_winner_identity": False,
    "reveal_clearing_price": True,
    "reveal_bid_distribution": False,
    "shill_penalty": 1.0,
    "withdrawal_penalty": 0.5,
    "collusion_penalty": 1.5,
    "coalition_policy": "penalize_suspected",
}


class ResetRequest(BaseModel):
    session_id: str = "default"
    n_agents: int = 8
    episode_length: int = 50
    curriculum_stage: int = 0


class StepRequest(BaseModel):
    session_id: str = "default"
    action: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


@app.post("/reset")
async def reset(req: ResetRequest) -> dict:
    """Start a fresh episode."""
    env = DaedalusEnvironment(
        n_agents=req.n_agents,
        episode_length=req.episode_length,
        curriculum_stage=req.curriculum_stage,
    )
    environments[req.session_id] = env
    obs = env.reset()
    return {"observation": obs, "session_id": req.session_id}


@app.post("/step")
async def step(req: StepRequest) -> StepResponse:
    """Take one step in the environment."""
    if req.session_id not in environments:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    env = environments[req.session_id]
    obs, reward, done, info = env.step(req.action)

    if done:
        del environments[req.session_id]

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/api/designer/status")
async def designer_status() -> dict:
    """Tell the frontend whether the AI designer is ready, loading, or errored."""
    return dict(DESIGNER_STATUS)


@app.post("/api/designer/warmup")
async def designer_warmup() -> dict:
    """Trigger (or retry) loading the designer."""
    if DESIGNER_STATUS["status"] == "error":
        # Allow retry by clearing state
        DESIGNER_STATUS["status"] = "idle"
        DESIGNER_STATUS["error"] = None
    _kickoff_designer_load()
    return dict(DESIGNER_STATUS)


@app.post("/api/design")
async def design_mechanism(observation: dict) -> dict:
    """
    Ask the trained AI Designer for a mechanism.

    Always returns:
        {
          "mechanism": {...},
          "source":    "ai" | "fallback",
          "status":    "ready" | "loading" | "error",
          "error":     <string or null>,
        }
    """
    status = DESIGNER_STATUS["status"]

    if status in ("idle", "loading"):
        if status == "idle":
            _kickoff_designer_load()
        return {
            "mechanism": _DEFAULT_MECHANISM,
            "source": "fallback",
            "status": DESIGNER_STATUS["status"],
            "error": "Designer is still loading; using safe default for this step.",
        }

    if status == "error" or DESIGNER_MODEL is None or DESIGNER_TOKENIZER is None:
        return {
            "mechanism": _DEFAULT_MECHANISM,
            "source": "fallback",
            "status": "error",
            "error": DESIGNER_STATUS["error"] or "Designer model is not available.",
        }

    try:
        model = DESIGNER_MODEL
        tokenizer = DESIGNER_TOKENIZER

        user_prompt = _build_prompt(observation)
        chat = [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        j_start = completion.find("{")
        j_end = completion.rfind("}") + 1
        if j_start >= 0 and j_end > j_start:
            try:
                mech = json.loads(completion[j_start:j_end])
                if isinstance(mech, dict):
                    return {
                        "mechanism": mech,
                        "source": "ai",
                        "status": "ready",
                        "error": None,
                    }
            except json.JSONDecodeError as je:
                return {
                    "mechanism": _DEFAULT_MECHANISM,
                    "source": "fallback",
                    "status": "ready",
                    "error": f"AI returned malformed JSON: {je}",
                }

        return {
            "mechanism": _DEFAULT_MECHANISM,
            "source": "fallback",
            "status": "ready",
            "error": "AI completion contained no JSON object.",
        }

    except Exception as e:
        traceback.print_exc()
        print(f"[designer] inference failed: {e}")
        return {
            "mechanism": _DEFAULT_MECHANISM,
            "source": "fallback",
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        }


@app.get("/state")
async def state(session_id: str = "default") -> dict:
    """Get current observable state."""
    if session_id not in environments:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"state": environments[session_id].state()}


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "environment": "daedalus",
        "version": "1.0.0",
        "designer": DESIGNER_STATUS["status"],
    }


# Serve static demo files
static_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(static_dir, "index.html")):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_demo():
        return FileResponse(os.path.join(static_dir, "index.html"))

    _STATIC_ASSETS = {
        "styles.css": "text/css",
        "app.js": "application/javascript",
        "favicon.ico": "image/x-icon",
        "favicon.png": "image/png",
    }

    for _name, _mime in _STATIC_ASSETS.items():
        _path = os.path.join(static_dir, _name)
        if not os.path.exists(_path):
            continue

        def _make_handler(path: str, mime: str):
            async def _handler():
                return FileResponse(path, media_type=mime)
            return _handler

        app.add_api_route(
            f"/{_name}",
            _make_handler(_path, _mime),
            methods=["GET"],
            include_in_schema=False,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
