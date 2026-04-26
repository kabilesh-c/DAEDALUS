"""
DAEDALUS Server — FastAPI wrapper for the OpenEnv environment.
Deploy as a Hugging Face Space or run locally with uvicorn.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from daedalus.env import DaedalusEnvironment
from daedalus.models import MechanismConfig

app = FastAPI(
    title="DAEDALUS Environment",
    description="Mechanism Design via Adversarial RL — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (per-session in production)
environments: Dict[str, DaedalusEnvironment] = {}

# --- AI Designer Loading ---
BASE_MODEL_ID = os.environ.get("DAEDALUS_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_ID = os.environ.get("DAEDALUS_ADAPTER", "kabilesh-c/daedalus-designer")

DESIGNER_MODEL = None
DESIGNER_TOKENIZER = None


def load_designer():
    """Lazy-load base model + LoRA adapter on first /api/design call."""
    global DESIGNER_MODEL, DESIGNER_TOKENIZER
    if DESIGNER_MODEL is None:
        print(f"[designer] loading base={BASE_MODEL_ID} adapter={ADAPTER_ID} ...")

        DESIGNER_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if DESIGNER_TOKENIZER.pad_token is None:
            DESIGNER_TOKENIZER.pad_token = DESIGNER_TOKENIZER.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        DESIGNER_MODEL = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        DESIGNER_MODEL.eval()
        print("[designer] ready")
    return DESIGNER_MODEL, DESIGNER_TOKENIZER


def _build_prompt(observation: dict) -> str:
    """Mirror train_hf.py::_format_prompt so inference matches training."""
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
        # Clean up finished sessions
        del environments[req.session_id]

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.post("/api/design")
async def design_mechanism(observation: dict):
    """Ask the trained AI Designer for a mechanism. Falls back to default if training is incomplete."""
    try:
        model, tokenizer = load_designer()

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
                return mech
            except json.JSONDecodeError:
                pass

    except Exception as e:
        print(f"[designer] AI generation unavailable (likely still training): {e}")

    # Graceful fallback to a robust default mechanism
    return _DEFAULT_MECHANISM


@app.get("/state")
async def state(session_id: str = "default") -> dict:
    """Get current observable state."""
    if session_id not in environments:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"state": environments[session_id].state()}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "environment": "daedalus", "version": "1.0.0"}


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
