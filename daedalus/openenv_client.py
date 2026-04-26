"""
DAEDALUS OpenEnv Client — Typed interface for the DAEDALUS environment.
Allows easy connection to the hosted HF Space.
"""
from typing import Optional
from openenv.core.env_client import EnvClient
from .openenv_models import DaedalusAction, DaedalusObservation, DaedalusState

class DaedalusEnvClient(EnvClient[DaedalusAction, DaedalusObservation, DaedalusState]):
    """
    Typed client for the DAEDALUS Mechanism Design environment.
    
    Usage:
        client = DaedalusEnvClient(base_url="https://kabilesh-c-daedalus-env.hf.space")
        obs = client.reset(seed=42).sync()
        obs = client.step(DaedalusAction(auction_type="second_price", ...)).sync()
    """
    def __init__(self, base_url: str, token: Optional[str] = None):
        super().__init__(
            base_url=base_url,
            token=token,
            action_type=DaedalusAction,
            observation_type=DaedalusObservation,
            state_type=DaedalusState
        )
