"""DAEDALUS Environment Package - Mechanism Design via Adversarial RL.

Public API:

OpenEnv-compliant (preferred for training, evaluation, judges):
    from daedalus import DaedalusOpenEnv, DaedalusAction, DaedalusObservation, DaedalusState

Legacy tuple-returning API (for the demo dashboard and old notebooks):
    from daedalus import DaedalusEnvironment
"""

from .env import DaedalusEnvironment
from .models import MechanismConfig, MarketOutcome, AgentState

try:
    from .openenv_env import DaedalusOpenEnv
    from .openenv_models import (
        DaedalusAction,
        DaedalusObservation,
        DaedalusState,
    )

    _OPENENV_AVAILABLE = True
except Exception:  # noqa: BLE001 - openenv-core not installed
    _OPENENV_AVAILABLE = False

__version__ = "2.0.0"

__all__ = [
    "DaedalusEnvironment",
    "MechanismConfig",
    "MarketOutcome",
    "AgentState",
]

if _OPENENV_AVAILABLE:
    __all__ += [
        "DaedalusOpenEnv",
        "DaedalusAction",
        "DaedalusObservation",
        "DaedalusState",
    ]
