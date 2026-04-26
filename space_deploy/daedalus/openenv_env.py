"""
DAEDALUS OpenEnv-compliant environment.

This is the canonical class to use. It inherits from
`openenv.core.env_server.interfaces.Environment`, exposes the standard
Gymnasium-style API (`reset`, `step`, `state`), and returns typed
Pydantic observations whose `.reward` and `.done` fields replace the
legacy `(obs, reward, done, info)` tuple.

The legacy `DaedalusEnvironment` (in `daedalus.env`) is preserved for
the demo dashboard and earlier notebooks so existing code keeps working.
This class wraps the legacy env so there is exactly one source of truth
for the auction / sub-agent / reward logic.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from .env import DaedalusEnvironment as _LegacyEnv
from .openenv_models import DaedalusAction, DaedalusObservation, DaedalusState
from .rubrics import DaedalusCompositeRubric


class DaedalusOpenEnv(Environment[DaedalusAction, DaedalusObservation, DaedalusState]):
    """
    OpenEnv-compliant DAEDALUS mechanism design environment.

    Theme #1 - Multi-Agent Interactions (primary):
        An LLM designer agent picks auction rules; a population of
        adaptive sub-agents (truthful bidders, shaders, colluders,
        strategic dropouts, budget exploiters) probe the mechanism for
        exploits. The composite reward
            R = welfare * fairness * participation * stability
        forces multi-objective tradeoffs that single-objective gaming
        cannot satisfy.

    Theme #4 - Self-Improvement (secondary):
        A 5-stage curriculum gradually mixes harder sub-agent types
        into the population so the designer self-improves against
        progressively stronger adversaries.

    Usage::

        from daedalus import DaedalusOpenEnv, DaedalusAction

        env = DaedalusOpenEnv(n_agents=8, episode_length=20)
        obs = env.reset(seed=42)

        action = DaedalusAction(
            auction_type="second_price",
            reserve_price=0.15,
            collusion_penalty=1.5,
            coalition_policy="penalize_suspected",
        )
        obs = env.step(action)
        print(obs.reward, obs.done)

        print(env.state.step_count, env.state.n_active_agents)
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        n_agents: int = 8,
        episode_length: int = 50,
        rounds_per_step: int = 5,
        curriculum_stage: int = 0,
        **kwargs: Any,
    ):
        super().__init__(rubric=DaedalusCompositeRubric(), **kwargs)
        self._env = _LegacyEnv(
            n_agents=n_agents,
            episode_length=episode_length,
            rounds_per_step=rounds_per_step,
            curriculum_stage=curriculum_stage,
        )
        self._episode_id: Optional[str] = None
        self._last_reward: float = 0.0

    # -------------------------------------------------------------------
    # OpenEnv abstract API
    # -------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DaedalusObservation:
        if seed is not None:
            random.seed(seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._last_reward = 0.0
        obs_dict = self._env.reset()
        return self._make_observation(obs_dict, reward=None, done=False)

    def step(
        self,
        action: DaedalusAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DaedalusObservation:
        action_dict = self._action_to_dict(action)
        obs_dict, reward, done, info = self._env.step(action_dict)
        self._last_reward = float(reward)
        observation = self._make_observation(
            obs_dict,
            reward=float(reward),
            done=bool(done),
            info=info,
        )
        return observation

    @property
    def state(self) -> DaedalusState:
        active = sum(1 for a in self._env.agents if a.state.active)
        total = len(self._env.agents)
        cumulative_welfare = sum(self._env.welfare_history)
        return DaedalusState(
            episode_id=self._episode_id,
            step_count=self._env.round,
            n_active_agents=active,
            n_total_agents=total,
            cumulative_welfare=cumulative_welfare,
            last_composite_reward=self._last_reward,
            curriculum_stage=self._env.curriculum_stage,
        )

    # -------------------------------------------------------------------
    # Convenience: legacy tuple-style step for demo / notebook code.
    # NOT part of the OpenEnv contract; kept so existing scripts keep
    # working without rewriting them.
    # -------------------------------------------------------------------
    def step_tuple(self, action: Any):
        if isinstance(action, DaedalusAction):
            action_dict = self._action_to_dict(action)
        else:
            action_dict = action
        return self._env.step(action_dict)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _action_to_dict(action: Any) -> dict:
        if isinstance(action, DaedalusAction):
            return action.model_dump(exclude={"metadata"})
        if isinstance(action, dict):
            return action
        if hasattr(action, "to_dict"):
            return action.to_dict()
        raise TypeError(
            f"DaedalusOpenEnv.step expected DaedalusAction or dict, "
            f"got {type(action).__name__}"
        )

    def _make_observation(
        self,
        obs_dict: dict,
        *,
        reward: Optional[float],
        done: bool,
        info: Optional[dict] = None,
    ) -> DaedalusObservation:
        metadata = info or {}
        return DaedalusObservation(
            mechanism_config=obs_dict.get("mechanism_config", {}),
            market_outcomes=obs_dict.get("market_outcomes", []),
            population_proxies=obs_dict.get("population_proxies", {}),
            round_number=int(obs_dict.get("round_number", 0)),
            episode_length=int(obs_dict.get("episode_length", 50)),
            curriculum_stage=int(obs_dict.get("curriculum_stage", 0)),
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def close(self) -> None:
        self._env = None
