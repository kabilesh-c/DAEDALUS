"""
DAEDALUS OpenEnv-compliant data models.

These Pydantic models subclass the official OpenEnv types from
`openenv.core.env_server.types`, so the environment satisfies the
hackathon's "Use OpenEnv (latest release). Build on top of the
framework" requirement properly.

`DaedalusObservation.reward` and `.done` are inherited from the OpenEnv
`Observation` base, so a single object captures everything a Gymnasium-
style step would normally return as a tuple.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


AuctionType = Literal["first_price", "second_price", "vcg"]
CoalitionPolicy = Literal[
    "allow",
    "restrict",
    "penalize_suspected",
    "penalize_confirmed",
]


class DaedalusAction(Action):
    """The mechanism the LLM designer is proposing for the next 5-round window."""

    auction_type: AuctionType = Field(
        default="second_price",
        description="Pricing rule for clearing the auction.",
    )
    reserve_price: float = Field(
        default=0.10, ge=0.0, le=0.9,
        description="Reserve price below which bids are rejected.",
    )

    reveal_reserve: bool = Field(default=False)
    reveal_competing_bids: bool = Field(default=False)
    reveal_winner_identity: bool = Field(default=False)
    reveal_clearing_price: bool = Field(default=True)
    reveal_bid_distribution: bool = Field(default=False)

    shill_penalty: float = Field(default=0.0, ge=0.0, le=3.0)
    withdrawal_penalty: float = Field(default=0.0, ge=0.0, le=3.0)
    collusion_penalty: float = Field(default=0.0, ge=0.0, le=3.0)

    coalition_policy: CoalitionPolicy = Field(default="allow")


class DaedalusObservation(Observation):
    """
    What the LLM designer agent sees at every step.

    `reward` and `done` are inherited from the OpenEnv `Observation`
    base, so this single object replaces the Gym-style
    `(obs, reward, done, info)` tuple.
    """

    mechanism_config: Dict[str, Any] = Field(default_factory=dict)
    market_outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    population_proxies: Dict[str, Any] = Field(default_factory=dict)
    round_number: int = Field(default=0, ge=0)
    episode_length: int = Field(default=50, ge=1)
    curriculum_stage: int = Field(default=0, ge=0)


class DaedalusState(State):
    """Server-side state, exposed via Environment.state for introspection."""

    n_active_agents: int = Field(default=0, ge=0)
    n_total_agents: int = Field(default=0, ge=0)
    cumulative_welfare: float = Field(default=0.0)
    last_composite_reward: float = Field(default=0.0)
    curriculum_stage: int = Field(default=0, ge=0)
