"""
DAEDALUS Data Models — Typed dataclasses for mechanism design environment.
"""
from dataclasses import dataclass, field
from typing import Literal, List, Optional
import json


@dataclass
class MechanismConfig:
    """Full mechanism specification — the designer's action."""
    auction_type: Literal["first_price", "second_price", "vcg"] = "second_price"
    reveal_reserve: bool = False
    reveal_competing_bids: bool = False
    reveal_winner_identity: bool = False
    reveal_clearing_price: bool = True
    reveal_bid_distribution: bool = False
    reserve_price: float = 0.10
    shill_penalty: float = 0.0
    withdrawal_penalty: float = 0.0
    collusion_penalty: float = 0.0
    coalition_policy: Literal["allow", "restrict", "penalize_suspected", "penalize_confirmed"] = "allow"

    def to_dict(self):
        return {
            "auction_type": self.auction_type,
            "reveal_reserve": self.reveal_reserve,
            "reveal_competing_bids": self.reveal_competing_bids,
            "reveal_winner_identity": self.reveal_winner_identity,
            "reveal_clearing_price": self.reveal_clearing_price,
            "reveal_bid_distribution": self.reveal_bid_distribution,
            "reserve_price": self.reserve_price,
            "shill_penalty": self.shill_penalty,
            "withdrawal_penalty": self.withdrawal_penalty,
            "collusion_penalty": self.collusion_penalty,
            "coalition_policy": self.coalition_policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MechanismConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "MechanismConfig":
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MarketOutcome:
    """Aggregate market outcome from one step (5 market rounds)."""
    welfare_ratio: float = 0.0
    gini_coefficient: float = 0.0
    participation_rate: float = 1.0
    clearing_price_mean: float = 0.0
    clearing_price_std: float = 0.0
    dropout_count: int = 0
    collusion_signal: float = 0.0
    stability_score: float = 1.0
    composite_reward: float = 0.0

    def to_dict(self):
        return {
            "welfare_ratio": self.welfare_ratio,
            "gini_coefficient": self.gini_coefficient,
            "participation_rate": self.participation_rate,
            "clearing_price_mean": self.clearing_price_mean,
            "clearing_price_std": self.clearing_price_std,
            "dropout_count": self.dropout_count,
            "collusion_signal": self.collusion_signal,
            "stability_score": self.stability_score,
            "composite_reward": self.composite_reward,
        }


@dataclass
class AgentState:
    """Internal state of a sub-agent (hidden from the designer)."""
    agent_id: int = 0
    agent_type: Literal["truthful", "shader", "colluder", "dropout", "exploiter"] = "truthful"
    valuation: float = 0.0
    bid: float = 0.0
    surplus: float = 0.0
    active: bool = True
    budget: float = float("inf")
    shade_factor: float = 0.0
    wins: int = 0
    # Colluder-specific
    partner_id: Optional[int] = None
    collusion_turn: bool = False
    # Dropout-specific
    dropout_threshold: float = 0.08
    cumulative_surplus: float = 0.0


@dataclass
class Observation:
    """What the designer agent sees at each timestep."""
    mechanism_config: dict = field(default_factory=dict)
    market_outcomes: List[dict] = field(default_factory=list)
    population_proxies: dict = field(default_factory=dict)
    round_number: int = 0
    episode_length: int = 50
    curriculum_stage: int = 0

    def to_dict(self):
        return {
            "mechanism_config": self.mechanism_config,
            "market_outcomes": self.market_outcomes,
            "population_proxies": self.population_proxies,
            "round_number": self.round_number,
            "episode_length": self.episode_length,
            "curriculum_stage": self.curriculum_stage,
        }

    def to_prompt(self) -> str:
        """Convert observation to natural language prompt for LLM."""
        lines = [
            "You are a mechanism designer. Analyze the current market state and propose an optimal mechanism configuration.",
            "",
            f"Round: {self.round_number} / {self.episode_length}",
            f"Curriculum Stage: {self.curriculum_stage}",
            "",
            "Current Mechanism:",
            f"  Auction Type: {self.mechanism_config.get('auction_type', 'second_price')}",
            f"  Reserve Price: {self.mechanism_config.get('reserve_price', 0.1):.3f}",
            f"  Coalition Policy: {self.mechanism_config.get('coalition_policy', 'allow')}",
            "",
            "Recent Market Outcomes:",
        ]

        for i, outcome in enumerate(self.market_outcomes[-5:]):
            lines.append(
                f"  Round {self.round_number - len(self.market_outcomes) + i + 1}: "
                f"W={outcome.get('welfare_ratio', 0):.3f} "
                f"F={1 - outcome.get('gini_coefficient', 0):.3f} "
                f"P={outcome.get('participation_rate', 1):.3f} "
                f"R={outcome.get('composite_reward', 0):.3f}"
            )

        proxies = self.population_proxies
        if proxies:
            lines.extend([
                "",
                "Population Signals:",
                f"  Active Bidders: {proxies.get('active_count', 8)}",
                f"  Bid Correlation: {proxies.get('bid_correlation', 0):.3f}",
                f"  Winner Rotation Entropy: {proxies.get('rotation_entropy', 1):.3f}",
                f"  Dropout Rate: {proxies.get('dropout_rate', 0):.3f}",
            ])

        lines.extend([
            "",
            "Respond with a JSON mechanism configuration:",
            '{"auction_type": "first_price|second_price|vcg", "reserve_price": float, '
            '"reveal_reserve": bool, "reveal_competing_bids": bool, "reveal_winner_identity": bool, '
            '"reveal_clearing_price": bool, "reveal_bid_distribution": bool, '
            '"shill_penalty": float, "withdrawal_penalty": float, "collusion_penalty": float, '
            '"coalition_policy": "allow|restrict|penalize_suspected|penalize_confirmed"}',
        ])
        return "\n".join(lines)
