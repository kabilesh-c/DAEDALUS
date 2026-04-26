"""
DAEDALUS Environment — OpenEnv-compliant RL environment.
Trains an LLM to design market mechanisms against adversarial sub-agents.

OpenEnv API:
    env = DaedalusEnvironment()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    state = env.state()
"""
import json
import math
import random
from typing import Tuple, Dict, Any, List, Optional

from .models import MechanismConfig, MarketOutcome, Observation, AgentState
from .agents import (
    SubAgent, create_default_population,
    TruthfulBidder, BidShader, Colluder, StrategicDropout, BudgetExploiter,
)
from .rewards import (
    compute_welfare_ratio, compute_fairness, compute_gini,
    compute_participation_rate, compute_stability,
    compute_collusion_signal, compute_composite_reward,
)


class DaedalusEnvironment:
    """
    DAEDALUS: Mechanism Design via Adversarial RL

    An OpenEnv-compliant environment where an LLM agent designs
    market mechanisms while strategic sub-agents probe for exploits.
    Reward is the product of welfare, fairness, participation, and stability.

    Themes:
        #1 Multi-Agent Interactions (primary)
        #4 Self-Improvement via curriculum (secondary)
    """

    def __init__(
        self,
        n_agents: int = 8,
        episode_length: int = 50,
        rounds_per_step: int = 5,
        curriculum_stage: int = 0,
    ):
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.rounds_per_step = rounds_per_step
        self.curriculum_stage = curriculum_stage

        # Internal state
        self.agents: List[SubAgent] = []
        self.mechanism = MechanismConfig()
        self.round = 0
        self.history: List[MarketOutcome] = []
        self.welfare_history: List[float] = []
        self.participation_history: List[float] = []
        self.clearing_prices: List[float] = []
        self.collusion_rotation = 0
        self.done = False

    def reset(self) -> dict:
        """
        Start a fresh episode.
        Samples new population, resets mechanism to default, clears history.

        Returns:
            Observation dict for the LLM agent.
        """
        self.agents = create_default_population(stage=self.curriculum_stage)
        self.mechanism = MechanismConfig()
        self.round = 0
        self.history = []
        self.welfare_history = []
        self.participation_history = []
        self.clearing_prices = []
        self.collusion_rotation = 0
        self.done = False

        # Generate initial valuations
        for agent in self.agents:
            agent.generate_valuation()

        # Run warm-up rounds with default mechanism
        for _ in range(3):
            self._run_market_round()

        return self._get_observation().to_dict()

    def step(self, action: Any) -> Tuple[dict, float, bool, dict]:
        """
        Apply a mechanism configuration action and run market rounds.

        Args:
            action: Either a MechanismConfig, dict, or JSON string.

        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation().to_dict(), 0.0, True, {"error": "Episode finished"}

        # Parse action
        if isinstance(action, MechanismConfig):
            self.mechanism = action
        elif isinstance(action, dict):
            self.mechanism = MechanismConfig.from_dict(action)
        elif isinstance(action, str):
            try:
                self.mechanism = MechanismConfig.from_json(action)
            except (json.JSONDecodeError, TypeError):
                return self._get_observation().to_dict(), 0.0, False, {"error": "Invalid JSON"}
        else:
            return self._get_observation().to_dict(), 0.0, False, {"error": "Invalid action type"}

        self.round += 1

        # Run market rounds
        round_outcomes = []
        for _ in range(self.rounds_per_step):
            outcome = self._run_market_round()
            round_outcomes.append(outcome)

        # Compute reward from post-adaptation rounds (rounds 3-5)
        eval_outcomes = round_outcomes[2:] if len(round_outcomes) >= 3 else round_outcomes
        avg_reward = sum(o.composite_reward for o in eval_outcomes) / max(len(eval_outcomes), 1)

        # Check termination
        self.done = self.round >= self.episode_length
        if self._check_collapse():
            self.done = True

        # Build info
        latest = round_outcomes[-1] if round_outcomes else MarketOutcome()
        info = {
            "round": self.round,
            "mechanism": self.mechanism.to_dict(),
            "welfare_ratio": latest.welfare_ratio,
            "gini_coefficient": latest.gini_coefficient,
            "participation_rate": latest.participation_rate,
            "stability_score": latest.stability_score,
            "composite_reward": avg_reward,
            "collusion_signal": latest.collusion_signal,
            "active_agents": sum(1 for a in self.agents if a.state.active),
            "dropout_count": latest.dropout_count,
        }

        return self._get_observation().to_dict(), avg_reward, self.done, info

    def state(self) -> dict:
        """Return current observable state (POMDP observation)."""
        return self._get_observation().to_dict()

    def _run_market_round(self) -> MarketOutcome:
        """Execute one auction round and return outcome."""
        mech = self.mechanism
        active_agents = [a for a in self.agents if a.state.active]
        n_active = len(active_agents)
        n_total = len(self.agents)

        # Generate new valuations
        for agent in active_agents:
            agent.generate_valuation()

        # Collect bids
        history = {
            "clearing_prices": self.clearing_prices,
            "participation_rates": self.participation_history,
        }
        for agent in self.agents:
            if agent.state.active:
                agent.compute_bid(mech, n_active, history)
            else:
                agent.state.bid = 0.0

        # Filter valid bids (above reserve)
        valid_bidders = [
            a for a in active_agents
            if a.state.bid >= mech.reserve_price
        ]
        valid_bidders.sort(key=lambda a: a.state.bid, reverse=True)

        # Allocation & Payment
        winner = None
        payment = 0.0
        winner_valuation = 0.0

        if valid_bidders:
            winner = valid_bidders[0]
            winner_valuation = winner.state.valuation

            if mech.auction_type == "first_price":
                payment = winner.state.bid
            elif mech.auction_type in ("second_price", "vcg"):
                payment = valid_bidders[1].state.bid if len(valid_bidders) > 1 else mech.reserve_price

            winner.state.surplus = winner.state.valuation - payment
            winner.state.wins += 1

            # Budget deduction for exploiter
            if winner.state.agent_type == "exploiter":
                winner.state.budget -= payment

            # Collusion penalty
            if mech.collusion_penalty > 0 and winner.state.agent_type == "colluder":
                partner = self._get_partner(winner)
                if partner and partner.state.active and partner.state.bid < mech.reserve_price * 0.8:
                    penalty_amount = payment * mech.collusion_penalty * 0.3
                    winner.state.surplus -= penalty_amount

        self.collusion_rotation += 1
        # Toggle collusion turns
        for a in self.agents:
            if a.state.agent_type == "colluder":
                a.state.collusion_turn = not a.state.collusion_turn

        # Adapt agents
        for agent in self.agents:
            won = winner is not None and agent.state.agent_id == winner.state.agent_id
            agent.adapt(won, mech, history)

        # Compute metrics
        all_valuations = [a.state.valuation for a in self.agents]
        surplus_dist = [max(a.state.surplus, 0) for a in active_agents]
        bids = [a.state.bid for a in active_agents if a.state.bid > 0]
        wins = [a.state.wins for a in self.agents]

        welfare = compute_welfare_ratio(winner_valuation, all_valuations)
        fairness = compute_fairness(surplus_dist)
        participation = compute_participation_rate(n_active, n_total)
        self.welfare_history.append(welfare)
        self.participation_history.append(participation)
        self.clearing_prices.append(payment)
        stability = compute_stability(self.welfare_history)
        collusion = compute_collusion_signal(bids, wins)

        # Anti-collusion multiplier
        collusion_mult = max(0.5, 1.0 - collusion * 0.5) if collusion > 0.5 else 1.0

        composite = compute_composite_reward(
            welfare, fairness, participation, stability,
            collusion_penalty=collusion_mult,
        )

        outcome = MarketOutcome(
            welfare_ratio=welfare,
            gini_coefficient=compute_gini(surplus_dist),
            participation_rate=participation,
            clearing_price_mean=payment,
            clearing_price_std=0.0,
            dropout_count=n_total - n_active,
            collusion_signal=collusion,
            stability_score=stability,
            composite_reward=composite,
        )
        self.history.append(outcome)
        return outcome

    def _get_observation(self) -> Observation:
        """Build observation for the LLM agent."""
        active = [a for a in self.agents if a.state.active]
        bids = [a.state.bid for a in active if a.state.bid > 0]

        # Population proxies (observable aggregate statistics)
        proxies = {
            "active_count": len(active),
            "total_agents": len(self.agents),
            "bid_mean": sum(bids) / max(len(bids), 1),
            "bid_std": (sum((b - sum(bids)/max(len(bids),1))**2 for b in bids) / max(len(bids),1)) ** 0.5 if bids else 0,
            "bid_correlation": compute_collusion_signal(
                bids, [a.state.wins for a in self.agents]
            ),
            "rotation_entropy": self._compute_rotation_entropy(),
            "dropout_rate": 1 - len(active) / max(len(self.agents), 1),
            "budget_exhaustion_count": sum(
                1 for a in self.agents
                if a.state.agent_type == "exploiter" and a.state.budget <= 0
            ),
            "clearing_price_trend": self._compute_trend(self.clearing_prices),
        }

        return Observation(
            mechanism_config=self.mechanism.to_dict(),
            market_outcomes=[o.to_dict() for o in self.history[-20:]],
            population_proxies=proxies,
            round_number=self.round,
            episode_length=self.episode_length,
            curriculum_stage=self.curriculum_stage,
        )

    def _get_partner(self, agent: SubAgent) -> Optional[SubAgent]:
        """Get collusion partner for a colluder agent."""
        if agent.state.partner_id is not None:
            for a in self.agents:
                if a.state.agent_id == agent.state.partner_id:
                    return a
        return None

    def _compute_rotation_entropy(self) -> float:
        """Compute entropy of winner distribution (cartel detection)."""
        wins = [a.state.wins for a in self.agents]
        total = sum(wins)
        if total <= 0:
            return 1.0
        entropy = 0.0
        for w in wins:
            if w > 0:
                p = w / total
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(wins)) if len(wins) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 1.0

    def _compute_trend(self, values: List[float], window: int = 5) -> float:
        """Compute first derivative (trend) of recent values."""
        if len(values) < 2:
            return 0.0
        recent = values[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)

    def _check_collapse(self) -> bool:
        """Check if mechanism has collapsed (zero participation)."""
        active = sum(1 for a in self.agents if a.state.active)
        return active == 0
