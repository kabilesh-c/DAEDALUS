"""
DAEDALUS Sub-Agents — Strategic agent implementations.
Each agent type exploits a different aspect of the mechanism.
"""
import math
import random
from typing import List, Optional
from .models import AgentState, MechanismConfig


class SubAgent:
    """Base class for all sub-agents."""

    def __init__(self, state: AgentState):
        self.state = state

    def generate_valuation(self) -> float:
        """Draw a new private valuation from the agent's distribution."""
        raise NotImplementedError

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        """Compute bid given the current mechanism and market history."""
        raise NotImplementedError

    def adapt(self, won: bool, mech: MechanismConfig, history: dict):
        """Adapt strategy based on outcome."""
        pass

    def check_participation(self, mech: MechanismConfig) -> bool:
        """Check if agent still wants to participate."""
        return self.state.active


class TruthfulBidder(SubAgent):
    """Bids true private valuation. No strategic model."""

    def generate_valuation(self) -> float:
        self.state.valuation = 0.3 + random.random() * 0.6
        return self.state.valuation

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        # Truthful: bid = valuation (dominant strategy under second-price)
        self.state.bid = self.state.valuation
        return self.state.bid

    def check_participation(self, mech: MechanismConfig) -> bool:
        return self.state.valuation >= mech.reserve_price


class BidShader(SubAgent):
    """
    First-order strategic agent. Shades bid below valuation.
    In first-price auctions with n symmetric bidders drawing uniformly on [0,1],
    the Bayes-Nash equilibrium bid is v × (n-1)/n.
    """

    def generate_valuation(self) -> float:
        self.state.valuation = 0.4 + random.random() * 0.5
        return self.state.valuation

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        v = self.state.valuation

        if mech.auction_type == "first_price":
            # BNE shading: bid = v × (n-1)/n
            shade_factor = (n_active - 1) / max(n_active, 2)

            # Adapt based on revealed information
            if mech.reveal_clearing_price:
                shade_factor -= 0.03  # Can calibrate more precisely
            if mech.reveal_competing_bids:
                shade_factor -= 0.05  # Full info enables strategic shading

            # Learn from price history
            prices = history.get("clearing_prices", [])
            if len(prices) > 2:
                avg_price = sum(prices[-5:]) / min(len(prices), 5)
                if avg_price < v * 0.7:
                    shade_factor -= 0.05  # Shade more aggressively

            shade_factor = max(0.5, min(shade_factor, 0.95))
            self.state.shade_factor = 1 - shade_factor
            self.state.bid = v * shade_factor
        else:
            # Second-price / VCG: truthful is dominant, minor bounded rationality noise
            self.state.shade_factor = 0.02 + random.random() * 0.03
            self.state.bid = v * (1 - self.state.shade_factor)

        return self.state.bid

    def adapt(self, won: bool, mech: MechanismConfig, history: dict):
        if won:
            self.state.shade_factor = min(self.state.shade_factor + 0.01, 0.35)
        else:
            self.state.shade_factor = max(self.state.shade_factor - 0.005, 0.02)

    def check_participation(self, mech: MechanismConfig) -> bool:
        return self.state.valuation >= mech.reserve_price * 1.1


class Colluder(SubAgent):
    """
    Coalition agent that coordinates with a partner.
    One bids high, partner bids very low, they rotate winning.
    Exploits information transparency for cartel enforcement.
    """

    def generate_valuation(self) -> float:
        self.state.valuation = 0.35 + random.random() * 0.5
        return self.state.valuation

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        v = self.state.valuation
        should_win = self.state.collusion_turn
        reserve = mech.reserve_price

        # Check if collusion is viable
        can_collude = mech.coalition_policy == "allow" or (
            mech.coalition_policy == "restrict" and random.random() > 0.5
        )
        penalty_risk = 0.4 if mech.collusion_penalty > 1 else 0.0

        if can_collude and random.random() > penalty_risk:
            if mech.reveal_winner_identity:
                # Strong collusion when winner is revealed (cartel can enforce)
                self.state.bid = reserve + 0.01 if should_win else reserve * 0.3
            else:
                if should_win:
                    target = reserve + 0.02 + random.random() * 0.05
                    self.state.bid = min(v, max(target, v * 0.5))
                else:
                    self.state.bid = reserve * 0.5
        else:
            # Compete honestly when collusion is too risky
            factor = 0.85 if mech.auction_type == "first_price" else 0.98
            self.state.bid = v * factor

        return self.state.bid

    def adapt(self, won: bool, mech: MechanismConfig, history: dict):
        # If facing high collusion penalties, gradually abandon collusion
        if mech.collusion_penalty > 1.5:
            self.state.shade_factor = min(self.state.shade_factor + 0.02, 0.5)

    def check_participation(self, mech: MechanismConfig) -> bool:
        return self.state.valuation >= mech.reserve_price * 0.8


class StrategicDropout(SubAgent):
    """
    Agent with participation threshold. Exits if expected surplus
    falls below outside option (reserve utility).
    """

    def generate_valuation(self) -> float:
        self.state.valuation = 0.15 + random.random() * 0.45
        return self.state.valuation

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        v = self.state.valuation
        expected_surplus = v - mech.reserve_price - 0.05

        # Update cumulative surplus tracker
        if expected_surplus > 0:
            self.state.cumulative_surplus += expected_surplus * 0.1
        else:
            self.state.cumulative_surplus -= 0.02

        # Check dropout condition
        if (self.state.cumulative_surplus < -self.state.dropout_threshold or
                mech.reserve_price > v * 0.9):
            self.state.active = False
            self.state.bid = 0.0
            return 0.0

        factor = 0.9 if mech.auction_type == "first_price" else 1.0
        self.state.bid = v * factor
        return self.state.bid

    def adapt(self, won: bool, mech: MechanismConfig, history: dict):
        # Re-entry: if conditions improve, come back
        if not self.state.active:
            participation = history.get("participation_rates", [])
            if len(participation) > 3:
                recent_avg = sum(participation[-3:]) / 3
                if recent_avg > 0.6 and mech.reserve_price < 0.3:
                    self.state.active = True
                    self.state.cumulative_surplus = 0.0

    def check_participation(self, mech: MechanismConfig) -> bool:
        return self.state.active


class BudgetExploiter(SubAgent):
    """
    Budget-constrained agent. Submits strategic bids within budget.
    Tests payment rule edge cases and attempts to exhaust market liquidity.
    """

    def generate_valuation(self) -> float:
        self.state.valuation = 0.2 + random.random() * 0.4
        return self.state.valuation

    def compute_bid(self, mech: MechanismConfig, n_active: int, history: dict) -> float:
        if self.state.budget <= 0:
            self.state.active = False
            self.state.bid = 0.0
            return 0.0

        v = self.state.valuation
        aggressiveness = 0.6 if self.state.budget > 1.5 else 0.3
        self.state.bid = min(v * aggressiveness, self.state.budget * 0.3)
        return self.state.bid

    def adapt(self, won: bool, mech: MechanismConfig, history: dict):
        # Budget gets deducted upon winning in the env step
        if self.state.budget <= 0:
            self.state.active = False

    def check_participation(self, mech: MechanismConfig) -> bool:
        return self.state.active and self.state.budget > 0


# ── Factory ──────────────────────────────────────────
AGENT_CLASSES = {
    "truthful": TruthfulBidder,
    "shader": BidShader,
    "colluder": Colluder,
    "dropout": StrategicDropout,
    "exploiter": BudgetExploiter,
}


def create_default_population(stage: int = 0) -> List[SubAgent]:
    """
    Create an 8-agent population based on the curriculum stage.
    Stage 0 is easiest (truthful baseline); Stage 4 is full adversarial.
    """
    # Base: 8 truthful agents
    agents = [
        TruthfulBidder(AgentState(agent_id=i, agent_type="truthful"))
        for i in range(8)
    ]

    # Stage 1: Add 2 Shaders
    if stage >= 1:
        agents[2] = BidShader(AgentState(agent_id=2, agent_type="shader"))
        agents[3] = BidShader(AgentState(agent_id=3, agent_type="shader"))

    # Stage 2: Add 2 strategic Dropouts
    if stage >= 2:
        agents[6] = StrategicDropout(AgentState(agent_id=6, agent_type="dropout", dropout_threshold=0.08))
        # Replacement at index 1
        agents[1] = StrategicDropout(AgentState(agent_id=1, agent_type="dropout", dropout_threshold=0.12))

    # Stage 3: Add 2 Colluders
    if stage >= 3:
        agents[4] = Colluder(AgentState(agent_id=4, agent_type="colluder", partner_id=5, collusion_turn=False))
        agents[5] = Colluder(AgentState(agent_id=5, agent_type="colluder", partner_id=4, collusion_turn=True))

    # Stage 4: Add 2 Budget Exploiters
    if stage >= 4:
        agents[7] = BudgetExploiter(AgentState(agent_id=7, agent_type="exploiter", budget=2.5))
        agents[0] = BudgetExploiter(AgentState(agent_id=0, agent_type="exploiter", budget=2.0))

    return agents
