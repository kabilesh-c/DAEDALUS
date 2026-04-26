"""
DAEDALUS Reward Functions — Multiple independent reward signals.
Multiplicative composite prevents single-objective gaming.
"""
import math
from typing import List, Dict


def compute_welfare_ratio(winner_valuation: float, all_valuations: List[float]) -> float:
    """
    W(t) — Social Welfare Ratio
    Total allocated utility / theoretical maximum welfare.
    Are goods going to agents who value them most?

    Range: [0, 1]
    """
    if not all_valuations:
        return 0.0
    theoretical_max = max(all_valuations)
    if theoretical_max <= 0:
        return 0.0
    return min(winner_valuation / theoretical_max, 1.0)


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient of a distribution."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total <= 0:
        return 0.0

    numerator = 0.0
    for i, v in enumerate(sorted_vals):
        numerator += (2 * (i + 1) - n - 1) * v
    return numerator / (n * total)


def compute_fairness(payment_distribution: List[float]) -> float:
    """
    F(t) — Fairness Score
    1 - Gini coefficient of payment/surplus distribution.
    Prevents one agent from capturing all surplus.

    Range: [0, 1]
    """
    gini = compute_gini(payment_distribution)
    return max(0.0, 1.0 - gini)


def compute_participation_rate(active_agents: int, total_agents: int) -> float:
    """
    P(t) — Participation Rate
    Fraction of eligible agents who submitted bids.
    Near zero drives composite reward to zero regardless of other terms.

    Range: [0, 1]
    """
    if total_agents <= 0:
        return 0.0
    return active_agents / total_agents


def compute_stability(welfare_history: List[float], window: int = 5) -> float:
    """
    S(t) — Stability Bonus
    1 - normalized standard deviation of welfare over last N rounds.
    Rewards consistent performance over high-variance mechanisms.

    Range: [0, 1]
    """
    if len(welfare_history) < window:
        return 1.0  # Give benefit of the doubt early on

    recent = welfare_history[-window:]
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    std_dev = math.sqrt(variance)
    return max(0.0, 1.0 - std_dev * 3)  # Normalize


def compute_collusion_signal(bids: List[float], wins: List[int]) -> float:
    """
    Detect collusion from bid correlation and winner rotation patterns.
    Returns a signal in [0, 1] where higher = more suspicious.
    """
    if len(bids) < 2:
        return 0.0

    # Check bid variance — cartels suppress bid variance
    mean_bid = sum(bids) / len(bids)
    if mean_bid <= 0:
        return 0.0
    variance = sum((b - mean_bid) ** 2 for b in bids) / len(bids)
    # Low variance relative to mean is suspicious
    cv = math.sqrt(variance) / mean_bid if mean_bid > 0 else 0
    variance_signal = max(0.0, 1.0 - cv * 3)

    # Check winner rotation — unnaturally even distribution
    if wins and max(wins) > 0:
        win_entropy = 0.0
        total_wins = sum(wins)
        if total_wins > 0:
            for w in wins:
                if w > 0:
                    p = w / total_wins
                    win_entropy -= p * math.log2(p)
            # High entropy = even rotation = suspicious
            max_entropy = math.log2(len(wins)) if len(wins) > 1 else 1
            rotation_signal = win_entropy / max_entropy if max_entropy > 0 else 0
        else:
            rotation_signal = 0.0
    else:
        rotation_signal = 0.0

    return (variance_signal + rotation_signal) / 2


def compute_composite_reward(
    welfare: float,
    fairness: float,
    participation: float,
    stability: float,
    exploration_bonus: float = 1.0,
    collusion_penalty: float = 1.0,
) -> float:
    """
    R(t) = W(t) × F(t) × P(t) × S(t) × E(t) × anti_collusion

    Multiplicative structure: all terms must be jointly positive.
    You cannot sacrifice one objective to maximize another.

    Returns: float in [0, ~1.2] (exploration can push slightly above 1)
    """
    reward = welfare * fairness * participation * stability
    reward *= exploration_bonus
    reward *= collusion_penalty
    return max(0.0, reward)


# ── Reward Function Registry (for TRL) ──────────────
def reward_welfare(output: str, env_state: dict, **kwargs) -> float:
    """Standalone welfare reward for TRL multi-reward setup."""
    return env_state.get("welfare_ratio", 0.0)


def reward_fairness(output: str, env_state: dict, **kwargs) -> float:
    """Standalone fairness reward for TRL."""
    return 1.0 - env_state.get("gini_coefficient", 0.0)


def reward_participation(output: str, env_state: dict, **kwargs) -> float:
    """Standalone participation reward for TRL."""
    return env_state.get("participation_rate", 1.0)


def reward_stability(output: str, env_state: dict, **kwargs) -> float:
    """Standalone stability reward for TRL."""
    return env_state.get("stability_score", 0.8)


def reward_composite(output: str, env_state: dict, **kwargs) -> float:
    """Composite multiplicative reward for TRL."""
    w = reward_welfare(output, env_state)
    f = reward_fairness(output, env_state)
    p = reward_participation(output, env_state)
    s = reward_stability(output, env_state)
    return w * f * p * s


# Registry for easy import
REWARD_FUNCTIONS = {
    "welfare": reward_welfare,
    "fairness": reward_fairness,
    "participation": reward_participation,
    "stability": reward_stability,
    "composite": reward_composite,
}
