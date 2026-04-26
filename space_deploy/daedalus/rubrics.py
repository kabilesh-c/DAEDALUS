"""
DAEDALUS Rubrics — OpenEnv-compliant reward rubrics.
Standardizes the multiplicative composite reward for mechanism design.
"""
from typing import Any, Dict, Optional
from openenv.core.env_server.rubric import Rubric, RubricResult

from .rewards import (
    compute_welfare_ratio,
    compute_fairness,
    compute_participation_rate,
    compute_stability,
    compute_collusion_signal,
)

class WelfareRubric(Rubric):
    """Measures allocative efficiency."""
    def __init__(self):
        super().__init__(name="welfare", weight=1.0)

    def evaluate(self, action: Any, observation: Any) -> RubricResult:
        # Note: In our environment, welfare is computed inside the step logic
        # and cached in the observation metadata.
        val = observation.metadata.get("welfare_ratio", 0.0)
        return RubricResult(
            score=float(val),
            reason=f"Social welfare ratio: {val:.3f}",
            metadata={"raw_value": val}
        )

class FairnessRubric(Rubric):
    """Measures payment equity (1 - Gini)."""
    def __init__(self):
        super().__init__(name="fairness", weight=1.0)

    def evaluate(self, action: Any, observation: Any) -> RubricResult:
        val = observation.metadata.get("fairness", 0.0)
        return RubricResult(
            score=float(val),
            reason=f"Payment fairness score: {val:.3f}",
            metadata={"raw_value": val}
        )

class ParticipationRubric(Rubric):
    """Measures market liquidity."""
    def __init__(self):
        super().__init__(name="participation", weight=1.0)

    def evaluate(self, action: Any, observation: Any) -> RubricResult:
        val = observation.metadata.get("participation_rate", 0.0)
        return RubricResult(
            score=float(val),
            reason=f"Market participation rate: {val:.3f}",
            metadata={"raw_value": val}
        )

class StabilityRubric(Rubric):
    """Measures consistency over time."""
    def __init__(self):
        super().__init__(name="stability", weight=1.0)

    def evaluate(self, action: Any, observation: Any) -> RubricResult:
        val = observation.metadata.get("stability_score", 0.0)
        return RubricResult(
            score=float(val),
            reason=f"Economic stability score: {val:.3f}",
            metadata={"raw_value": val}
        )

class DaedalusCompositeRubric(Rubric):
    """
    Multiplicative Composite Reward: R = W * F * P * S * CollusionPenalty.
    This is the core DAEDALUS reward logic wrapped for OpenEnv.
    """
    def __init__(self):
        super().__init__(name="daedalus_composite", weight=1.0)
        self.welfare = WelfareRubric()
        self.fairness = FairnessRubric()
        self.participation = ParticipationRubric()
        self.stability = StabilityRubric()

    def evaluate(self, action: Any, observation: Any) -> RubricResult:
        w = self.welfare.evaluate(action, observation).score
        f = self.fairness.evaluate(action, observation).score
        p = self.participation.evaluate(action, observation).score
        s = self.stability.evaluate(action, observation).score
        
        # Pull collusion multiplier from info
        collusion_signal = observation.metadata.get("collusion_signal", 0.0)
        collusion_mult = max(0.5, 1.0 - collusion_signal * 0.5) if collusion_signal > 0.5 else 1.0
        
        composite = w * f * p * s * collusion_mult
        
        return RubricResult(
            score=float(composite),
            reason=f"Multiplicative Reward: {composite:.4f}",
            metadata={
                "welfare": w,
                "fairness": f,
                "participation": p,
                "stability": s,
                "collusion_penalty": collusion_mult
            }
        )
