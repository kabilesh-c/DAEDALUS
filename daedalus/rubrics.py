"""
DAEDALUS Rubrics — OpenEnv-compliant reward rubrics.

Uses the real openenv.core.rubrics.base.Rubric API:
  - Subclass Rubric, implement forward(action, observation) -> float
  - Child rubrics auto-register when assigned as attributes
  - Call rubric(action, observation) via __call__ (hooks + last_score)
"""
from typing import Any

from openenv.core.rubrics.base import Rubric


class WelfareRubric(Rubric):
    """Allocative efficiency: fraction of total value captured as surplus."""

    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.metadata.get("welfare_ratio", 0.0))


class FairnessRubric(Rubric):
    """Payment equity: 1 − Gini(surplus). Pre-computed in _make_observation."""

    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.metadata.get("fairness", 0.0))


class ParticipationRubric(Rubric):
    """Market liquidity: fraction of agents still active."""

    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.metadata.get("participation_rate", 0.0))


class StabilityRubric(Rubric):
    """Reward consistency: 1 − 3σ(welfare) over recent rounds."""

    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.metadata.get("stability_score", 0.0))


class DaedalusCompositeRubric(Rubric):
    """
    Multiplicative composite reward: R = W × F × P × S × anti_collusion.

    Child rubrics (welfare, fairness, participation, stability) are
    auto-registered by Rubric.__setattr__ so they appear in
    env.rubric.named_rubrics() for training introspection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.welfare      = WelfareRubric()
        self.fairness     = FairnessRubric()
        self.participation = ParticipationRubric()
        self.stability    = StabilityRubric()

    def forward(self, action: Any, observation: Any) -> float:
        w = self.welfare(action, observation)
        f = self.fairness(action, observation)
        p = self.participation(action, observation)
        s = self.stability(action, observation)

        collusion_signal = float(observation.metadata.get("collusion_signal", 0.0))
        anti_collusion = max(0.5, 1.0 - collusion_signal * 0.5) if collusion_signal > 0.5 else 1.0

        return w * f * p * s * anti_collusion
