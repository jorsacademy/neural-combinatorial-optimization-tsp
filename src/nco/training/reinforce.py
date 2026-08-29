"""Training utilities for policy-gradient NCO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nco.problems.tsp import tour_length


@dataclass
class MovingAverageBaseline:
    """Exponential moving-average baseline for REINFORCE variance reduction."""

    beta: float = 0.9
    value: float | None = None

    def update(self, costs: Tensor) -> Tensor:
        mean_cost = float(costs.detach().mean())
        self.value = mean_cost if self.value is None else self.beta * self.value + (1 - self.beta) * mean_cost
        return torch.as_tensor(self.value, device=costs.device, dtype=costs.dtype)


def reinforce_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    coords: Tensor,
    baseline: MovingAverageBaseline,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one REINFORCE update and return scalar training diagnostics."""
    model.train()
    tour, log_prob = model(coords, decode_type="sampling")
    costs = tour_length(coords, tour)
    baseline_value = baseline.update(costs)
    advantage = costs.detach() - baseline_value
    loss = (advantage * log_prob).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "mean_cost": float(costs.detach().mean()),
        "baseline": float(baseline_value.detach()),
        "grad_norm": float(grad_norm),
    }
