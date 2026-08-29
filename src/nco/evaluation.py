"""Evaluation helpers for learned, heuristic, and exact TSP solvers."""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn

from nco.exact import held_karp
from nco.heuristics import nearest_neighbor, two_opt
from nco.problems.tsp import is_valid_tour, tour_length


def optimality_gap(cost: Tensor, optimum: Tensor) -> Tensor:
    """Return percentage optimality gap; both tensors must contain positive costs."""
    if torch.any(optimum <= 0):
        raise ValueError("optimum values must be positive")
    return 100.0 * (cost - optimum) / optimum


@torch.inference_mode()
def evaluate_batch(
    model: nn.Module,
    coords: Tensor,
    *,
    exact: bool = False,
    max_exact_nodes: int = 12,
) -> dict[str, float]:
    """Evaluate greedy neural decoding against heuristic and optional exact baselines."""
    model.eval()

    started = time.perf_counter()
    neural_tour, _ = model(coords, decode_type="greedy")
    neural_seconds = time.perf_counter() - started
    neural_cost = tour_length(coords, neural_tour)

    started = time.perf_counter()
    nn_tour = nearest_neighbor(coords)
    nn_seconds = time.perf_counter() - started
    nn_cost = tour_length(coords, nn_tour)

    started = time.perf_counter()
    opt_tour = two_opt(coords, nn_tour)
    two_opt_seconds = time.perf_counter() - started
    opt_cost = tour_length(coords, opt_tour)

    metrics = {
        "neural_mean_cost": float(neural_cost.mean()),
        "nearest_neighbor_mean_cost": float(nn_cost.mean()),
        "two_opt_mean_cost": float(opt_cost.mean()),
        "neural_feasibility_rate": float(is_valid_tour(neural_tour).float().mean()),
        "neural_seconds": neural_seconds,
        "nearest_neighbor_seconds": nn_seconds,
        "two_opt_seconds": two_opt_seconds,
    }

    if exact:
        started = time.perf_counter()
        exact_tour, exact_cost = held_karp(coords, max_nodes=max_exact_nodes)
        exact_seconds = time.perf_counter() - started
        if not bool(is_valid_tour(exact_tour).all()):
            raise RuntimeError("exact solver returned an invalid tour")

        metrics.update(
            {
                "exact_mean_cost": float(exact_cost.mean()),
                "exact_seconds": exact_seconds,
                "neural_mean_gap_pct": float(optimality_gap(neural_cost, exact_cost).mean()),
                "nearest_neighbor_mean_gap_pct": float(
                    optimality_gap(nn_cost, exact_cost).mean()
                ),
                "two_opt_mean_gap_pct": float(optimality_gap(opt_cost, exact_cost).mean()),
            }
        )

    return metrics
