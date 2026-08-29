from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nco.evaluation import evaluate_batch
from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NCO TSP policy against heuristics.")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionTSPPolicy().to(device)

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state_dict"])

    coords = generate_euclidean_tsp(args.instances, args.nodes, device=device)
    metrics = evaluate_batch(model, coords)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
