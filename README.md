# Neural Combinatorial Optimization for TSP

A compact research-oriented implementation of **Neural Combinatorial Optimization (NCO)** for the Euclidean Traveling Salesman Problem (TSP), combining attention-based neural policies, reinforcement learning, classical heuristics, and OR-style evaluation.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Why this repository exists

Neural Combinatorial Optimization studies how learned policies can construct or improve solutions to combinatorial optimization problems. This repository focuses on the TSP as a controlled testbed and deliberately compares learned solutions against classical baselines instead of reporting neural results in isolation.

The implementation is inspired by the research line around:

- Vinyals, Fortunato, and Jaitly (2015), *Pointer Networks*.
- Bello et al. (2016), *Neural Combinatorial Optimization with Reinforcement Learning*.
- Dai et al. (2017), *Learning Combinatorial Optimization Algorithms over Graphs*.
- Kool, van Hoof, and Welling (2019), *Attention, Learn to Solve Routing Problems!*

This is an educational/research implementation, not a reproduction claiming paper-level benchmark parity.

## Features

- Euclidean TSP instance generation
- Tour-length and feasibility utilities
- Nearest-neighbor baseline
- 2-opt local search baseline
- Transformer-style encoder with attention decoder
- Greedy and sampling decoding
- REINFORCE training with a moving-average baseline
- Evaluation with heuristic comparisons and optimality-gap support when an exact optimum is available
- Unit tests for feasibility, decoding, geometry, and heuristics
- GitHub Actions CI with pytest and Ruff

## Project structure

```text
.
├── src/nco/
│   ├── problems/tsp.py
│   ├── models/policy.py
│   ├── training/reinforce.py
│   ├── heuristics.py
│   └── evaluation.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Quick start

Train on randomly generated Euclidean TSP instances:

```bash
python scripts/train.py --nodes 20 --steps 1000 --batch-size 128
```

Evaluate an untrained or saved model against heuristics:

```bash
python scripts/evaluate.py --nodes 20 --instances 128
python scripts/evaluate.py --nodes 20 --instances 128 --checkpoint checkpoints/nco_tsp.pt
```

## What is optimized?

For coordinates \(x_i \in \mathbb{R}^2\), a tour \(\pi\) has cost

\[
L(\pi)=\sum_{t=1}^{n-1}\lVert x_{\pi_t}-x_{\pi_{t+1}}\rVert_2
+\lVert x_{\pi_n}-x_{\pi_1}\rVert_2.
\]

The neural policy defines a distribution over permutations. REINFORCE minimizes expected tour length using

\[
\nabla_\theta J(\theta) \approx (L(\pi)-b)\nabla_\theta \log p_\theta(\pi\mid x),
\]

where \(b\) is a moving-average baseline.

## OR-style evaluation

A learned optimizer should be assessed on more than inference speed. This project reports:

- mean tour length,
- feasibility rate,
- comparison with nearest neighbor,
- comparison with nearest neighbor + 2-opt,
- inference time,
- and, where exact solutions are available, optimality gap:

\[
\mathrm{gap}(\%) = 100\,\frac{z-z^*}{z^*}.
\]

The purpose is to treat NCO as an optimization method, not merely as a neural-network demonstration.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

## Research roadmap

Planned extensions include rollout baselines, multi-start decoding, exact Held-Karp comparisons for small instances, size/distribution generalization experiments, CVRP support, and learned improvement operators.

## References

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS.
2. Bello, I. et al. (2016). Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940.
3. Dai, H. et al. (2017). Learning Combinatorial Optimization Algorithms over Graphs. NeurIPS.
4. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR.

## License

This repository is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not permitted. Review `LICENSE` before copying, modifying, distributing, or deploying the software.
