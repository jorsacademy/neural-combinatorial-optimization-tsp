# Neural Combinatorial Optimization for TSP

A compact research-oriented implementation of **Neural Combinatorial Optimization (NCO)** for the Euclidean Traveling Salesman Problem (TSP), combining attention-based neural policies, reinforcement learning, classical heuristics, exact dynamic programming, and OR-style evaluation.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Why this repository exists

Neural Combinatorial Optimization studies how learned policies can construct or improve solutions to combinatorial optimization problems. This repository focuses on the TSP as a controlled testbed and deliberately compares learned solutions against classical and exact baselines instead of reporting neural results in isolation.

The implementation is inspired by the research line around:

- Vinyals, Fortunato, and Jaitly (2015), *Pointer Networks*.
- Bello et al. (2016), *Neural Combinatorial Optimization with Reinforcement Learning*.
- Dai et al. (2017), *Learning Combinatorial Optimization Algorithms over Graphs*.
- Kool, van Hoof, and Welling (2019), *Attention, Learn to Solve Routing Problems!*

This is an educational/research implementation, not a reproduction claiming paper-level benchmark parity.

## Features

- Uniform Euclidean TSP instance generation
- Clustered TSP instance generation for distribution-shift experiments
- Tour-length and feasibility utilities
- Nearest-neighbor baseline
- 2-opt local search baseline
- Exact Held-Karp dynamic programming for small instances
- Transformer-style encoder with attention decoder
- Greedy and sampling decoding
- REINFORCE training with a moving-average baseline
- Optimality-gap evaluation against exact solutions
- Size generalization benchmark across multiple node counts
- Distribution generalization benchmark: uniform vs clustered instances
- Unit tests for feasibility, decoding, geometry, heuristics, exact optimization, and shifted data generation
- GitHub Actions CI with pytest and Ruff

## Project structure

```text
.
├── src/nco/
│   ├── problems/tsp.py
│   ├── models/policy.py
│   ├── training/reinforce.py
│   ├── heuristics.py
│   ├── exact.py
│   └── evaluation.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark_generalization.py
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

For small instances, add an exact Held-Karp benchmark and report optimality gaps:

```bash
python scripts/evaluate.py --nodes 10 --instances 32 --exact
```

Held-Karp has exponential complexity, so exact evaluation is capped at 12 nodes by default. The cap can be changed explicitly with `--max-exact-nodes`, but increasing it can become expensive quickly.

## Generalization benchmark

A learned combinatorial optimizer may perform well on the same problem distribution used during training while degrading under size or distribution shift. This repository therefore includes an explicit generalization benchmark.

Train a policy, for example on TSP20:

```bash
python scripts/train.py --nodes 20 --steps 1000 --checkpoint checkpoints/tsp20.pt
```

Then evaluate the same checkpoint on TSP20, TSP30, and TSP50 under both uniform and clustered spatial distributions:

```bash
python scripts/benchmark_generalization.py \
  --checkpoint checkpoints/tsp20.pt \
  --train-nodes 20 \
  --test-nodes 20 30 50 \
  --instances 128
```

The script prints CSV-compatible rows with:

- distribution,
- test node count,
- test/train size ratio,
- neural mean tour cost,
- nearest-neighbor mean tour cost,
- 2-opt mean tour cost,
- neural feasibility rate.

Two shifts are intentionally separated:

1. **Size shift:** the model is trained on one graph size and evaluated on larger graph sizes.
2. **Distribution shift:** the model is trained on uniform random coordinates and evaluated on clustered coordinates.

The clustered generator samples random cluster centers and places nodes around those centers with Gaussian noise, clipped to the unit square. Use `--clusters` and `--cluster-std` to control the shift severity.

The raw tour cost grows with instance size, so size-shift results should not be interpreted only through absolute cost. Compare the neural policy against the same-instance heuristic baselines and, for sufficiently small instances, against the exact Held-Karp optimum.

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

## Exact benchmark

For small instances the repository uses Held-Karp dynamic programming, which solves TSP exactly in \(O(n^2 2^n)\) time. Exact solutions provide a proper OR benchmark rather than relying only on relative comparisons between heuristics.

The evaluator can report:

- neural mean tour cost,
- nearest-neighbor mean tour cost,
- 2-opt mean tour cost,
- exact mean tour cost,
- feasibility rate,
- runtime for each method,
- neural optimality gap,
- nearest-neighbor optimality gap,
- 2-opt optimality gap.

For an obtained cost \(z\) and exact optimum \(z^*\), the reported gap is

\[
\mathrm{gap}(\%) = 100\,\frac{z-z^*}{z^*}.
\]

This makes it possible to ask the optimization question that matters: how much solution quality is being traded for learned inference speed?

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

## Research roadmap

Planned extensions include rollout baselines, multi-start decoding, normalized regret/generalization summaries, CVRP support, learned improvement operators, and benchmark datasets beyond synthetic Euclidean instances.

## References

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS.
2. Bello, I. et al. (2016). Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940.
3. Dai, H. et al. (2017). Learning Combinatorial Optimization Algorithms over Graphs. NeurIPS.
4. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR.
5. Held, M., & Karp, R. M. (1962). A Dynamic Programming Approach to Sequencing Problems. Journal of the Society for Industrial and Applied Mathematics.

## License

This repository is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not permitted. Review `LICENSE` before copying, modifying, distributing, or deploying the software.
