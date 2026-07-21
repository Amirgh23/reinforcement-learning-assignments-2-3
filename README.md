# Reinforcement Learning Assignments 2 and 3

Python implementations for two reinforcement learning exercises:

1. **Exercise 2 — 10-Armed Bandit:** compares greedy and epsilon-greedy action selection on the standard stationary testbed.
2. **Exercise 3 — Recycling Robot:** constructs and solves the Bellman expectation equations for a uniform random policy.

## Repository structure

```text
.
├── exercise-2-bandit/
│   └── bandit.py
├── exercise-3-recycling-robot/
│   └── recycling_robot.py
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Exercise 2: 10-Armed Bandit

The true action values are sampled independently from a standard normal distribution:

```text
q*(a) ~ N(0, 1)
```

After selecting action `a`, the observed reward is sampled from:

```text
R_t ~ N(q*(a), 1)
```

The program compares `epsilon = 0`, `0.01`, and `0.1` over 2,000 independent runs with 1,000 steps per run. Action-value estimates are updated using the incremental sample average.

Run it with:

```bash
python exercise-2-bandit/bandit.py
```

The script prints a numerical summary and saves `bandit_results.png` in the current directory. Parameters can be changed from the command line:

```bash
python exercise-2-bandit/bandit.py --runs 2000 --steps 1000 --seed 42 --output bandit_results.png
```

## Exercise 3: Recycling Robot

The recycling robot has two battery states:

- `high`: available actions are `search` and `wait`.
- `low`: available actions are `search`, `wait`, and `recharge`.

The policy selects uniformly among the actions available in each state. The program builds the resulting two-by-two Bellman linear system and solves for `V(high)` and `V(low)`.

Run it with:

```bash
python exercise-3-recycling-robot/recycling_robot.py
```

All model parameters can be overridden. For example:

```bash
python exercise-3-recycling-robot/recycling_robot.py --alpha 0.5 --beta 0.5 --gamma 0.9 --search-reward 1 --wait-reward 0 --rescue-reward -3
```

## Notes

- Both implementations use fixed random seeds or deterministic linear algebra, so their outputs are reproducible.
- The bandit simulation is vectorized across independent runs for fast execution.
- The recycling-robot script verifies the computed solution by substituting it back into the Bellman equations.

## Author

Amirreza Ghaffarian — M.Sc. student in Artificial Intelligence and Robotics.
