"""Vectorized simulation of the stationary 10-armed bandit testbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EPSILONS = (0.0, 0.01, 0.1)


def simulate_bandits(
    *,
    k: int = 10,
    runs: int = 2_000,
    steps: int = 1_000,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    seed: int = 42,
) -> dict[float, dict[str, np.ndarray]]:
    """Simulate epsilon-greedy agents on a shared set of bandit problems.

    Each row represents one independent stationary k-armed bandit. All epsilon
    values are evaluated on the same true action values for a fair comparison.
    Ties between greedy actions are broken uniformly at random.
    """

    if k < 2 or runs < 1 or steps < 1:
        raise ValueError("k must be at least 2, and runs/steps must be positive")
    if any(not 0.0 <= epsilon <= 1.0 for epsilon in epsilons):
        raise ValueError("each epsilon must be between 0 and 1")

    rng = np.random.default_rng(seed)
    q_true = rng.normal(0.0, 1.0, size=(runs, k))
    optimal_actions = np.argmax(q_true, axis=1)
    run_indices = np.arange(runs)
    results: dict[float, dict[str, np.ndarray]] = {}

    for epsilon in epsilons:
        estimates = np.zeros((runs, k), dtype=float)
        counts = np.zeros((runs, k), dtype=np.int32)
        average_rewards = np.empty(steps, dtype=float)
        optimal_action_percent = np.empty(steps, dtype=float)

        for step in range(steps):
            # Generate random scores only for actions tied at the current maximum.
            best_mask = estimates == estimates.max(axis=1, keepdims=True)
            tie_scores = rng.random((runs, k))
            tie_scores[~best_mask] = -1.0
            greedy_actions = np.argmax(tie_scores, axis=1)

            explore = rng.random(runs) < epsilon
            exploratory_actions = rng.integers(0, k, size=runs)
            actions = np.where(explore, exploratory_actions, greedy_actions)

            rewards = rng.normal(q_true[run_indices, actions], 1.0)
            average_rewards[step] = rewards.mean()
            optimal_action_percent[step] = (
                100.0 * np.mean(actions == optimal_actions)
            )

            counts[run_indices, actions] += 1
            old_estimates = estimates[run_indices, actions]
            action_counts = counts[run_indices, actions]
            estimates[run_indices, actions] = (
                old_estimates + (rewards - old_estimates) / action_counts
            )

        results[epsilon] = {
            "average_rewards": average_rewards,
            "optimal_action_percent": optimal_action_percent,
            "final_estimates": estimates,
        }

    return results


def plot_results(
    results: dict[float, dict[str, np.ndarray]], output: Path
) -> None:
    """Plot average reward and optimal-action percentage."""

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for epsilon, values in results.items():
        label = f"epsilon = {epsilon:g}" + (" (greedy)" if epsilon == 0 else "")
        axes[0].plot(values["average_rewards"], label=label)
        axes[1].plot(values["optimal_action_percent"], label=label)

    axes[0].set_title("Average Reward on the 10-Armed Testbed")
    axes[0].set_ylabel("Average reward")
    axes[1].set_title("Percentage of Optimal Action")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action (%)")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def print_summary(results: dict[float, dict[str, np.ndarray]]) -> None:
    """Print mean performance over the final 100 steps."""

    print("Final-100-step summary")
    for epsilon, values in results.items():
        reward = values["average_rewards"][-100:].mean()
        optimal = values["optimal_action_percent"][-100:].mean()
        print(
            f"epsilon={epsilon:g}: average reward={reward:.4f}, "
            f"optimal action={optimal:.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--runs", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("bandit_results.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = simulate_bandits(
        k=args.arms,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
    )
    print_summary(results)
    plot_results(results, args.output)
    print(f"Plot saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()

