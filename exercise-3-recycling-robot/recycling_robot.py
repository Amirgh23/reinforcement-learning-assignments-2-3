"""Solve the Bellman expectation equations for the recycling robot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotParameters:
    """Transition probabilities, rewards, and discount factor."""

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.9
    search_reward: float = 1.0
    wait_reward: float = 0.0
    rescue_reward: float = -3.0

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")


def bellman_linear_system(params: RobotParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return A and b for the linear system A @ [V(high), V(low)] = b.

    The policy is uniform over the legal actions:
      - high state: search and wait, each with probability 1/2
      - low state: search, wait, and recharge, each with probability 1/3
    """

    params.validate()
    alpha = params.alpha
    beta = params.beta
    gamma = params.gamma

    matrix = np.array(
        [
            [
                1.0 - gamma * (alpha + 1.0) / 2.0,
                -gamma * (1.0 - alpha) / 2.0,
            ],
            [
                -gamma * (2.0 - beta) / 3.0,
                1.0 - gamma * (beta + 1.0) / 3.0,
            ],
        ],
        dtype=float,
    )

    target = np.array(
        [
            (params.search_reward + params.wait_reward) / 2.0,
            (
                beta * params.search_reward
                + (1.0 - beta) * params.rescue_reward
                + params.wait_reward
            )
            / 3.0,
        ],
        dtype=float,
    )
    return matrix, target


def solve_values(params: RobotParameters) -> np.ndarray:
    """Solve for V(high) and V(low)."""

    matrix, target = bellman_linear_system(params)
    if np.linalg.cond(matrix) > 1.0e12:
        raise ValueError("the Bellman linear system is numerically singular")
    return np.linalg.solve(matrix, target)


def bellman_backup(params: RobotParameters, values: np.ndarray) -> np.ndarray:
    """Evaluate the right-hand side of both Bellman equations."""

    high, low = map(float, values)
    search_high = params.search_reward + params.gamma * (
        params.alpha * high + (1.0 - params.alpha) * low
    )
    wait_high = params.wait_reward + params.gamma * high

    search_low = (
        params.beta * (params.search_reward + params.gamma * low)
        + (1.0 - params.beta)
        * (params.rescue_reward + params.gamma * high)
    )
    wait_low = params.wait_reward + params.gamma * low
    recharge_low = params.gamma * high

    return np.array(
        [
            0.5 * search_high + 0.5 * wait_high,
            (search_low + wait_low + recharge_low) / 3.0,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--search-reward", type=float, default=1.0)
    parser.add_argument("--wait-reward", type=float, default=0.0)
    parser.add_argument("--rescue-reward", type=float, default=-3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = RobotParameters(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        search_reward=args.search_reward,
        wait_reward=args.wait_reward,
        rescue_reward=args.rescue_reward,
    )
    matrix, target = bellman_linear_system(params)
    values = solve_values(params)
    backed_up_values = bellman_backup(params, values)

    print("Bellman linear system A @ V = b")
    print("A =")
    print(matrix)
    print("b =")
    print(target)
    print(f"V(high) = {values[0]:.6f}")
    print(f"V(low)  = {values[1]:.6f}")
    print(f"maximum Bellman residual = {np.max(np.abs(values - backed_up_values)):.3e}")


if __name__ == "__main__":
    main()

