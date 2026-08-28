#!/usr/bin/env python3
"""Compute normalized exponential progress diff scale."""

import argparse
import math


def compute_scale(horizon: int, beta: float) -> float:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if abs(beta) < 1e-12:
        return (horizon + 1) / 2.0
    denom = math.exp(beta) - 1.0
    return sum(
        (math.exp(beta * t / horizon) - 1.0) / denom
        for t in range(1, horizon + 1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--beta", type=float, default=2.0)
    args = parser.parse_args()
    print(f"{compute_scale(args.horizon, args.beta):.12g}")


if __name__ == "__main__":
    main()
