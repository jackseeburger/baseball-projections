"""Remove the bookmaker's margin from a set of quoted prices.

A book quotes each outcome so the implied probabilities sum to more than
one; the excess is its margin ("vig"). To compare a book to an exchange or
to a model we need fair probabilities that sum to one.

Two standard methods:
    multiplicative  scale every implied probability by the same factor
    power           raise each to the power k that makes them sum to one;
                    removes proportionally more from longshots, which is
                    closer to how books actually shade (favourite-longshot bias)
"""
from __future__ import annotations

from math import log


def implied(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must exceed 1, got {decimal_odds}")
    return 1.0 / decimal_odds


def overround(decimal_odds: list[float]) -> float:
    """Sum of implied probabilities; 1.0 is a fair book, 1.04 is a 4% margin."""
    return sum(implied(o) for o in decimal_odds)


def multiplicative(decimal_odds: list[float]) -> list[float]:
    imp = [implied(o) for o in decimal_odds]
    total = sum(imp)
    return [p / total for p in imp]


def power(decimal_odds: list[float], tol: float = 1e-10, max_iter: int = 100) -> list[float]:
    """Find k with Σ p_i^k = 1 by bisection; k=1 means no margin."""
    imp = [implied(o) for o in decimal_odds]
    if abs(sum(imp) - 1.0) < tol:
        return imp
    lo, hi = 1.0, 1.0
    while sum(p ** hi for p in imp) > 1.0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("power de-vig did not converge")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        s = sum(p ** mid for p in imp)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    out = [p ** k for p in imp]
    total = sum(out)
    return [p / total for p in out]


METHODS = {"multiplicative": multiplicative, "power": power}
