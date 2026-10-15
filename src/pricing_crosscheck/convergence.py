"""Observed convergence orders, as defined in docs/protocol.md section 10."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pricing_crosscheck import spec

_CONV = spec.load()["convergence"]
FLOOR_REL = _CONV["floor_rel"]  # errors below FLOOR_REL * max(1, |ref|) are at rounding level
FIT_LEVELS = _CONV["fit_last_levels"]
MIN_LEVELS = _CONV["min_levels"]


@dataclass
class OrderFit:
    order: float  # nan if not measurable
    levels_used: list[float]  # resolutions entering the fit
    local_orders: list[float]

    @property
    def measurable(self) -> bool:
        return not math.isnan(self.order)


def local_orders(resolutions: Sequence[float], errors: Sequence[float]) -> list[float]:
    out = []
    for (n0, e0), (n1, e1) in zip(zip(resolutions, errors), zip(resolutions[1:], errors[1:])):
        if e0 == 0.0 or e1 == 0.0:
            out.append(float("nan"))
        else:
            out.append(math.log(abs(e0) / abs(e1)) / math.log(n1 / n0))
    return out


def fitted_order(resolutions: Sequence[float], errors: Sequence[float], reference: float = 1.0) -> OrderFit:
    """Least-squares slope of -log|error| on log(resolution).

    Uses the last FIT_LEVELS levels whose |error| is above the rounding floor
    FLOOR_REL * max(1, |reference|); needs at least MIN_LEVELS of them.
    """
    floor = FLOOR_REL * max(1.0, abs(reference))
    pairs = [(n, e) for n, e in zip(resolutions, errors) if abs(e) > floor]
    pairs = pairs[-FIT_LEVELS:]
    loc = local_orders(list(resolutions), list(errors))
    used = [float(p[0]) for p in pairs]
    if len(pairs) < MIN_LEVELS:
        return OrderFit(float("nan"), used, loc)
    n = np.log(np.array([p[0] for p in pairs], dtype=float))
    e = np.log(np.abs(np.array([p[1] for p in pairs], dtype=float)))
    slope = np.polyfit(n, e, 1)[0]
    return OrderFit(float(-slope), used, loc)
