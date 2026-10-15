"""Load ``specs/phase1.json`` and expand its parameter grids.

The JSON file is the single source of the numbers fixed by docs/protocol.md.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pricing_crosscheck.env import REPO_ROOT

SPEC_PATH = REPO_ROOT / "specs" / "phase1.json"
OPTION_TYPES = ("call", "put")
BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


@lru_cache(maxsize=None)
def load(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class VanillaPoint:
    option: str  # "call" or "put"
    strike: float
    days: int
    vol: float
    rate: float
    div: float
    spot: float = 100.0

    @property
    def T(self) -> float:
        return self.days / 365.0

    @property
    def point_id(self) -> str:
        return (
            f"{self.option}|K={self.strike:g}|D={self.days}|v={self.vol:g}"
            f"|r={self.rate:g}|q={self.div:g}"
        )


@dataclass(frozen=True)
class BarrierPoint:
    barrier_type: str  # one of BARRIER_TYPES
    option: str
    barrier: float
    strike: float
    days: int
    vol: float
    rate: float
    div: float
    spot: float = 100.0

    @property
    def T(self) -> float:
        return self.days / 365.0

    @property
    def is_down(self) -> bool:
        return self.barrier_type.startswith("down")

    @property
    def is_knock_in(self) -> bool:
        return self.barrier_type.endswith("-in")

    @property
    def point_id(self) -> str:
        return (
            f"{self.barrier_type}|{self.option}|H={self.barrier:g}|K={self.strike:g}"
            f"|D={self.days}|v={self.vol:g}|r={self.rate:g}|q={self.div:g}"
        )

    def vanilla(self) -> VanillaPoint:
        return VanillaPoint(self.option, self.strike, self.days, self.vol, self.rate, self.div, self.spot)


def grid_axes(name: str, profile: str, spec: dict[str, Any] | None = None) -> dict[str, list]:
    spec = spec or load()
    if profile not in spec["profiles"]:
        raise ValueError(f"unknown profile {profile!r}")
    override = spec["profiles"][profile]
    return override.get(name, spec["grids"][name])


def vanilla_points(profile: str, grid: str = "vanilla", spec: dict[str, Any] | None = None) -> list[VanillaPoint]:
    spec = spec or load()
    axes = grid_axes(grid, profile, spec)
    spot = spec["market"]["spot"]
    return [
        VanillaPoint(opt, float(k), int(d), float(v), float(r), float(q), spot)
        for opt, k, d, v, r, q in itertools.product(
            OPTION_TYPES, axes["strike"], axes["days"], axes["vol"], axes["rate"], axes["div"]
        )
    ]


def barrier_points(profile: str, grid: str = "barrier", spec: dict[str, Any] | None = None) -> list[BarrierPoint]:
    spec = spec or load()
    axes = grid_axes(grid, profile, spec)
    spot = spec["market"]["spot"]
    points = []
    for btype in BARRIER_TYPES:
        levels = axes["barrier_down"] if btype.startswith("down") else axes["barrier_up"]
        for opt, h, k, d, v, (r, q) in itertools.product(
            OPTION_TYPES, levels, axes["strike"], axes["days"], axes["vol"], axes["rate_div"]
        ):
            points.append(BarrierPoint(btype, opt, float(h), float(k), int(d), float(v), float(r), float(q), spot))
    return points


def seed_for(instrument: str, point_id: str, engine: str, run: str = "", spec: dict[str, Any] | None = None) -> int:
    """Deterministic seed in [1, 2147483646], independent of execution order.

    ``run`` distinguishes sample counts, monitoring frequencies and refinement
    levels so that different runs use independent random numbers.
    """
    spec = spec or load()
    base = spec["seeds"]["base"]
    key = f"{base}|{instrument}|{point_id}|{engine}|{run}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 2147483646


def run_tag(samples: int | None = None, steps_per_year: int | None = None, refine: int | None = None) -> str:
    """Canonical ``run`` string for ``seed_for`` (protocol section 10)."""
    parts = []
    if samples is not None:
        parts.append(f"n={samples}")
    if steps_per_year is not None:
        parts.append(f"steps_per_year={steps_per_year}")
    if refine is not None:
        parts.append(f"refine={refine}")
    return ";".join(parts)


def tolerance(kind: str, reference: float, instrument: str = "european", spec: dict[str, Any] | None = None) -> float:
    """Absolute price tolerance of class ``exact`` or ``numerical`` at a reference value."""
    spec = spec or load()
    tol = spec["tolerances"]
    if kind == "exact":
        return max(tol["exact"]["abs"], tol["exact"]["rel"] * abs(reference))
    if kind == "numerical":
        floor = tol["numerical"]["abs_per_unit_notional"] * tol["notional"][instrument]
        return max(floor, tol["numerical"]["rel"] * abs(reference))
    raise ValueError(f"unknown tolerance class {kind!r}")


def greek_tolerance(greek: str, reference: float, instrument: str, spec: dict[str, Any] | None = None) -> float:
    """Tolerance for a Greek computed by a numerical method or a bump."""
    spec = spec or load()
    tol = spec["tolerances"]
    floor = tol["greeks"]["abs_per_unit_notional"][greek] * tol["notional"][instrument]
    return max(floor, tol["greeks"]["rel"] * abs(reference))
