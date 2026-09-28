"""Traceable parameter uncertainty helpers for channel/device studies."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class UncertainParameter:
    name: str
    nominal: float
    relative_sigma: float
    distribution: str = "lognormal"
    source: str = "unspecified"
    status: str = "assumption"

    def sample(self, rng: random.Random) -> float:
        if self.nominal < 0 or self.relative_sigma < 0:
            raise ValueError("nominal and relative_sigma must be non-negative")
        if self.distribution == "lognormal":
            if self.nominal == 0:
                return 0.0
            sigma = math.sqrt(math.log1p(self.relative_sigma ** 2))
            mu = math.log(self.nominal) - 0.5 * sigma ** 2
            return rng.lognormvariate(mu, sigma)
        if self.distribution == "normal":
            return max(0.0, rng.normalvariate(self.nominal,
                                              self.nominal * self.relative_sigma))
        raise ValueError(f"unsupported distribution {self.distribution!r}")

    def bounds(self, z: float = 2.0) -> tuple[float, float]:
        if z < 0:
            raise ValueError("z must be non-negative")
        spread = self.nominal * self.relative_sigma * z
        return max(0.0, self.nominal - spread), self.nominal + spread
