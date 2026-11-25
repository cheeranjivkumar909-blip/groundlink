"""Analytical orbit propagation: Kepler + J2 secular drift (vectorised).

Accurate to the level needed for contact planning over a few days: pass
timing errors are a few seconds when initialised from fresh elements.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH
from .elements import KeplerianElements, kepler_to_cartesian, mean_to_true
from .tle import TLE


def j2_secular_rates(el: KeplerianElements, mu: float = MU_EARTH) -> tuple[float, float, float]:
    """Secular rates (dRAAN/dt, dargp/dt, dM/dt) [rad/s] caused by J2."""
    n = math.sqrt(mu / el.a**3)
    p = el.a * (1.0 - el.e**2)
    k = 1.5 * J2 * (R_EARTH / p) ** 2 * n
    ci = math.cos(el.i)
    return -k * ci, 0.5 * k * (5.0 * ci**2 - 1.0), n + 0.5 * k * math.sqrt(1.0 - el.e**2) * (3.0 * ci**2 - 1.0)


class SecularJ2Propagator:
    """Keplerian motion with J2 secular node, perigee and mean-anomaly drift."""

    def __init__(self, elements: KeplerianElements, epoch_s: float, mu: float = MU_EARTH):
        self.elements = elements
        self.epoch_s = float(epoch_s)
        self.mu = mu
        self.raan_dot, self.argp_dot, self.mean_dot = j2_secular_rates(elements, mu)
        self._m0 = elements.mean_anomaly

    @classmethod
    def from_tle(cls, tle: TLE) -> "SecularJ2Propagator":
        return cls(tle.to_elements(), tle.epoch_seconds)

    def states(self, t) -> np.ndarray:
        """ECI states, shape (N, 6): position [km] and velocity [km/s]."""
        dt = np.atleast_1d(np.asarray(t, dtype=float)) - self.epoch_s
        el = self.elements
        nu = mean_to_true(self._m0 + self.mean_dot * dt, el.e)
        r, v = kepler_to_cartesian(
            el.a, el.e, el.i, el.raan + self.raan_dot * dt, el.argp + self.argp_dot * dt, nu, self.mu
        )
        return np.hstack([r, v])
