"""Ground stations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import enu_basis, geodetic_to_ecef


@dataclass(frozen=True)
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 5.0
    g_over_t_db: float = 25.0  # receive figure of merit [dB/K]
    ecef: np.ndarray = field(init=False, repr=False, compare=False)
    enu: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("latitude must be within [-90, 90] degrees")
        object.__setattr__(self, "ecef", geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_km))
        object.__setattr__(self, "enu", enu_basis(self.lat_deg, self.lon_deg))


def demo_network() -> list[GroundStation]:
    """A five-site X-band network at approximate, publicly known ground-station locations.

    Coordinates are rounded and the antenna figures are generic assumptions;
    the network is illustrative and does not represent any operator.
    """
    return [
        GroundStation("Svalbard", 78.23, 15.41, 0.50, 5.0, 28.0),
        GroundStation("Fairbanks", 64.86, -147.85, 0.30, 5.0, 26.0),
        GroundStation("Bengaluru", 13.03, 77.51, 0.89, 10.0, 24.0),
        GroundStation("Hartebeesthoek", -25.89, 27.69, 1.40, 10.0, 25.0),
        GroundStation("Punta Arenas", -52.94, -70.85, 0.05, 5.0, 24.0),
    ]
