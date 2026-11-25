"""Visibility (pass) prediction, Doppler profile and per-pass data volume."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .epochs import format_iso
from .frames import eci_to_ecef, look_angles
from .link import LinkBudget, doppler_shift_hz
from .propagation import SecularJ2Propagator
from .stations import GroundStation


@dataclass(frozen=True)
class Pass:
    satellite: str
    station: str
    aos_s: float
    los_s: float
    tmax_s: float
    max_elevation_deg: float
    aos_azimuth_deg: float
    los_azimuth_deg: float
    truncated: bool = False

    @property
    def duration_s(self) -> float:
        return self.los_s - self.aos_s

    def to_dict(self) -> dict:
        return {
            "satellite": self.satellite,
            "station": self.station,
            "aos_utc": format_iso(self.aos_s),
            "los_utc": format_iso(self.los_s),
            "duration_s": round(self.duration_s, 1),
            "max_elevation_deg": round(self.max_elevation_deg, 2),
            "aos_azimuth_deg": round(self.aos_azimuth_deg, 1),
            "los_azimuth_deg": round(self.los_azimuth_deg, 1),
        }


def look(prop: SecularJ2Propagator, station: GroundStation, t):
    """Azimuth, elevation [rad], range [km] and range rate [km/s] at epochs t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    r, v = eci_to_ecef(prop.states(t), t)
    return look_angles(r, v, station.ecef, station.enu)


def find_passes(
    prop: SecularJ2Propagator,
    station: GroundStation,
    start_s: float,
    end_s: float,
    step_s: float = 20.0,
    satellite: str = "SAT",
) -> list[Pass]:
    """All passes above the station's elevation mask in [start_s, end_s].

    Elevation is sampled on a grid; AOS/LOS are refined to 1 ms with Brent's
    method and the culmination is found by bounded scalar optimisation.
    Passes shorter than ~``step_s`` that start and end between samples can be missed.
    """
    if end_s <= start_s:
        raise ValueError("end_s must be after start_s")
    mask = math.radians(station.min_elevation_deg)
    t = np.linspace(start_s, end_s, int(math.ceil((end_s - start_s) / step_s)) + 1)
    el = look(prop, station, t)[1] - mask

    def f(tt: float) -> float:
        return float(look(prop, station, tt)[1][0] - mask)

    passes: list[Pass] = []
    aos = start_s if el[0] >= 0.0 else None
    for k in range(len(t) - 1):
        if el[k] < 0.0 <= el[k + 1]:
            aos = brentq(f, t[k], t[k + 1], xtol=1e-3)
        elif el[k] >= 0.0 > el[k + 1] and aos is not None:
            los = brentq(f, t[k], t[k + 1], xtol=1e-3)
            passes.append(_build(prop, station, satellite, aos, los, truncated=aos == start_s))
            aos = None
    if aos is not None:
        passes.append(_build(prop, station, satellite, aos, end_s, truncated=True))
    return passes


def _build(prop, station, satellite, aos, los, truncated) -> Pass:
    res = minimize_scalar(
        lambda tt: -float(look(prop, station, tt)[1][0]), bounds=(aos, los), method="bounded", options={"xatol": 1e-2}
    )
    az = look(prop, station, [aos, los])[0]
    return Pass(
        satellite=satellite,
        station=station.name,
        aos_s=float(aos),
        los_s=float(los),
        tmax_s=float(res.x),
        max_elevation_deg=math.degrees(-res.fun),
        aos_azimuth_deg=math.degrees(az[0]),
        los_azimuth_deg=math.degrees(az[1]),
        truncated=bool(truncated),
    )


@dataclass(frozen=True)
class PassProfile:
    t: np.ndarray
    elevation_deg: np.ndarray
    range_km: np.ndarray
    doppler_hz: np.ndarray
    data_rate_bps: np.ndarray

    @property
    def volume_bits(self) -> float:
        r, t = self.data_rate_bps, self.t
        return float(np.sum(0.5 * (r[1:] + r[:-1]) * np.diff(t)))

    @property
    def max_doppler_hz(self) -> float:
        return float(np.max(np.abs(self.doppler_hz)))

    @property
    def peak_rate_bps(self) -> float:
        return float(np.max(self.data_rate_bps))


def pass_profile(
    prop: SecularJ2Propagator, station: GroundStation, p: Pass, link: LinkBudget, dt_s: float = 5.0
) -> PassProfile:
    """Sample elevation, range, Doppler and ACM data rate through a pass."""
    n = max(2, int(math.ceil(p.duration_s / dt_s)) + 1)
    t = np.linspace(p.aos_s, p.los_s, n)
    _, el, rng, rr = look(prop, station, t)
    return PassProfile(
        t=t,
        elevation_deg=np.degrees(el),
        range_km=rng,
        doppler_hz=doppler_shift_hz(rr, link.frequency_hz),
        data_rate_bps=link.data_rate_bps(rng, el, station.g_over_t_db),
    )
