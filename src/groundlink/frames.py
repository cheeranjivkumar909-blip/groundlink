"""Earth rotation, geodetic coordinates and topocentric look angles."""

from __future__ import annotations

import math

import numpy as np

from .constants import FLATTENING, OMEGA_EARTH, R_EARTH

_E2 = FLATTENING * (2.0 - FLATTENING)


def gmst(t_s) -> np.ndarray:
    """Greenwich Mean Sidereal Time [rad] (IAU-82), t in seconds since J2000 (UT1 ~ UTC)."""
    t = np.asarray(t_s, dtype=float)
    T = t / (86400.0 * 36525.0)
    seconds = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
    return np.deg2rad(np.mod(seconds, 86400.0) / 240.0)


def eci_to_ecef(states: np.ndarray, t_s) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ECI states (N, 6) into Earth-fixed position and velocity (N, 3 each)."""
    theta = np.atleast_1d(gmst(t_s))
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    vx, vy, vz = states[:, 3], states[:, 4], states[:, 5]
    xe = c * x + s * y
    ye = -s * x + c * y
    r = np.stack([xe, ye, z], axis=-1)
    # v_ecef = R v_eci - omega x r_ecef
    v = np.stack([c * vx + s * vy + OMEGA_EARTH * ye, -s * vx + c * vy - OMEGA_EARTH * xe, vz], axis=-1)
    return r, v


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS-84 geodetic coordinates to Earth-fixed Cartesian position [km]."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = R_EARTH / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_km) * math.cos(lat) * math.cos(lon),
            (n + alt_km) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - _E2) + alt_km) * math.sin(lat),
        ]
    )


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows: East, North, Up unit vectors in ECEF for a geodetic location."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def look_angles(r_ecef: np.ndarray, v_ecef: np.ndarray, site_ecef: np.ndarray, site_enu: np.ndarray):
    """Azimuth [rad], elevation [rad], range [km] and range rate [km/s] from a ground site."""
    rho = r_ecef - site_ecef
    enu = rho @ site_enu.T
    rng = np.linalg.norm(rho, axis=-1)
    el = np.arcsin(np.clip(enu[..., 2] / rng, -1.0, 1.0))
    az = np.mod(np.arctan2(enu[..., 0], enu[..., 1]), 2.0 * np.pi)
    range_rate = np.einsum("...i,...i->...", rho, v_ecef) / rng
    return az, el, rng, range_rate
