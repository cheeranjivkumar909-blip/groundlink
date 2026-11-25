import math

import numpy as np
import pytest

from groundlink.constants import OMEGA_EARTH, R_EARTH
from groundlink.frames import eci_to_ecef, enu_basis, geodetic_to_ecef, gmst, look_angles
from groundlink.stations import GroundStation


def test_gmst_at_j2000():
    assert math.degrees(float(gmst(0.0))) == pytest.approx(280.46061837, abs=1e-6)


def test_gmst_vallado_example_3_5():
    # Vallado, Fundamentals of Astrodynamics, Example 3-5: 1992-08-20 12:14 UT1.
    t = (2448855.009722 - 2451545.0) * 86400.0
    assert math.degrees(float(gmst(t))) == pytest.approx(152.578787886, abs=1e-4)


def test_geodetic_to_ecef_reference_points():
    assert np.allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [R_EARTH, 0.0, 0.0])
    assert np.allclose(geodetic_to_ecef(90.0, 0.0, 0.0), [0.0, 0.0, 6356.752314245], atol=1e-6)
    assert np.allclose(geodetic_to_ecef(0.0, 90.0, 1.0), [0.0, R_EARTH + 1.0, 0.0])


def test_enu_basis_is_orthonormal_and_right_handed():
    m = enu_basis(13.03, 77.51)
    assert np.allclose(m @ m.T, np.eye(3))
    assert np.allclose(np.cross(m[0], m[1]), m[2])


def test_overhead_target_has_90_deg_elevation():
    st = GroundStation("site", 28.6, 77.2, 0.2)
    up = st.enu[2]
    r = st.ecef + 600.0 * up
    az, el, rng, rr = look_angles(r[None, :], np.zeros((1, 3)), st.ecef, st.enu)
    assert math.degrees(el[0]) == pytest.approx(90.0, abs=1e-9)
    assert rng[0] == pytest.approx(600.0)
    assert rr[0] == pytest.approx(0.0)


def test_azimuth_convention_north_and_east():
    st = GroundStation("site", 0.0, 0.0, 0.0)
    north = st.ecef + 100.0 * st.enu[1] + 1.0 * st.enu[2]
    east = st.ecef + 100.0 * st.enu[0] + 1.0 * st.enu[2]
    az, _, _, _ = look_angles(np.vstack([north, east]), np.zeros((2, 3)), st.ecef, st.enu)
    assert math.degrees(az[0]) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(az[1]) == pytest.approx(90.0, abs=1e-9)


def test_eci_to_ecef_preserves_norm_and_handles_rotation():
    t = np.array([0.0, 3600.0])
    states = np.array([[7000.0, 0.0, 100.0, 0.0, 7.5, 0.1]] * 2)
    r, v = eci_to_ecef(states, t)
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(states[:, :3], axis=1))
    # A point fixed in inertial space appears to move west at omega * r in ECEF.
    still = np.array([[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, v_still = eci_to_ecef(still, [0.0])
    assert np.linalg.norm(v_still) == pytest.approx(OMEGA_EARTH * 7000.0)
