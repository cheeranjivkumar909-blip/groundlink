import math

import numpy as np
import pytest

from groundlink.link import MODCODS, LinkBudget, doppler_shift_hz, free_space_path_loss_db
from groundlink.passes import find_passes, look, pass_profile
from groundlink.planner import demo_problem
from groundlink.stations import GroundStation


@pytest.fixture(scope="module")
def problem():
    return demo_problem(24.0)


def test_fspl_reference_value():
    # 2.2 GHz over 1000 km: 20log10(1e6) + 20log10(2.2e9) - 147.55 = 159.30 dB
    assert float(free_space_path_loss_db(1000.0, 2.2e9)) == pytest.approx(159.30, abs=0.01)


def test_doppler_sign_and_magnitude():
    assert float(doppler_shift_hz(-7.0, 8.2e9)) == pytest.approx(191_469, rel=1e-3)
    assert float(doppler_shift_hz(7.0, 8.2e9)) < 0.0


def test_modcod_selection_is_monotonic_and_closes():
    link = LinkBudget()
    esn0 = np.linspace(-10, 25, 300)
    idx = link.select_modcod(esn0)
    eff = np.array([MODCODS[i][2] if i >= 0 else 0.0 for i in idx])
    assert np.all(np.diff(eff) >= 0)
    assert idx[0] == -1  # link closed at very low Es/N0
    assert MODCODS[idx[-1]][0] == "16APSK 9/10"


def test_data_rate_increases_with_elevation():
    link = LinkBudget()
    ranges = np.array([2300.0, 1200.0, 560.0])
    els = np.radians([5.0, 25.0, 90.0])
    rates = link.data_rate_bps(ranges, els, 25.0)
    assert np.all(np.diff(rates) >= 0) and rates[-1] > 50e6


def test_pass_geometry(problem):
    sat = problem.satellites[0]
    st = next(s for s in problem.stations if s.name == "Svalbard")
    passes = find_passes(sat.propagator, st, problem.start_s, problem.end_s, satellite=sat.name)
    # A sun-synchronous satellite is visible from 78 deg N on most of its ~15 daily orbits.
    assert 10 <= len(passes) <= 16
    for p in passes:
        assert p.aos_s < p.tmax_s < p.los_s
        assert p.max_elevation_deg >= st.min_elevation_deg
        if not p.truncated:
            el = np.degrees(look(sat.propagator, st, [p.aos_s, p.los_s])[1])
            assert np.allclose(el, st.min_elevation_deg, atol=1e-3)
        assert 60.0 < p.duration_s < 16 * 60.0


def test_pass_profile_doppler_flips_sign(problem):
    sat = problem.satellites[0]
    st = next(s for s in problem.stations if s.name == "Svalbard")
    p = max(find_passes(sat.propagator, st, problem.start_s, problem.end_s), key=lambda p: p.max_elevation_deg)
    prof = pass_profile(sat.propagator, st, p, LinkBudget())
    assert prof.doppler_hz[0] > 0 > prof.doppler_hz[-1]  # approaching, then receding
    assert prof.max_doppler_hz < 8.2e9 * 8.0 / 299792.458
    assert prof.volume_bits > 0
    k = int(np.argmax(prof.elevation_deg))
    assert prof.data_rate_bps[k] == prof.peak_rate_bps


def test_equatorial_station_sees_fewer_polar_passes(problem):
    sat = problem.satellites[0]
    polar = GroundStation("polar", 78.0, 15.0, 0.0, 5.0)
    equatorial = GroundStation("equator", 0.0, 15.0, 0.0, 5.0)
    n_polar = len(find_passes(sat.propagator, polar, problem.start_s, problem.end_s))
    n_eq = len(find_passes(sat.propagator, equatorial, problem.start_s, problem.end_s))
    assert n_polar > 2 * n_eq


def test_find_passes_rejects_bad_window(problem):
    with pytest.raises(ValueError):
        find_passes(problem.satellites[0].propagator, problem.stations[0], 10.0, 0.0)
    assert math.isfinite(problem.start_s)
