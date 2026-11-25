import itertools

import pytest

from groundlink.cli import main
from groundlink.planner import build_opportunities, demo_problem
from groundlink.scheduler import (
    Contact,
    SatelliteDemand,
    clique_rows,
    conflicts,
    schedule_greedy,
    schedule_optimal,
)
from groundlink.tle import parse_tle

ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


def _feasible(schedule, setup_s):
    for a, b in itertools.combinations(schedule.contacts, 2):
        assert not conflicts(a, b, setup_s), (a, b)


def test_optimal_beats_greedy_on_classic_counterexample():
    # Greedy grabs the single most valuable contact A, which blocks B and C (worth more together).
    contacts = [
        Contact("S1", "G1", 0.0, 600.0, 10.0),
        Contact("S2", "G1", -300.0, 100.0, 6.0),
        Contact("S3", "G1", 500.0, 900.0, 6.0),
    ]
    greedy = schedule_greedy(contacts, setup_s=0.0)
    best = schedule_optimal(contacts, setup_s=0.0)
    assert greedy.total_delivered_gbit == pytest.approx(10.0)
    assert best.total_delivered_gbit == pytest.approx(12.0)
    _feasible(best, 0.0)


def test_setup_time_blocks_back_to_back_contacts():
    contacts = [Contact("S1", "G1", 0.0, 100.0, 5.0), Contact("S2", "G1", 150.0, 250.0, 5.0)]
    assert schedule_optimal(contacts, setup_s=0.0).total_delivered_gbit == pytest.approx(10.0)
    assert schedule_optimal(contacts, setup_s=60.0).total_delivered_gbit == pytest.approx(5.0)


def test_satellite_cannot_use_two_stations_at_once():
    contacts = [Contact("S1", "G1", 0.0, 100.0, 5.0), Contact("S1", "G2", 50.0, 150.0, 4.0)]
    assert len(schedule_optimal(contacts).contacts) == 1


def test_backlog_caps_delivery_and_priority_steers():
    contacts = [Contact("S1", "G1", 0.0, 100.0, 50.0), Contact("S2", "G1", 50.0, 150.0, 30.0)]
    demands = [SatelliteDemand("S1", backlog_gbit=20.0), SatelliteDemand("S2", priority=1.0)]
    best = schedule_optimal(contacts, demands, setup_s=0.0)
    assert best.delivered_gbit == {"S1": 0.0, "S2": 30.0}
    demands = [SatelliteDemand("S1", backlog_gbit=20.0, priority=2.0), SatelliteDemand("S2")]
    best = schedule_optimal(contacts, demands, setup_s=0.0)
    assert best.delivered_gbit["S1"] == pytest.approx(20.0)


def test_clique_rows_cover_every_conflicting_pair():
    contacts = [Contact(f"S{k % 3}", f"G{k % 2}", k * 70.0, k * 70.0 + 200.0, 1.0) for k in range(12)]
    rows = clique_rows(contacts, setup_s=30.0)
    for i, j in itertools.combinations(range(len(contacts)), 2):
        if conflicts(contacts[i], contacts[j], 30.0):
            assert any(i in r and j in r for r in rows)


@pytest.fixture(scope="module")
def demo():
    problem = demo_problem(24.0)
    opps = build_opportunities(problem)
    contacts = [o.contact for o in opps]
    demands = [s.demand for s in problem.satellites]
    return problem, contacts, demands


def test_demo_schedules_are_feasible_and_optimal_dominates(demo):
    problem, contacts, demands = demo
    greedy = schedule_greedy(contacts, demands, problem.setup_s)
    best = schedule_optimal(contacts, demands, problem.setup_s)
    _feasible(greedy, problem.setup_s)
    _feasible(best, problem.setup_s)
    assert best.objective >= greedy.objective - 1e-6
    for d in demands:
        assert best.delivered_gbit[d.name] <= d.backlog_gbit + 1e-6


def test_tle_parsing_shared_module():
    tle = parse_tle(ISS_L1, ISS_L2)
    assert tle.norad_id == 25544 and tle.inclination_deg == pytest.approx(51.6416)


def test_cli_demo_runs(tmp_path, capsys):
    svg = tmp_path / "s.svg"
    assert main(["demo", "--hours", "6", "--svg", str(svg), "--json", str(tmp_path / "s.json")]) == 0
    assert svg.read_text().startswith("<svg")
    assert "Optimal (MILP)" in capsys.readouterr().out
