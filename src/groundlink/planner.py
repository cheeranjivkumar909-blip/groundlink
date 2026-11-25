"""End-to-end planning: constellation + network -> passes -> contacts -> schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .constants import R_EARTH
from .elements import KeplerianElements
from .epochs import datetime_to_seconds
from .link import LinkBudget
from .passes import Pass, PassProfile, find_passes, pass_profile
from .propagation import SecularJ2Propagator
from .scheduler import Contact, SatelliteDemand
from .stations import GroundStation, demo_network


@dataclass
class Satellite:
    name: str
    propagator: SecularJ2Propagator
    demand: SatelliteDemand


@dataclass
class PlanningProblem:
    satellites: list[Satellite]
    stations: list[GroundStation]
    start_s: float
    end_s: float
    link: LinkBudget = field(default_factory=LinkBudget)
    setup_s: float = 120.0


@dataclass
class Opportunity:
    pass_: Pass
    profile: PassProfile
    contact: Contact


def build_opportunities(problem: PlanningProblem, step_s: float = 20.0) -> list[Opportunity]:
    """Predict every pass and convert it into a schedulable contact with its data volume."""
    out: list[Opportunity] = []
    for sat in problem.satellites:
        for st in problem.stations:
            for p in find_passes(sat.propagator, st, problem.start_s, problem.end_s, step_s, sat.name):
                prof = pass_profile(sat.propagator, st, p, problem.link)
                volume = prof.volume_bits / 1e9
                if volume <= 0.0:
                    continue
                out.append(
                    Opportunity(p, prof, Contact(sat.name, st.name, p.aos_s, p.los_s, volume, p.max_elevation_deg))
                )
    out.sort(key=lambda o: o.contact.start_s)
    return out


def demo_problem(hours: float = 24.0) -> PlanningProblem:
    """Six-satellite Earth-observation constellation (two sun-synchronous planes) and a five-site network."""
    epoch = datetime_to_seconds(datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc))
    specs = [
        # name, altitude [km], RAAN [deg], phase [deg], backlog [Gbit], priority
        ("EO-SAT-1", 520.0, 30.0, 0.0, 1200.0, 1.0),
        ("EO-SAT-2", 520.0, 30.0, 120.0, 900.0, 1.0),
        ("EO-SAT-3", 520.0, 30.0, 240.0, 1500.0, 2.0),
        ("EO-SAT-4", 560.0, 120.0, 60.0, 800.0, 1.0),
        ("EO-SAT-5", 560.0, 120.0, 180.0, 1100.0, 1.5),
        ("EO-SAT-6", 560.0, 120.0, 300.0, 1000.0, 1.0),
    ]
    sats = []
    for name, alt, raan, phase, backlog, prio in specs:
        a = R_EARTH + alt
        # Sun-synchronous inclination for a circular orbit at this altitude.
        n = math.sqrt(398600.4418 / a**3)
        cos_i = -(2.0 * math.pi / (365.2422 * 86400.0)) / (1.5 * 1.08262668e-3 * (R_EARTH / a) ** 2 * n)
        el = KeplerianElements(a, 0.001, math.acos(cos_i), math.radians(raan), 0.0, math.radians(phase))
        sats.append(Satellite(name, SecularJ2Propagator(el, epoch), SatelliteDemand(name, backlog, prio)))
    return PlanningProblem(sats, demo_network(), epoch, epoch + hours * 3600.0)
