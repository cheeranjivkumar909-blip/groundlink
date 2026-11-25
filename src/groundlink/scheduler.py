"""Contact scheduling: assign passes to antennas to maximise delivered data.

Problem
    Given candidate contacts i (satellite s_i, station g_i, interval [a_i, b_i],
    downlinkable volume V_i) choose x_i in {0, 1} to

        maximise   sum_s w_s * y_s
        subject to y_s <= sum_{i: s_i = s} V_i x_i      (data actually downlinked)
                   0 <= y_s <= B_s                     (on-board backlog)
                   one contact at a time per antenna, with set-up/slew time T_setup
                   one contact at a time per satellite

Overlap constraints are written as clique inequalities: for interval graphs the
maximal cliques are the sets of intervals containing some interval's start
point, so the formulation is tight and compact. Solved exactly with HiGHS
(``scipy.optimize.milp``); a priority-greedy heuristic is provided as baseline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .epochs import format_iso


@dataclass(frozen=True)
class Contact:
    satellite: str
    station: str
    start_s: float
    end_s: float
    volume_gbit: float
    max_elevation_deg: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SatelliteDemand:
    name: str
    backlog_gbit: float = math.inf
    priority: float = 1.0


@dataclass
class Schedule:
    method: str
    contacts: list[Contact]
    delivered_gbit: dict[str, float]
    objective: float
    solve_time_s: float
    candidates: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total_delivered_gbit(self) -> float:
        return float(sum(self.delivered_gbit.values()))

    def station_busy_s(self) -> dict[str, float]:
        busy: dict[str, float] = {}
        for c in self.contacts:
            busy[c.station] = busy.get(c.station, 0.0) + c.duration_s
        return busy

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "objective": round(self.objective, 3),
            "total_delivered_gbit": round(self.total_delivered_gbit, 3),
            "delivered_gbit": {k: round(v, 3) for k, v in self.delivered_gbit.items()},
            "solve_time_s": round(self.solve_time_s, 4),
            "contacts": [
                {
                    "satellite": c.satellite,
                    "station": c.station,
                    "start_utc": format_iso(c.start_s),
                    "end_utc": format_iso(c.end_s),
                    "volume_gbit": round(c.volume_gbit, 3),
                }
                for c in sorted(self.contacts, key=lambda c: c.start_s)
            ],
        }


def _demand_map(contacts: list[Contact], demands: list[SatelliteDemand] | None) -> dict[str, SatelliteDemand]:
    d = {s.name: s for s in (demands or [])}
    for c in contacts:
        d.setdefault(c.satellite, SatelliteDemand(c.satellite))
    return d


def _overlaps(a: Contact, b: Contact, setup_s: float) -> bool:
    pad = setup_s if a.station == b.station else 0.0
    return a.start_s < b.end_s + pad and b.start_s < a.end_s + pad


def conflicts(a: Contact, b: Contact, setup_s: float) -> bool:
    """True if two contacts cannot both be scheduled."""
    if a.station == b.station or a.satellite == b.satellite:
        return _overlaps(a, b, setup_s)
    return False


def clique_rows(contacts: list[Contact], setup_s: float) -> list[list[int]]:
    """Clique inequalities for every antenna and every satellite (deduplicated)."""
    rows: set[tuple[int, ...]] = set()
    for key in ("station", "satellite"):
        pad = setup_s if key == "station" else 0.0
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(contacts):
            groups.setdefault(getattr(c, key), []).append(i)
        for members in groups.values():
            for i in members:
                s = contacts[i].start_s
                clique = tuple(
                    sorted(j for j in members if contacts[j].start_s <= s < contacts[j].end_s + pad)
                )
                if len(clique) > 1:
                    rows.add(clique)
    return [list(r) for r in sorted(rows)]


def schedule_greedy(
    contacts: list[Contact], demands: list[SatelliteDemand] | None = None, setup_s: float = 120.0
) -> Schedule:
    """Highest weighted-volume-first heuristic, respecting backlogs and conflicts."""
    t0 = time.perf_counter()
    dm = _demand_map(contacts, demands)
    delivered = {k: 0.0 for k in dm}
    chosen: list[Contact] = []
    for c in sorted(contacts, key=lambda c: -dm[c.satellite].priority * c.volume_gbit):
        room = dm[c.satellite].backlog_gbit - delivered[c.satellite]
        if room <= 0.0:
            continue
        if any(conflicts(c, o, setup_s) for o in chosen):
            continue
        chosen.append(c)
        delivered[c.satellite] += min(c.volume_gbit, room)
    objective = sum(dm[k].priority * v for k, v in delivered.items())
    return Schedule("greedy", chosen, delivered, objective, time.perf_counter() - t0, len(contacts))


def schedule_optimal(
    contacts: list[Contact],
    demands: list[SatelliteDemand] | None = None,
    setup_s: float = 120.0,
    time_limit_s: float = 60.0,
) -> Schedule:
    """Exact mixed-integer solution (HiGHS branch-and-cut via scipy.optimize.milp)."""
    t0 = time.perf_counter()
    dm = _demand_map(contacts, demands)
    sats = sorted(dm)
    n, m = len(contacts), len(sats)
    if n == 0:
        return Schedule("milp", [], {s: 0.0 for s in sats}, 0.0, 0.0, 0)
    sat_idx = {s: k for k, s in enumerate(sats)}

    # Variables: x_0..x_{n-1} (binary), y_0..y_{m-1} (continuous delivered volume).
    # Tiny per-contact cost: among equally valuable plans, prefer fewer antenna bookings.
    c = np.concatenate([np.full(n, 1e-3), -np.array([dm[s].priority for s in sats])])
    integrality = np.concatenate([np.ones(n), np.zeros(m)])
    upper = np.concatenate([np.ones(n), [dm[s].backlog_gbit for s in sats]])
    bounds = Bounds(np.zeros(n + m), upper)

    rows = clique_rows(contacts, setup_s)
    a_clique = np.zeros((len(rows), n + m))
    for r, members in enumerate(rows):
        a_clique[r, members] = 1.0
    a_vol = np.zeros((m, n + m))
    for i, ct in enumerate(contacts):
        a_vol[sat_idx[ct.satellite], i] = -ct.volume_gbit
    a_vol[np.arange(m), n + np.arange(m)] = 1.0

    constraints = [LinearConstraint(a_vol, -np.inf, 0.0)]
    if rows:
        constraints.append(LinearConstraint(a_clique, -np.inf, 1.0))
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": time_limit_s, "mip_rel_gap": 1e-7})
    if res.x is None:  # pragma: no cover - defensive
        raise RuntimeError(f"MILP failed: {res.message}")
    x = res.x[:n] > 0.5
    chosen = [ct for ct, keep in zip(contacts, x) if keep]
    delivered = {s: float(min(res.x[n + sat_idx[s]], sum(ct.volume_gbit for ct in chosen if ct.satellite == s))) for s in sats}
    objective = sum(dm[s].priority * v for s, v in delivered.items())
    notes = [] if res.status == 0 else [f"solver status {res.status}: {res.message}"]
    return Schedule("milp", chosen, delivered, objective, time.perf_counter() - t0, n, notes)
