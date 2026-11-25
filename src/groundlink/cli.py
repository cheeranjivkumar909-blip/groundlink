"""Command-line interface: ``groundlink demo | passes``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import __version__
from .epochs import datetime_to_seconds, format_iso, parse_iso
from .link import MODCODS, LinkBudget
from .passes import find_passes, pass_profile
from .planner import build_opportunities, demo_problem
from .propagation import SecularJ2Propagator
from .scheduler import schedule_greedy, schedule_optimal
from .stations import GroundStation, demo_network
from .timeline import schedule_svg
from .tle import parse_tle_file

RULE = "-" * 72


def run_demo(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    problem = demo_problem(args.hours)
    opps = build_opportunities(problem)
    contacts = [o.contact for o in opps]
    demands = [s.demand for s in problem.satellites]
    greedy = schedule_greedy(contacts, demands, problem.setup_s)
    best = schedule_optimal(contacts, demands, problem.setup_s)

    print("GroundLink - Ground-Station Contact Planning & Downlink Scheduling")
    print("=" * 72)
    print(f"Constellation : {len(problem.satellites)} EO satellites, 2 sun-synchronous planes (520/560 km)")
    print(f"Network       : {len(problem.stations)} X-band stations | horizon {args.hours:g} h from {format_iso(problem.start_s)}")
    print(f"Link          : {problem.link.frequency_hz / 1e9:.1f} GHz, EIRP {problem.link.eirp_dbw:g} dBW, "
          f"{problem.link.symbol_rate_sps / 1e6:g} Msps, DVB-S2 ACM ({len(MODCODS)} MODCODs)")
    print(RULE)
    print(f"{'Station':<16}{'Passes':>7}{'Contact [min]':>15}{'Mean max-el':>13}{'Volume [Gbit]':>15}")
    for st in problem.stations:
        mine = [o for o in opps if o.contact.station == st.name]
        if not mine:
            continue
        print(f"{st.name:<16}{len(mine):>7}{sum(o.contact.duration_s for o in mine) / 60:>15.1f}"
              f"{sum(o.pass_.max_elevation_deg for o in mine) / len(mine):>12.1f}°"
              f"{sum(o.contact.volume_gbit for o in mine):>15.1f}")
    print(f"Total candidate contacts: {len(opps)} | offered volume {sum(c.volume_gbit for c in contacts):.0f} Gbit")

    best_pass = max(opps, key=lambda o: o.profile.peak_rate_bps)
    print(RULE)
    print("EXAMPLE PASS")
    p, prof = best_pass.pass_, best_pass.profile
    print(f"  {p.satellite} over {p.station}: AOS {format_iso(p.aos_s)}  LOS {format_iso(p.los_s)}  ({p.duration_s / 60:.1f} min)")
    print(f"  Max elevation {p.max_elevation_deg:.1f}° | max Doppler ±{prof.max_doppler_hz / 1e3:.0f} kHz | "
          f"peak rate {prof.peak_rate_bps / 1e6:.1f} Mbit/s | volume {prof.volume_bits / 1e9:.1f} Gbit")

    print(RULE)
    print(f"{'SCHEDULE':<22}{'Contacts':>10}{'Delivered [Gbit]':>18}{'Weighted value':>16}{'Solve [s]':>11}")
    for s in (greedy, best):
        label = "Greedy heuristic" if s.method == "greedy" else "Optimal (MILP)"
        print(f"{label:<22}{len(s.contacts):>10}{s.total_delivered_gbit:>18.1f}{s.objective:>16.1f}{s.solve_time_s:>11.3f}")
    gain = (best.objective / greedy.objective - 1.0) * 100.0 if greedy.objective else 0.0
    print(f"Optimiser gain over greedy: {gain:+.1f} % weighted value, "
          f"{best.total_delivered_gbit - greedy.total_delivered_gbit:+.1f} Gbit")
    print(RULE)
    print(f"{'Satellite':<12}{'Priority':>9}{'Backlog':>10}{'Greedy':>10}{'Optimal':>10}")
    for sat in problem.satellites:
        d = sat.demand
        print(f"{d.name:<12}{d.priority:>9g}{d.backlog_gbit:>10.0f}{greedy.delivered_gbit[d.name]:>10.1f}{best.delivered_gbit[d.name]:>10.1f}")
    print(RULE)
    print(f"Completed in {time.perf_counter() - t0:.1f} s")
    if args.svg:
        svg = schedule_svg(best, [s.name for s in problem.stations], [s.name for s in problem.satellites],
                           problem.start_s, problem.end_s, "GroundLink optimal contact schedule (24 h)")
        Path(args.svg).write_text(svg, encoding="utf-8")
        print(f"Timeline written to {args.svg}")
    if args.json:
        Path(args.json).write_text(json.dumps({"greedy": greedy.to_dict(), "optimal": best.to_dict()}, indent=2), encoding="utf-8")
        print(f"JSON schedule written to {args.json}")
    return 0


def run_passes(args: argparse.Namespace) -> int:
    tles = parse_tle_file(Path(args.tle).read_text(encoding="utf-8"))
    if args.lat is not None and args.lon is not None:
        stations = [GroundStation("Custom site", args.lat, args.lon, args.alt, args.min_el)]
    else:
        stations = [s for s in demo_network() if s.name.lower() == args.station.lower()]
        if not stations:
            raise SystemExit(f"unknown station {args.station!r}; available: {', '.join(s.name for s in demo_network())}")
    station = stations[0]
    link = LinkBudget()
    for tle in tles:
        prop = SecularJ2Propagator.from_tle(tle)
        start = datetime_to_seconds(parse_iso(args.start)) if args.start else tle.epoch_seconds
        print(f"{tle.name} over {station.name} (mask {station.min_elevation_deg:g}°)")
        print(f"  {'AOS (UTC)':<26}{'LOS (UTC)':<26}{'Dur [min]':>10}{'Max el':>8}{'Doppler':>11}")
        for p in find_passes(prop, station, start, start + args.hours * 3600.0, satellite=tle.name):
            prof = pass_profile(prop, station, p, link)
            print(f"  {format_iso(p.aos_s):<26}{format_iso(p.los_s):<26}{p.duration_s / 60:>10.1f}"
                  f"{p.max_elevation_deg:>7.1f}°{prof.max_doppler_hz / 1e3:>8.0f} kHz")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groundlink", description="Ground-station contact planning and downlink scheduling")
    parser.add_argument("--version", action="version", version=f"groundlink {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="plan a 24 h schedule for a 6-satellite constellation")
    demo.add_argument("--hours", type=float, default=24.0)
    demo.add_argument("--svg", help="write the optimal schedule as an SVG timeline")
    demo.add_argument("--json", help="write both schedules as JSON")
    demo.set_defaults(func=run_demo)

    passes = sub.add_parser("passes", help="predict passes of TLE objects over a station")
    passes.add_argument("tle", help="TLE file (2-line or 3-line format)")
    passes.add_argument("--station", default="Bengaluru", help="name of a demo-network station")
    passes.add_argument("--lat", type=float, help="custom site latitude [deg]")
    passes.add_argument("--lon", type=float, help="custom site longitude [deg]")
    passes.add_argument("--alt", type=float, default=0.0, help="custom site altitude [km]")
    passes.add_argument("--min-el", type=float, default=10.0, help="custom site elevation mask [deg]")
    passes.add_argument("--start", help="ISO-8601 start epoch (default: TLE epoch)")
    passes.add_argument("--hours", type=float, default=24.0)
    passes.set_defaults(func=run_passes)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
