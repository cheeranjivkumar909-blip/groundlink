"""Dependency-free SVG Gantt chart of a contact schedule."""

from __future__ import annotations

from html import escape

from .scheduler import Schedule

PALETTE = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563")


def schedule_svg(
    schedule: Schedule,
    stations: list[str],
    satellites: list[str],
    start_s: float,
    end_s: float,
    title: str = "Ground-station contact schedule",
) -> str:
    width, left, right, top, row_h = 1100, 150, 30, 70, 38
    height = top + row_h * len(stations) + 90
    span = end_s - start_s
    plot_w = width - left - right

    def x(t: float) -> float:
        return left + (t - start_s) / span * plot_w

    color = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(satellites)}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="18" font-weight="bold" fill="#111827">{escape(title)}</text>',
        f'<text x="{left}" y="52" font-size="12" fill="#4b5563">'
        f"{len(schedule.contacts)} contacts, {schedule.total_delivered_gbit:.0f} Gbit delivered ({schedule.method})</text>",
    ]
    hours = int(round(span / 3600.0))
    step = 3 if hours > 12 else 1
    for h in range(0, hours + 1, step):
        xx = x(start_s + h * 3600.0)
        out.append(f'<line x1="{xx:.1f}" y1="{top - 6}" x2="{xx:.1f}" y2="{top + row_h * len(stations)}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xx:.1f}" y="{top + row_h * len(stations) + 18}" font-size="11" fill="#6b7280" text-anchor="middle">{h:02d}h</text>')
    for r, st in enumerate(stations):
        y = top + r * row_h
        out.append(f'<text x="{left - 10}" y="{y + row_h / 2 + 4:.1f}" font-size="13" fill="#111827" text-anchor="end">{escape(st)}</text>')
        out.append(f'<rect x="{left}" y="{y + 4}" width="{plot_w}" height="{row_h - 8}" fill="#f9fafb" stroke="#e5e7eb"/>')
        for c in schedule.contacts:
            if c.station != st:
                continue
            x0, x1 = x(c.start_s), x(c.end_s)
            out.append(
                f'<rect x="{x0:.1f}" y="{y + 7}" width="{max(x1 - x0, 1.5):.1f}" height="{row_h - 14}" rx="2" '
                f'fill="{color[c.satellite]}"><title>{escape(c.satellite)} - {c.volume_gbit:.1f} Gbit</title></rect>'
            )
    ly = top + row_h * len(stations) + 45
    for k, s in enumerate(satellites):
        lx = left + k * 140
        out.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{color[s]}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly}" font-size="12" fill="#374151">{escape(s)}</text>')
    out.append("</svg>")
    return "\n".join(out)
