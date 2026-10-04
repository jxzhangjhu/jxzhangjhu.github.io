#!/usr/bin/env python3
"""Generate the original vector figures for the calibration blog post.

The script uses only the Python standard library so the diagrams remain easy to
rebuild in a clean checkout. Paper figures are extracted separately from their
source PDFs and are not generated here.
"""

from __future__ import annotations

import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "img" / "blog" / "calibrating-long-horizon-agents"

INK = "#202634"
MUTED = "#687386"
BLUE = "#2864dc"
BLUE_LIGHT = "#edf3ff"
GREEN = "#109a70"
GREEN_LIGHT = "#ecf8f3"
ORANGE = "#d97706"
ORANGE_LIGHT = "#fff4e5"
PURPLE = "#6d4fd3"
GRID = "#d9dee7"
CARD = "#f7f8fa"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(x: float, y: float, value: str, *, size: int = 24, weight: int = 400,
         fill: str = INK, anchor: str = "middle", italic: bool = False) -> str:
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" style="{style}">{esc(value)}</text>'
    )


def multiline(x: float, y: float, lines: list[str], *, size: int = 22,
              weight: int = 400, fill: str = INK, gap: int = 30,
              anchor: str = "middle") -> str:
    body = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">'
    ]
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else gap
        body.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    body.append("</text>")
    return "".join(body)


def rounded_rect(x: float, y: float, w: float, h: float, *, fill: str = "white",
                 stroke: str = GRID, width: int = 2, radius: int = 18) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def svg_doc(width: int, height: int, body: str, *, title: str, desc: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(desc)}</desc>
  <defs>
    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/>
    </marker>
    <marker id="arrow-muted" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>
'''


def write(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content, encoding="utf-8")


def reliability_loop() -> None:
    width, height = 1600, 820
    cards = [
        (55, "1", "MEASURE", ["trajectory uncertainty", "state + tool evidence"], BLUE, BLUE_LIGHT),
        (440, "2", "CALIBRATE", ["confidence to", "trajectory success"], PURPLE, "#f2efff"),
        (825, "3", "ACT", ["ask · verify · reflect", "route · stop"], ORANGE, ORANGE_LIGHT),
        (1210, "4", "SELF-EVOLVE", ["memory · skills · tools", "verified recoveries"], GREEN, GREEN_LIGHT),
    ]
    parts = [
        text(800, 66, "THE AGENTIC RELIABILITY LOOP", size=43, weight=800),
        text(800, 108, "Frozen model · uncertainty as the runtime control plane", size=25, fill=MUTED),
    ]
    card_w, card_h, card_y = 335, 250, 190
    for x, number, heading, sublines, color, fill in cards:
        parts.append(rounded_rect(x, card_y, card_w, card_h, fill=fill, stroke=color, width=4, radius=22))
        parts.append(f'<circle cx="{x + 45}" cy="{card_y + 47}" r="25" fill="{color}"/>')
        parts.append(text(x + 45, card_y + 56, number, size=24, weight=800, fill="white"))
        parts.append(text(x + card_w / 2, card_y + 105, heading, size=29, weight=800, fill=color))
        parts.append(multiline(x + card_w / 2, card_y + 160, sublines, size=22, gap=35, fill=INK))
    for x1, x2 in [(390, 432), (775, 817), (1160, 1202)]:
        parts.append(f'<path d="M {x1} 315 L {x2} 315" stroke="{BLUE}" stroke-width="7" fill="none" marker-end="url(#arrow-blue)"/>')
    parts.extend([
        f'<path d="M 1378 447 C 1378 620, 1260 685, 1085 685 L 300 685 C 125 685, 95 585, 130 455" stroke="{BLUE}" stroke-width="7" fill="none" marker-end="url(#arrow-blue)"/>',
        rounded_rect(435, 545, 730, 78, fill=CARD, stroke=GRID, width=2, radius=39),
        text(800, 579, "Persistent state", size=20, weight=700, fill=MUTED),
        text(800, 609, "evidence · confidence · provenance · memory", size=24, weight=600),
        f'<path d="M 992 440 C 975 490, 945 520, 905 544" stroke="{MUTED}" stroke-width="4" fill="none" marker-end="url(#arrow-muted)"/>',
        text(800, 770, "Detect → decide → recover → retain what worked", size=25, weight=600, fill=MUTED, italic=True),
    ])
    write("fig1_agentic_reliability_loop.svg", svg_doc(width, height, "\n".join(parts),
          title="The agentic reliability loop",
          desc="Measure and calibrate trajectory uncertainty, act on it, then retain verified recoveries in memory, skills, and tools."))


def reliability_profile() -> None:
    width, height = 1800, 560
    items = [
        ("Consistency", ["stable outcomes", "across reruns"], MUTED, CARD),
        ("Robustness", ["graceful under tool, prompt,", "and environment shift"], MUTED, CARD),
        ("Predictability", ["confidence predicts", "trajectory success"], BLUE, BLUE_LIGHT),
        ("Safety", ["bounded harm; safe", "fallback and escalation"], GREEN, GREEN_LIGHT),
    ]
    parts = [
        text(900, 66, "A RELIABILITY PROFILE FOR LONG-HORIZON AGENTS", size=42, weight=800),
        text(900, 108, "Calibrated uncertainty directly supports predictability and safety", size=25, fill=MUTED),
    ]
    x0, gap, w, h, y = 55, 30, 400, 310, 160
    for i, (heading, lines, color, fill) in enumerate(items):
        x = x0 + i * (w + gap)
        parts.append(rounded_rect(x, y, w, h, fill=fill, stroke=color, width=4, radius=22))
        parts.append(text(x + w / 2, y + 98, heading.upper(), size=30, weight=800, fill=color))
        parts.append(multiline(x + w / 2, y + 165, lines, size=24, gap=36))
        if i >= 2:
            parts.append(text(x + w / 2, y + 270, "uncertainty lever", size=19, weight=700, fill=color))
    parts.append(text(900, 525, "Reliability is a profile—not a single benchmark score.", size=24, weight=600, fill=MUTED, italic=True))
    write("fig2_reliability_profile.svg", svg_doc(width, height, "\n".join(parts),
          title="A reliability profile for long-horizon agents",
          desc="Consistency, robustness, predictability, and safety; calibrated uncertainty most directly supports the latter two."))


def horizon_plot() -> None:
    width, height = 1700, 900
    left, right, top, bottom = 150, 70, 135, 120
    pw, ph = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 50.0, 100.0, 0.0, 150.0

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * pw

    def sy(value: float) -> float:
        return top + ph - (value - ymin) / (ymax - ymin) * ph

    parts = [
        text(width / 2, 62, "SMALL PER-STEP GAINS → LARGE HORIZON GAINS", size=42, weight=800),
        f'<rect x="{sx(80)}" y="{top}" width="{sx(100)-sx(80)}" height="{ph}" fill="{ORANGE_LIGHT}"/>',
    ]
    for yv in range(0, 151, 25):
        parts.append(f'<line x1="{left}" y1="{sy(yv)}" x2="{width-right}" y2="{sy(yv)}" stroke="{GRID}" stroke-width="2"/>')
        parts.append(text(left - 22, sy(yv) + 8, str(yv), size=21, anchor="end"))
    for xv in range(50, 101, 10):
        parts.append(f'<line x1="{sx(xv)}" y1="{top}" x2="{sx(xv)}" y2="{top+ph}" stroke="{GRID}" stroke-width="2"/>')
        parts.append(text(sx(xv), top + ph + 45, str(xv), size=21))
    parts.extend([
        f'<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}" stroke="{INK}" stroke-width="4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="{INK}" stroke-width="4"/>',
        text(left + pw / 2, height - 34, "per-step reliability p (%)", size=28, weight=600),
        f'<text x="42" y="{top + ph/2}" transform="rotate(-90 42 {top + ph/2})" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="28" font-weight="600" fill="{INK}">reliable horizon length Hₛ(p)</text>',
        multiline(sx(89), sy(118), ["high-accuracy", "regime"], size=27, weight=800, fill=ORANGE, gap=31),
    ])
    for success, color, dash, label in [(0.5, BLUE, "", "target task success s = 0.5"), (0.8, GREEN, "14 10", "target task success s = 0.8")]:
        points = []
        for i in range(500):
            p = 0.50 + (0.995 - 0.50) * i / 499
            horizon = min(ymax, math.log(success) / math.log(p))
            points.append(f"{sx(p*100):.1f},{sy(horizon):.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="{dash}"/>')
        ly = 155 if success == 0.5 else 202
        parts.append(f'<line x1="190" y1="{ly}" x2="280" y2="{ly}" stroke="{color}" stroke-width="8" stroke-dasharray="{dash}"/>')
        parts.append(text(305, ly + 8, label, size=22, anchor="start"))
    write("fig3_horizon_length.svg", svg_doc(width, height, "\n".join(parts),
          title="Reliable horizon length as per-step reliability approaches one",
          desc="A small per-step reliability increase produces a large increase in sustainable task horizon in the high-accuracy regime."))


def reliability_diagram() -> None:
    width, height = 1650, 900
    left, right, top, bottom = 155, 85, 135, 125
    pw, ph = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + value * pw

    def sy(value: float) -> float:
        return top + ph - value * ph

    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    raw = [0.02, 0.06, 0.15, 0.24, 0.32, 0.41, 0.50, 0.60, 0.71]
    calibrated = [0.08, 0.18, 0.29, 0.39, 0.50, 0.61, 0.72, 0.83, 0.93]
    parts = [text(width / 2, 62, "TRAJECTORY RELIABILITY DIAGRAM", size=42, weight=800)]
    for v in [i / 5 for i in range(6)]:
        parts.append(f'<line x1="{left}" y1="{sy(v)}" x2="{width-right}" y2="{sy(v)}" stroke="{GRID}" stroke-width="2"/>')
        parts.append(f'<line x1="{sx(v)}" y1="{top}" x2="{sx(v)}" y2="{top+ph}" stroke="{GRID}" stroke-width="2"/>')
        parts.append(text(left - 20, sy(v) + 8, f"{v:.1f}", size=21, anchor="end"))
        parts.append(text(sx(v), top + ph + 43, f"{v:.1f}", size=21))
    parts.extend([
        f'<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}" stroke="{INK}" stroke-width="4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="{INK}" stroke-width="4"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="{INK}" stroke-width="4" stroke-dasharray="14 10"/>',
        text(left + pw / 2, height - 34, "predicted trajectory confidence", size=28, weight=600),
        f'<text x="43" y="{top + ph/2}" transform="rotate(-90 43 {top + ph/2})" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="28" font-weight="600" fill="{INK}">empirical trajectory success</text>',
    ])
    for values, color, label, shape in [
        (raw, ORANGE, "raw confidence (overconfident)", "circle"),
        (calibrated, GREEN, "after trajectory calibration", "square"),
    ]:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(bins, values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>')
        for x, y in zip(bins, values):
            if shape == "circle":
                parts.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="9" fill="white" stroke="{color}" stroke-width="6"/>')
            else:
                parts.append(f'<rect x="{sx(x)-8}" y="{sy(y)-8}" width="16" height="16" rx="2" fill="white" stroke="{color}" stroke-width="6"/>')
    legend_x, legend_y = 205, 166
    parts.extend([
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+70}" y2="{legend_y}" stroke="{INK}" stroke-width="4" stroke-dasharray="14 10"/>',
        text(legend_x + 92, legend_y + 7, "perfect calibration", size=21, anchor="start"),
        f'<line x1="{legend_x+390}" y1="{legend_y}" x2="{legend_x+460}" y2="{legend_y}" stroke="{ORANGE}" stroke-width="7"/>',
        text(legend_x + 482, legend_y + 7, "raw confidence", size=21, anchor="start"),
        f'<line x1="{legend_x+735}" y1="{legend_y}" x2="{legend_x+805}" y2="{legend_y}" stroke="{GREEN}" stroke-width="7"/>',
        text(legend_x + 827, legend_y + 7, "calibrated", size=21, anchor="start"),
        text(width - 105, height - 34, "Illustrative bins", size=18, fill=MUTED, anchor="end", italic=True),
    ])
    write("fig7_trajectory_reliability.svg", svg_doc(width, height, "\n".join(parts),
          title="Illustrative trajectory reliability diagram",
          desc="Raw confidence lies below the perfect-calibration diagonal, while calibrated trajectory confidence tracks empirical success."))


def measure_align_internalize() -> None:
    width, height = 1800, 650
    items = [
        ("MEASURE", "AUQ", "Inference time", ["propagate trajectory uncertainty", "trigger targeted reflection"], BLUE, BLUE_LIGHT),
        ("ALIGN", "ACC / HTC", "Post hoc", ["map confidence to success", "transfer an interpretable calibrator"], GREEN, GREEN_LIGHT),
        ("INTERNALIZE", "CaOPD", "Training time", ["calibration-aware distillation", "preserve capability and honesty"], ORANGE, ORANGE_LIGHT),
    ]
    parts = [
        text(900, 64, "FROM MEASUREMENT TO NATIVE RELIABILITY", size=42, weight=800),
        text(900, 106, "Uncertainty becomes progressively cheaper and more deeply integrated", size=25, fill=MUTED),
    ]
    x0, gap, w, h, y = 70, 105, 485, 385, 170
    for i, (stage, method, timing, lines, color, fill) in enumerate(items):
        x = x0 + i * (w + gap)
        parts.append(rounded_rect(x, y, w, h, fill="white", stroke=color, width=4, radius=24))
        parts.append(f'<path d="M {x+24} {y} H {x+w-24} Q {x+w} {y} {x+w} {y+24} V {y+92} H {x} V {y+24} Q {x} {y} {x+24} {y}" fill="{fill}"/>')
        parts.append(text(x + w / 2, y + 61, stage, size=30, weight=800, fill=color))
        parts.append(text(x + w / 2, y + 157, method, size=38, weight=800, fill=color))
        parts.append(text(x + w / 2, y + 197, timing, size=21, weight=600, fill=MUTED))
        parts.append(multiline(x + w / 2, y + 270, lines, size=22, gap=38))
        if i < 2:
            ax = x + w + 25
            parts.append(f'<path d="M {ax} {y+190} L {ax+55} {y+190}" stroke="{BLUE}" stroke-width="8" fill="none" marker-end="url(#arrow-blue)"/>')
    parts.extend([
        text(900, 606, "runtime control  →  calibrated prediction  →  calibrated policy", size=25, weight=700, fill=MUTED),
    ])
    write("slide_measure_align_internalize.svg", svg_doc(width, height, "\n".join(parts),
          title="Measure, align, and internalize uncertainty",
          desc="AUQ measures trajectory uncertainty, ACC and HTC align confidence with success, and CaOPD internalizes calibration during training."))


if __name__ == "__main__":
    reliability_loop()
    reliability_profile()
    horizon_plot()
    reliability_diagram()
    measure_align_internalize()
    print(f"Wrote five SVG figures to {OUT}")
