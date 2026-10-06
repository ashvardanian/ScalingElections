"""Renders one measurement chart as a light and dark SVG pair.

Every number arrives on the command line, so the README tables stay the only place the
measurements live and a chart cannot quietly disagree with the table beside it.

    python assets/charts.py schulze \
        --title "Schulze Strongest Paths" \
        --subtitle "Widest paths over the max-min semiring. Both axes logarithmic, lower is better." \
        --alt "Schulze wall clock against field size, log-log" \
        --x-title "alternatives ranked" --y-title "wall clock" \
        --columns "1 K,4 K,16 K,64 K" \
        --y-ticks "10 ms=0.01,100 ms=0.1,1 s=1,10 s=10,100 s=100" \
        --series "132× SM90=0.010,0.088,2.53,59.39" \
        --series "16× Xeon4=0.026,1.55,96.55,-"

A dash in a series marks a point that was not measured, and the line skips it.
"""

import argparse
import math
import pathlib

WIDTH, HEIGHT = 934, 500
PLOT_LEFT, PLOT_RIGHT = 96, 894
FIRST_POINT, LAST_POINT = 125.6, 864.4
BASELINE_Y, PIXELS_PER_DECADE = 404.0, 44.0
SANS = "ui-sans-serif,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#9861d6",
    "#c64f68",
    "#8c7853",
)

LEGEND_FONT_SIZE = 12.5
LEGEND_SWATCH = 11.0
LEGEND_GAP = 26.0

# Advance widths for the sans stack at one point, enough to lay a legend out without a font engine.
_NARROW = set(" ijlt.,;:'!|")
_WIDE = set("mwMW@")


def text_width(text: str, font_size: float) -> float:
    """Approximates a rendered label's width, so entries are spaced by content rather than by count."""
    units = sum(0.34 if character in _NARROW else 0.86 if character in _WIDE else 0.55 for character in text)
    return units * font_size


THEMES = {
    "light": {
        "page": "#fcfcfb",
        "title": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e7e6e2",
        "tick": "#8a8984",
    },
    "dark": {
        "page": "#1a1a19",
        "title": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#2e2e2c",
        "tick": "#8a8980",
    },
}


def labelled_pairs(text: str) -> list[tuple[str, float]]:
    """Parses `label=value,label=value` into pairs, keeping the order given."""
    pairs = []
    for entry in text.split(","):
        label, _, value = entry.rpartition("=")
        pairs.append((label.strip(), float(value)))
    return pairs


def series_argument(text: str) -> tuple[str, list[float | None]]:
    """Parses `label=v1,v2,-` into a name and its points, a dash meaning unmeasured."""
    label, _, values = text.partition("=")
    return label.strip(), [None if v.strip() == "-" else float(v) for v in values.split(",")]


def render(options: argparse.Namespace, theme: str) -> str:
    """Render the supplied measurements and labels as a themed SVG chart."""
    palette = THEMES[theme]
    ticks = labelled_pairs(options.y_ticks)
    floor = min(value for _, value in ticks)
    columns = options.columns.split(",")
    spacing = (LAST_POINT - FIRST_POINT) / (len(columns) - 1)
    names = [series_argument(entry)[0] for entry in options.series]
    legend_rows: list[list[tuple[int, str, float]]] = [[]]
    row_width = 0.0
    for order, name in enumerate(names):
        width = LEGEND_SWATCH + text_width(name, LEGEND_FONT_SIZE)
        if width > PLOT_RIGHT - PLOT_LEFT:
            raise SystemExit(f"legend entry is too wide: {name}")
        if legend_rows[-1] and (
            row_width + LEGEND_GAP + width > PLOT_RIGHT - PLOT_LEFT
            or (name.startswith("Pref-Voting") and not legend_rows[-1][-1][1].startswith("Pref-Voting"))
        ):
            legend_rows.append([])
            row_width = 0.0
        row_width += (LEGEND_GAP if legend_rows[-1] else 0.0) + width
        legend_rows[-1].append((order, name, width))
    height = HEIGHT + 22 * (len(legend_rows) - 1)

    def y_of(value: float) -> float:
        """Map a positive measurement onto the logarithmic vertical axis."""
        return BASELINE_Y - (math.log10(value) - math.log10(floor)) * PIXELS_PER_DECADE

    def x_of(index: int) -> float:
        """Place a measurement column on the horizontal axis."""
        return FIRST_POINT + index * spacing

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="{SANS}" role="img" aria-label="{options.alt}">',
        f'<rect width="{WIDTH}" height="{height}" rx="14" fill="{palette["page"]}"/>',
        f'<text x="96" y="40" font-size="21" font-weight="650" fill="{palette["title"]}">{options.title}</text>',
        f'<text x="96" y="62" font-size="13" fill="{palette["muted"]}">{options.subtitle}</text>',
    ]

    for label, value in ticks:
        y = y_of(value)
        parts.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}" '
            f'stroke="{palette["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="84" y="{y + 4:.1f}" font-size="11.5" text-anchor="end" '
            f'fill="{palette["tick"]}" font-family="{MONO}">{label}</text>'
        )

    for index, label in enumerate(columns):
        parts.append(
            f'<text x="{x_of(index):.1f}" y="424" font-size="11.5" text-anchor="middle" '
            f'fill="{palette["tick"]}" font-family="{MONO}">{label.strip()}</text>'
        )

    parts.append(
        f'<text x="495" y="448" font-size="12" text-anchor="middle" fill="{palette["muted"]}">{options.x_title}</text>'
    )
    parts.append(
        f'<text x="20" y="250" font-size="12" text-anchor="middle" fill="{palette["muted"]}" '
        f'transform="rotate(-90 20 250)">{options.y_title}</text>'
    )

    for order, entry in enumerate(options.series):
        name, values = series_argument(entry)
        color = SERIES_COLORS[order % len(SERIES_COLORS)]
        points = [(x_of(i), y_of(v)) for i, v in enumerate(values) if v is not None]
        path = " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<path d="M{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in points:
            # The halo takes the page colour, so it follows the theme rather than staying light.
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{palette["page"]}" stroke-width="2"/>'
            )

    for row_index, row in enumerate(legend_rows):
        width = sum(entry[2] for entry in row) + LEGEND_GAP * (len(row) - 1)
        legend_x = PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT - width) / 2
        legend_y = 472 + row_index * 22
        for order, name, width in row:
            parts.append(
                f'<circle cx="{legend_x:.1f}" cy="{legend_y}" r="4" '
                f'fill="{SERIES_COLORS[order % len(SERIES_COLORS)]}"/>'
            )
            parts.append(
                f'<text x="{legend_x + LEGEND_SWATCH:.1f}" y="{legend_y + 4}" font-size="{LEGEND_FONT_SIZE}" '
                f'fill="{palette["muted"]}">{name}</text>'
            )
            legend_x += width + LEGEND_GAP

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write light and dark charts from command-line measurements."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "name",
        help="Basename for the pair, written as <name>-light.svg and <name>-dark.svg",
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--alt", required=True, help="Accessible description of the whole chart")
    parser.add_argument("--x-title", required=True)
    parser.add_argument("--y-title", required=True)
    parser.add_argument(
        "--columns",
        required=True,
        help="Comma-separated tick labels along the horizontal axis",
    )
    parser.add_argument(
        "--y-ticks",
        required=True,
        help="Comma-separated `label=value` pairs, log scale",
    )
    parser.add_argument(
        "--series",
        required=True,
        action="append",
        help="`label=v1,v2,...` with a dash for an unmeasured point; repeat for more series",
    )
    options = parser.parse_args()

    here = pathlib.Path(__file__).parent
    for theme in THEMES:
        target = here / f"{options.name}-{theme}.svg"
        target.write_text(render(options, theme))
        print(f"  wrote {target.relative_to(here.parent)}")


if __name__ == "__main__":
    main()
