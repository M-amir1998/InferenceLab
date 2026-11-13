"""Benchmark plots (static PNGs for markdown reports).

Design rules (validated categorical palette, light surface): colour follows the
*entity* (a backend/config keeps its hue across every chart of a report), fixed slot
order, never more than 8 series, one y-scale per chart (no dual axes — related measures
become small multiples), 2 px lines, 8 px markers with a surface ring, hairline solid
grids, text in ink colours rather than series colours, direct end-labels for <= 4
series plus a legend for >= 2. Every plot is accompanied by a table in the report, which
is the relief for the three palette slots below 3:1 contrast.

matplotlib is optional (``pip install 'inferencelab[plots]'``); without it the functions
return ``None`` and reports omit the images.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SERIES_COLORS",
    "assign_colors",
    "latency_cdf",
    "plots_available",
    "small_multiples_timeline",
    "sweep",
    "timeline",
]

SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
AXIS = "#d3d1cb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
DPI = 144  # 1 pt = 2 px
LINE_W = 1.0  # 2 px
MARKER = 4.0  # 8 px diameter
RING = 1.0  # 2 px surface ring


def plots_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def assign_colors(names: Sequence[str]) -> dict[str, str]:
    """Stable entity → colour mapping (sorted names, fixed slot order, max 8)."""
    unique = sorted(set(names))
    if len(unique) > len(SERIES_COLORS):
        raise ValueError("more than 8 series: fold the tail into 'Other' or use small multiples")
    return {name: SERIES_COLORS[i] for i, name in enumerate(unique)}


def _setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.5,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": TEXT_PRIMARY,
            "font.family": "DejaVu Sans",
        }
    )
    return plt


def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.5)


def _finish(fig: Any, ax: Any, path: Path, *, n_series: int, title: str, note: str | None) -> Path:
    ax.set_title(title)
    if n_series >= 2:
        ax.legend(loc="best")
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=TEXT_SECONDARY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _end_label(ax: Any, x: float, y: float, text: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=TEXT_PRIMARY,
        va="center",
    )


def sweep(
    series: Mapping[str, tuple[Sequence[float], Sequence[float | None]]],
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: Mapping[str, str] | None = None,
    note: str | None = None,
    log_x: bool = False,
) -> Path | None:
    """One line per series (e.g. backend): y vs x (concurrency, request rate, goodput)."""
    if not plots_available() or not series:
        return None
    plt = _setup()
    colors = colors or assign_colors(list(series))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style_axes(ax)
    direct = len(series) <= 4
    for name, (xs, ys) in series.items():
        pts = sorted((x, y) for x, y in zip(xs, ys, strict=True) if y is not None)
        if not pts:
            continue
        px, py = zip(*pts, strict=True)
        ax.plot(
            px,
            py,
            color=colors[name],
            linewidth=LINE_W,
            marker="o",
            markersize=MARKER,
            markeredgecolor=SURFACE,
            markeredgewidth=RING,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=name,
        )
        if direct:
            _end_label(ax, px[-1], py[-1], name)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    return _finish(fig, ax, path, n_series=len(series), title=title, note=note)


def latency_cdf(
    series: Mapping[str, Sequence[float]],
    *,
    path: Path,
    title: str,
    xlabel: str,
    colors: Mapping[str, str] | None = None,
    note: str | None = None,
) -> Path | None:
    """Empirical CDF per series — shows the whole distribution, not just percentiles."""
    if not plots_available():
        return None
    data = {k: sorted(v) for k, v in series.items() if v}
    if not data:
        return None
    plt = _setup()
    colors = colors or assign_colors(list(data))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style_axes(ax)
    for name, values in data.items():
        n = len(values)
        ys = [(i + 1) / n for i in range(n)]
        ax.step(values, ys, where="post", color=colors[name], linewidth=LINE_W, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    return _finish(fig, ax, path, n_series=len(data), title=title, note=note)


def timeline(
    t: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    path: Path,
    title: str,
    ylabel: str,
    colors: Mapping[str, str] | None = None,
    note: str | None = None,
    area: bool = True,
) -> Path | None:
    """Value over time (queue depth, in-flight requests). Area = 10 % wash under a 2 px line."""
    if not plots_available() or not t or not series:
        return None
    plt = _setup()
    colors = colors or assign_colors(list(series))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    _style_axes(ax)
    for name, values in series.items():
        ax.plot(t, values, color=colors[name], linewidth=LINE_W, label=name, drawstyle="steps-post")
        if area:
            ax.fill_between(t, values, step="post", color=colors[name], alpha=0.10, linewidth=0)
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    return _finish(fig, ax, path, n_series=len(series), title=title, note=note)


def small_multiples_timeline(
    t: Sequence[float],
    panels: Sequence[tuple[str, Mapping[str, Sequence[float]]]],
    *,
    path: Path,
    title: str,
    note: str | None = None,
) -> Path | None:
    """Stacked panels sharing the time axis — each panel has its own single y-scale."""
    if not plots_available() or not t or not panels:
        return None
    plt = _setup()
    names = [n for _, s in panels for n in s]
    colors = assign_colors(names)
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 1.9 * len(panels)), sharex=True)
    axes = list(axes) if len(panels) > 1 else [axes]
    for ax, (ylabel, series) in zip(axes, panels, strict=True):
        _style_axes(ax)
        for name, values in series.items():
            ax.plot(
                t, values, color=colors[name], linewidth=LINE_W, label=name, drawstyle="steps-post"
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if len(series) >= 2:
            ax.legend(loc="upper left")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=TEXT_SECONDARY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
