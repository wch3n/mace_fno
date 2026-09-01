#!/usr/bin/env python3
"""Render the frozen-MACE residual-FNO workflow used in the report.

The labels follow the implemented path in mace_fno.coupling.MACEFNOResidual:
frozen but differentiable MACE descriptors, a neutral latent source head,
particle--mesh deposition, a geometry-aware field operator, a scalar residual
energy, and forces obtained by differentiating the summed energy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


MM_TO_INCH = 1.0 / 25.4
COLORS = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "line": "#9AA5B1",
    "mace": "#2F6690",
    "mace_fill": "#E8F1F8",
    "fno": "#C44E52",
    "fno_fill": "#FBEDEE",
    "total": "#4C956C",
    "total_fill": "#EAF5EF",
    "neutral_fill": "#F5F7FA",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rectangle_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str = "",
    *,
    facecolor: str,
    edgecolor: str,
    title_color: str | None = None,
    linewidth: float = 1.15,
) -> Rectangle:
    cx, cy = center
    width, height = size
    patch = Rectangle(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    multiline_title = "\n" in title
    multiline_subtitle = "\n" in subtitle
    if multiline_title:
        title_y = cy + 0.052
        subtitle_y = cy - 0.052
    elif multiline_subtitle:
        title_y = cy + 0.050
        subtitle_y = cy - 0.040
    else:
        title_y = cy + 0.028
        subtitle_y = cy - 0.035
    ax.text(
        cx,
        title_y if subtitle else cy,
        title,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=title_color or COLORS["ink"],
        zorder=4,
    )
    if subtitle:
        ax.text(
            cx,
            subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.4,
            color=COLORS["muted"],
            linespacing=1.10,
            zorder=4,
        )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linewidth: float = 1.25,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=linewidth,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=2,
        )
    )


def elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    corner: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linewidth: float = 1.25,
) -> None:
    """Draw a two-segment orthogonal connector with one terminal arrowhead."""
    ax.plot(
        [start[0], corner[0]],
        [start[1], corner[1]],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=2,
    )
    arrow(ax, corner, end, color=color, linewidth=linewidth)


def build_figure() -> plt.Figure:
    configure_style()
    width_mm = 183
    height_mm = 78
    fig, ax = plt.subplots(figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH))
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.035, top=0.985)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.535,
        0.475,
        "learned residual branch",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["fno"],
        zorder=1,
    )

    rectangle_box(
        ax,
        (0.075, 0.735),
        (0.130, 0.300),
        "Atomic\nstructure",
        r"$\{\mathbf{R},\mathbf{H}\}$",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["line"],
    )
    rectangle_box(
        ax,
        (0.245, 0.735),
        (0.170, 0.300),
        "Frozen MACE",
        "weights fixed\ncoordinate gradients\nretained",
        facecolor=COLORS["mace_fill"],
        edgecolor=COLORS["mace"],
        title_color=COLORS["mace"],
    )
    rectangle_box(
        ax,
        (0.450, 0.735),
        (0.170, 0.300),
        "Local energy",
        r"$E_{\mathrm{MACE}}$"
        "\nshort-range baseline",
        facecolor=COLORS["mace_fill"],
        edgecolor=COLORS["mace"],
        title_color=COLORS["mace"],
    )
    rectangle_box(
        ax,
        (0.340, 0.270),
        (0.130, 0.300),
        "Descriptors\n" r"$\mathbf{h}_i$",
        "local chemistry",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["line"],
    )
    rectangle_box(
        ax,
        (0.505, 0.270),
        (0.180, 0.300),
        "Source head\n+ deposition",
        r"$g_\psi(\mathbf{h}_i)\rightarrow s_{ic}$"
        "\nneutral latent channels\n"
        r"B spline $\rightarrow \rho_c(\mathbf{r})$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )
    rectangle_box(
        ax,
        (0.685, 0.270),
        (0.145, 0.300),
        "Geometry-\naware FNO",
        "2D FNO / 3D FNO\n"
        r"$\rho_c(\mathbf{r})\rightarrow\phi_c(\mathbf{r})$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )
    rectangle_box(
        ax,
        (0.840, 0.270),
        (0.140, 0.300),
        "Residual\nenergy",
        r"$\Delta E_{\mathrm{FNO}}$"
        "\n" r"$0.5\int\boldsymbol{\rho}\cdot\boldsymbol{\phi}\,\mathrm{d}\mathbf{r}$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )

    rectangle_box(
        ax,
        (0.860, 0.735),
        (0.240, 0.300),
        "Total energy and\nconservative forces",
        r"$E=E_{\mathrm{MACE}}+\Delta E_{\mathrm{FNO}}$"
        "\n"
        r"$\mathbf{F}_i=-\partial E/\partial\mathbf{r}_i$",
        facecolor=COLORS["total_fill"],
        edgecolor=COLORS["total"],
        title_color=COLORS["total"],
    )

    arrow(ax, (0.140, 0.735), (0.160, 0.735))
    arrow(ax, (0.330, 0.735), (0.365, 0.735), color=COLORS["mace"])
    arrow(ax, (0.535, 0.735), (0.740, 0.735), color=COLORS["mace"])
    elbow_arrow(
        ax,
        (0.245, 0.585),
        (0.245, 0.270),
        (0.275, 0.270),
        color=COLORS["line"],
    )
    arrow(ax, (0.405, 0.270), (0.415, 0.270), color=COLORS["fno"])
    arrow(ax, (0.595, 0.270), (0.6125, 0.270), color=COLORS["fno"])
    arrow(ax, (0.7575, 0.270), (0.770, 0.270), color=COLORS["fno"])
    arrow(ax, (0.840, 0.420), (0.840, 0.585), color=COLORS["fno"])

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="External directory for generated PDF/SVG/TIFF/PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "fno_workflow"
    fig = build_figure()
    pdf_metadata = {
        "Title": "Frozen-MACE residual-FNO workflow",
        "Author": "Wei Chen",
        "Subject": "Information flow in the MACE-FNO architecture",
    }
    svg_metadata = {
        "Title": "Frozen-MACE residual-FNO workflow",
        "Creator": "Wei Chen",
        "Description": "Information flow in the MACE-FNO architecture",
    }
    fig.savefig(stem.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(stem.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
