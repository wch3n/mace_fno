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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


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


def rounded_box(
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
) -> FancyBboxPatch:
    cx, cy = center
    width, height = size
    patch = FancyBboxPatch(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy + (0.020 if subtitle else 0.0),
        title,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=title_color or COLORS["ink"],
        zorder=4,
    )
    if subtitle:
        ax.text(
            cx,
            cy - 0.031,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.2,
            color=COLORS["muted"],
            linespacing=1.15,
            zorder=4,
        )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    connectionstyle: str = "arc3,rad=0.0",
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
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=2,
        )
    )


def build_figure() -> plt.Figure:
    configure_style()
    width_mm = 183
    height_mm = 105
    fig, ax = plt.subplots(figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH))
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.035, top=0.985)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # A light enclosure identifies the parameters optimized during residual training.
    learned_region = FancyBboxPatch(
        (0.285, 0.205),
        0.695,
        0.425,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        edgecolor="#D8A0A3",
        facecolor="#FFF9F9",
        zorder=0,
    )
    ax.add_patch(learned_region)
    ax.text(
        0.695,
        0.605,
        "learned residual branch",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["fno"],
        zorder=1,
    )

    rounded_box(
        ax,
        (0.50, 0.915),
        (0.245, 0.085),
        r"Atomic structure  $\{\mathbf{R},\mathbf{H}\}$",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["line"],
    )
    rounded_box(
        ax,
        (0.50, 0.755),
        (0.335, 0.170),
        "Frozen MACE (weights fixed)",
        "coordinate gradients retained",
        facecolor=COLORS["mace_fill"],
        edgecolor=COLORS["mace"],
        title_color=COLORS["mace"],
    )
    rounded_box(
        ax,
        (0.145, 0.445),
        (0.225, 0.105),
        r"Local energy  $E_{\mathrm{MACE}}$",
        "short-range baseline",
        facecolor=COLORS["mace_fill"],
        edgecolor=COLORS["mace"],
        title_color=COLORS["mace"],
    )
    rounded_box(
        ax,
        (0.365, 0.455),
        (0.155, 0.105),
        r"Descriptors  $\mathbf{h}_i$",
        "local chemistry",
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["line"],
    )
    rounded_box(
        ax,
        (0.585, 0.455),
        (0.245, 0.145),
        "Source head + deposition",
        r"$g_\psi(\mathbf{h}_i)\rightarrow$ neutral $s_{ic}$"
        "\n"
        r"B spline $\rightarrow$ latent $\rho_c(\mathbf{r})$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )
    rounded_box(
        ax,
        (0.855, 0.455),
        (0.205, 0.145),
        "Geometry-aware FNO",
        "2.5D slab / 3D bulk\n"
        r"$\rho_c(\mathbf{r})\rightarrow\phi_c(\mathbf{r})$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )
    rounded_box(
        ax,
        (0.815, 0.275),
        (0.285, 0.100),
        r"Residual energy  $\Delta E_{\mathrm{FNO}}$",
        r"$0.5\int\boldsymbol{\rho}\cdot\boldsymbol{\phi}\,\mathrm{d}\mathbf{r}$",
        facecolor=COLORS["fno_fill"],
        edgecolor=COLORS["fno"],
        title_color=COLORS["fno"],
    )

    merge_center = (0.50, 0.220)
    merge = Circle(
        merge_center,
        radius=0.025,
        facecolor=COLORS["total"],
        edgecolor=COLORS["total"],
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(merge)
    rounded_box(
        ax,
        (0.50, 0.085),
        (0.365, 0.115),
        r"Total energy and conservative forces",
        r"$E=E_{\mathrm{MACE}}+\Delta E_{\mathrm{FNO}},\quad"
        r"\mathbf{F}_i=-\partial E/\partial\mathbf{r}_i$",
        facecolor=COLORS["total_fill"],
        edgecolor=COLORS["total"],
        title_color=COLORS["total"],
    )

    arrow(ax, (0.50, 0.870), (0.50, 0.835))
    arrow(
        ax,
        (0.39, 0.665),
        (0.17, 0.505),
        color=COLORS["mace"],
        connectionstyle="arc3,rad=0.16",
    )
    arrow(ax, (0.50, 0.665), (0.39, 0.515), color=COLORS["line"])
    arrow(ax, (0.443, 0.455), (0.460, 0.455), color=COLORS["fno"])
    arrow(ax, (0.708, 0.455), (0.752, 0.455), color=COLORS["fno"])
    arrow(
        ax,
        (0.855, 0.382),
        (0.825, 0.327),
        color=COLORS["fno"],
        connectionstyle="arc3,rad=0.10",
    )
    arrow(
        ax,
        (0.145, 0.391),
        (0.452, 0.220),
        color=COLORS["mace"],
        connectionstyle="arc3,rad=-0.10",
    )
    arrow(
        ax,
        (0.672, 0.275),
        (0.548, 0.220),
        color=COLORS["fno"],
        connectionstyle="arc3,rad=0.08",
    )
    arrow(ax, (0.50, 0.195), (0.50, 0.143), color=COLORS["total"])

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
