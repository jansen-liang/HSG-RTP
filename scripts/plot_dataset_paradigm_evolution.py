from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures"


@dataclass(frozen=True)
class DatasetPoint:
    name: str
    x: float
    y: float
    category: str
    label_offset: tuple[float, float]


CATEGORY_STYLE = {
    "language_action": {"color": "#3977B8", "marker": "o", "label": "Language--action"},
    "state_rich": {"color": "#E08B2C", "marker": "s", "label": "State-rich interaction"},
    "structured": {"color": "#3A9B72", "marker": "D", "label": "Structured-environment planning"},
}


DATASETS = [
    DatasetPoint("VirtualHome", 1.00, 2.35, "language_action", (-0.58, 0.15)),
    DatasetPoint("ALFRED", 1.25, 2.75, "language_action", (-0.55, 0.17)),
    DatasetPoint("TEACh", 1.48, 2.55, "language_action", (0.10, -0.25)),
    DatasetPoint("CALVIN", 1.72, 1.18, "language_action", (-0.55, 0.13)),
    DatasetPoint("ReALFRED", 1.72, 3.02, "language_action", (0.10, 0.27)),
    DatasetPoint("ARNOLD", 2.15, 2.22, "state_rich", (-0.60, -0.25)),
    DatasetPoint("BEHAVIOR-1K", 2.45, 2.88, "state_rich", (-0.78, 0.30)),
    DatasetPoint("PARTNR", 3.12, 3.25, "state_rich", (0.10, 0.14)),
    DatasetPoint("TASKOGRAPHY", 3.35, 4.05, "structured", (-0.98, 0.15)),
    DatasetPoint("GRID", 3.28, 2.78, "structured", (0.12, -0.24)),
    DatasetPoint("MomaGraph", 4.10, 2.98, "structured", (0.12, 0.14)),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 9.4,
            "axes.titlesize": 10.4,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_stage_backgrounds(ax: plt.Axes) -> None:
    bands = [
        (0.55, 1.55, "#EDF4FB", "Demonstration-centric"),
        (1.55, 2.60, "#FFF5E8", "Goal/state-centric"),
        (2.60, 3.75, "#ECF7F2", "Structured world models"),
        (3.75, 4.85, "#F5EEFA", "Closed-loop recovery"),
    ]
    for left, right, color, label in bands:
        ax.axvspan(left, right, color=color, alpha=0.72, zorder=0)
        ax.text(
            (left + right) / 2,
            5.13,
            label,
            ha="center",
            va="bottom",
            color="#50545A",
            fontsize=7.6,
            fontweight="semibold",
        )


def add_underexplored_region(ax: plt.Axes) -> None:
    region = Ellipse(
        (4.22, 4.58),
        width=1.16,
        height=0.92,
        facecolor="#C8B1DC",
        edgecolor="#76518E",
        linewidth=1.35,
        linestyle=(0, (4, 2)),
        alpha=0.28,
        zorder=1,
    )
    ax.add_patch(region)
    ax.text(
        4.22,
        4.76,
        "Comparatively underexplored",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#5A3D6C",
        fontweight="semibold",
        zorder=3,
    )


def add_target_hlr(ax: plt.Axes) -> None:
    box = FancyBboxPatch(
        (3.92, 4.35),
        0.58,
        0.28,
        boxstyle="round,pad=0.03,rounding_size=0.035",
        facecolor="#EAF3FF",
        edgecolor="#2D67A3",
        linewidth=1.25,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        4.23,
        4.49,
        "Target HSG-RTP",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#245889",
        fontweight="bold",
        zorder=5,
    )


def draw_points(ax: plt.Axes) -> None:
    for point in DATASETS:
        style = CATEGORY_STYLE[point.category]
        ax.scatter(
            point.x,
            point.y,
            s=48,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        label_x = point.x + point.label_offset[0]
        label_y = point.y + point.label_offset[1]
        ax.annotate(
            point.name,
            xy=(point.x, point.y),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7.6,
            color="#25282C",
            arrowprops={
                "arrowstyle": "-",
                "color": "#8B9096",
                "linewidth": 0.55,
                "shrinkA": 1.5,
                "shrinkB": 3.5,
            },
            zorder=5,
        )


def configure_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.55, 4.85)
    ax.set_ylim(0.78, 5.10)
    ax.set_xticks([1.05, 2.08, 3.18, 4.30])
    ax.set_xticklabels(
        [
            "Action\ndemonstrations",
            "Goal/state\nspecification",
            "Structured world\nmodeling",
            "Intervention and\nplan recovery",
        ]
    )
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(
        [
            "Tabletop /\nworkspace",
            "Single room",
            "Multi-room\nresidence",
            "Large indoor\nenvironment",
            "Multi-floor\nbuilding",
        ]
    )
    ax.set_xlabel(
        "Supervision paradigm: from action demonstrations to intervention-aware recovery",
        labelpad=8,
    )
    ax.set_ylabel("Spatial planning scale", labelpad=8)
    ax.grid(axis="y", color="#C9CDD2", linewidth=0.6, alpha=0.72)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_color("#8D9298")
        spine.set_linewidth(0.8)


def add_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=6.8,
            label=style["label"],
        )
        for style in CATEGORY_STYLE.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#76518E",
            linewidth=1.3,
            linestyle=(0, (4, 2)),
            label="Underexplored region / target design",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.988),
        frameon=True,
        framealpha=0.94,
        edgecolor="#C8CCD0",
        ncol=1,
        columnspacing=1.1,
        handletextpad=0.45,
        borderpad=0.55,
    )


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.15, 4.65))
    add_stage_backgrounds(ax)
    configure_axes(ax)
    add_underexplored_region(ax)
    draw_points(ax)
    add_target_hlr(ax)
    add_legend(ax)

    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.15, top=0.90)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            OUTPUT_DIR / f"dataset_paradigm_evolution.{suffix}",
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
