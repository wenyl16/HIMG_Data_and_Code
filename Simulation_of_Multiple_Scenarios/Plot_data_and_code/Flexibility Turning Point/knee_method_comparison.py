"""Single-panel comparison of flexibility-knee disagreement among methods.

For every scenario-target combination, each method is expressed as a signed
deviation from the three-method mean.  This removes the large change in the
absolute knee location across targets and makes method disagreement directly
comparable on one common axis.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
CLEAN_DIR = RESULT_DIR / "clean"
INPUT_CSV = RESULT_DIR / "domestic_flexibility_knee_method_differences.csv"
OUTPUT_SVG = RESULT_DIR / "knee_method_deviations.svg"
CLEAN_SVG = CLEAN_DIR / "knee_method_deviations.svg"
OUTPUT_CSV = RESULT_DIR / "knee_method_deviations_plot_data.csv"

SCENARIOS = ["Rural", "Commercial", "Industrial", "Island"]
SCENARIO_LABELS = {
    "Rural": "Rural",
    "Commercial": "Commercial",
    "Industrial": "Industrial",
    "Island": "Island",
}
TARGETS = [60, 70, 80, 90, 100]
METHODS = ["maximum_curvature", "kneedle", "L_method"]
# Draw L_method before Kneedle so coincident triangles remain on top of squares.
DRAW_METHODS = ["maximum_curvature", "L_method", "kneedle"]
METHOD_LABELS = {
    "maximum_curvature": "Maximum curvature",
    "kneedle": "Kneedle",
    "L_method": "L_method",
}
METHOD_STYLES = {
    "maximum_curvature": {"marker": "o", "color": "#D9822B", "offset": 0.00, "size": 24},
    "kneedle": {"marker": "^", "color": "#377EB8", "offset": 0.00, "size": 34},
    "L_method": {"marker": "s", "color": "#6B7280", "offset": 0.00, "size": 20},
}
METHOD_COLUMNS = {
    method: f"{method}_penetration_%" for method in METHODS
}
SPAN_COLUMN = "three_method_max_span_pp"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Microsoft YaHei", "SimHei", "Arial", "Helvetica",
        "DejaVu Sans", "sans-serif",
    ],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})


def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {
        "scenario", "consumption_target_%", SPAN_COLUMN, *METHOD_COLUMNS.values(),
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[df["scenario"].isin(SCENARIOS)].copy()
    df["scenario"] = pd.Categorical(df["scenario"], SCENARIOS, ordered=True)
    df = df.sort_values(["scenario", "consumption_target_%"]).reset_index(drop=True)
    if len(df) != len(SCENARIOS) * len(TARGETS):
        raise ValueError(f"Expected 20 rows, found {len(df)}")
    value_columns = [METHOD_COLUMNS[method] for method in METHODS]
    df["three_method_mean_penetration_%"] = df[value_columns].mean(axis=1)
    deviation_columns = []
    for method in METHODS:
        column = f"{method}_deviation_from_three_method_mean_pp"
        df[column] = df[METHOD_COLUMNS[method]] - df["three_method_mean_penetration_%"]
        deviation_columns.append(column)

    ordered = [
        "scenario", "consumption_target_%",
        *[METHOD_COLUMNS[method] for method in METHODS],
        "three_method_mean_penetration_%",
        *deviation_columns,
        "maximum_curvature_vs_kneedle_abs_pp",
        "maximum_curvature_vs_L_method_abs_pp",
        "kneedle_vs_L_method_abs_pp",
        SPAN_COLUMN,
    ]
    return df[ordered]


def draw_figure(df, labeled=True):
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    group_gap = 1.15
    group_size = len(TARGETS)
    all_y = []
    all_target_labels = []
    band_colors = ["#F6F8FA", "#FFFFFF", "#F6F8FA", "#FFFFFF"]

    for group_index, scenario in enumerate(SCENARIOS):
        sub = (
            df[df["scenario"] == scenario]
            .sort_values("consumption_target_%")
            .reset_index(drop=True)
        )
        start = group_index * (group_size + group_gap)
        y_base = start + np.arange(len(sub), dtype=float)
        all_y.extend(y_base)
        all_target_labels.extend([f"{target}%" for target in sub["consumption_target_%"]])

        deviation_columns = [
            f"{method}_deviation_from_three_method_mean_pp" for method in METHODS
        ]
        values = sub[deviation_columns].to_numpy(float)
        xmin = np.nanmin(values, axis=1)
        xmax = np.nanmax(values, axis=1)

        ax.axhspan(
            start - 0.48, start + group_size - 0.52,
            color=band_colors[group_index], zorder=-3,
        )

        for y, lo, hi in zip(y_base, xmin, xmax):
            ax.plot(
                [lo, hi], [y, y], color="#B7BEC7", linewidth=2.2,
                solid_capstyle="round", zorder=1,
            )

        for method in DRAW_METHODS:
            style = METHOD_STYLES[method]
            ax.scatter(
                sub[f"{method}_deviation_from_three_method_mean_pp"],
                y_base + style["offset"],
                s=style["size"],
                marker=style["marker"],
                facecolor="white",
                edgecolor=style["color"],
                linewidth=1.15,
                zorder=4,
            )

        if labeled:
            header_y = start - 0.70
            ax.text(
                -3.85, header_y, SCENARIO_LABELS[scenario],
                fontsize=7.3, fontweight="bold", ha="left", va="center",
            )

    if labeled:
        ax.text(
            3.85, -0.70,
            f"Overall mean span  {df[SPAN_COLUMN].mean():.1f} pp"
            f"   |   Max span  {df[SPAN_COLUMN].max():.1f} pp",
            fontsize=5.8, color="#525A65", ha="right", va="center",
        )

    lower_limit = (len(SCENARIOS) - 1) * (group_size + group_gap) + group_size - 0.45
    ax.set_xlim(-4, 4)
    ax.set_ylim(lower_limit, -1.15)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(all_y)
    ax.axvline(0, color="#747B84", linewidth=0.8, zorder=0)
    ax.grid(axis="x", color="#E3E7EB", linewidth=0.55, zorder=-1)
    ax.tick_params(axis="both", length=2.5, width=0.6, labelsize=6)

    if labeled:
        ax.set_yticklabels(all_target_labels)
        ax.set_xlabel(
            "Signed deviation from the three-method mean (percentage points)",
            fontsize=7,
        )
        ax.set_ylabel("Self-consumption target", fontsize=7)
        ax.text(
            0.50, 1.005, "← lower knee estimate     higher knee estimate →",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5.8, color="#525A65",
        )

        handles = [
            Line2D(
                [0], [0], linestyle="none",
                marker=METHOD_STYLES[method]["marker"], markersize=5,
                markerfacecolor="white",
                markeredgecolor=METHOD_STYLES[method]["color"],
                markeredgewidth=1.15,
                label=METHOD_LABELS[method],
            )
            for method in METHODS
        ]
        handles.append(Line2D(
            [0], [0], color="#B7BEC7", linewidth=2.2,
            label="Three-method span",
        ))
        fig.legend(
            handles=handles, loc="upper center", ncol=4, fontsize=6.2,
            bbox_to_anchor=(0.5, 0.995), handletextpad=0.5,
            columnspacing=1.4,
        )
        fig.subplots_adjust(
            left=0.15, right=0.97, bottom=0.10, top=0.92,
        )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.subplots_adjust(
            left=0.035, right=0.99, bottom=0.03, top=0.99,
        )
    return fig


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    CLEAN_DIR.mkdir(exist_ok=True)
    df = load_data()
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.4f")

    fig = draw_figure(df, labeled=True)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)

    clean = draw_figure(df, labeled=False)
    clean.savefig(CLEAN_SVG)
    plt.close(clean)

    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {CLEAN_SVG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
