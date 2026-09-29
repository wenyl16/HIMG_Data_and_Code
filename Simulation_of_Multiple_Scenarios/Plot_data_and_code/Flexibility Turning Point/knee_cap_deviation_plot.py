"""Visualize three-method cap sensitivity relative to the 200 h baseline.

Reads the aggregated output from ``knee_cap_sensitivity_analysis.py`` and
exports one integrated 2x2 scenario figure in editable SVG format.  All
deviations are penetration-rate percentage points.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results"
SOURCE_FILE = RESULT_DIR / "cap_sensitivity_all_results.csv"

SCENARIOS = ["乡村", "商业", "工业", "海岛"]
METHODS = ["曲率最大", "kneedle", "L_method"]
METHOD_LABELS = {
    "kneedle": "Kneedle",
    "曲率最大": "Maximum curvature",
    "L_method": "L-method",
}
TARGETS = [60, 70, 80, 90, 100]
BASE_CAP = 200
SENSITIVITY_CAPS = [150, 250]
CAP_STYLES = {
    150: {
        "color": "#377EB8", "marker": "o", "linestyle": "--",
        "label": "150 h",
    },
    250: {
        "color": "#E69F00", "marker": "s", "linestyle": "-",
        "label": "250 h",
    },
}
METHOD_STYLES = {
    "kneedle": {"color": "#377EB8"},
    "曲率最大": {"color": "#D9822B"},
    "L_method": {"color": "#6A51A3"},
}
SCENARIO_STYLES = {
    "乡村": {"color": "#4C78A8"},
    "商业": {"color": "#F28E2B"},
    "工业": {"color": "#59A14F"},
    "海岛": {"color": "#B279A2"},
}
VERTICAL_OFFSETS = {150: -0.12, 250: 0.12}
TOLERANCE_BAND = 1.0
ANNOTATION_THRESHOLD = 2.0


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
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})


def load_deviations():
    """Return long-form cap deviations and scenario-method robustness."""
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"Missing sensitivity results: {SOURCE_FILE}. "
            "Run knee_cap_sensitivity_analysis.py first."
        )
    wide = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    required = {"Y轴上限(h)", "场景", "消纳率目标(%)"}
    for method in METHODS:
        required.add(f"{method}_能量渗透率(%)")
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise ValueError(f"Sensitivity table is missing columns: {missing}")

    rows = []
    for method in METHODS:
        value_col = f"{method}_能量渗透率(%)"
        method_data = wide[[
            "Y轴上限(h)", "场景", "消纳率目标(%)", value_col,
        ]].rename(columns={
            "Y轴上限(h)": "cap",
            "场景": "scenario",
            "消纳率目标(%)": "target",
            value_col: "value",
        })
        baseline = (
            method_data[method_data["cap"].eq(BASE_CAP)]
            [["scenario", "target", "value"]]
            .rename(columns={"value": "baseline"})
        )
        sensitivity = method_data[
            method_data["cap"].isin(SENSITIVITY_CAPS)
        ].merge(baseline, on=["scenario", "target"], how="left")
        sensitivity["method"] = method
        sensitivity["deviation"] = (
            sensitivity["value"] - sensitivity["baseline"]
        )
        rows.append(sensitivity)

    long = pd.concat(rows, ignore_index=True)
    long["abs_deviation"] = long["deviation"].abs()
    robustness = (
        long.groupby(["scenario", "method"], as_index=False)
        .agg(
            max_abs_deviation=("abs_deviation", "max"),
            mean_abs_deviation=("abs_deviation", "mean"),
        )
    )
    return long, robustness


def deviation_limit(long):
    """Choose a shared symmetric axis with half-point headroom."""
    maximum = float(long["abs_deviation"].max())
    return max(3.5, np.ceil((maximum + 0.15) * 2.0) / 2.0)


def style_deviation_axis(ax, row, col, x_limit):
    ax.axvspan(
        -TOLERANCE_BAND, TOLERANCE_BAND,
        color="#ECECEC", alpha=0.65, linewidth=0, zorder=0,
    )
    ax.axvline(0, color="#303030", linewidth=0.75, zorder=1)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.5, len(TARGETS) - 0.5)
    ax.set_yticks(np.arange(len(TARGETS)))
    ax.set_yticklabels([f"{target}%" for target in TARGETS], fontsize=6)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.4, linestyle=":")
    ax.tick_params(
        axis="both", labelsize=6, length=2.3, width=0.55, pad=2,
        labelleft=(col == 0), labelbottom=(row == len(SCENARIOS) - 1),
    )


def populate_deviation_grid(axes, long, add_row_labels=True):
    """Populate a 4x3 set of axes with baseline-centered lollipops."""
    x_limit = deviation_limit(long)
    for row, scenario in enumerate(SCENARIOS):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            style_deviation_axis(ax, row, col, x_limit)
            panel = long[
                long["scenario"].eq(scenario)
                & long["method"].eq(method)
            ]
            for target_index, target in enumerate(TARGETS):
                target_data = panel[panel["target"].eq(target)]
                for cap in SENSITIVITY_CAPS:
                    point = target_data[target_data["cap"].eq(cap)]
                    if point.empty:
                        continue
                    delta = float(point["deviation"].iloc[0])
                    y = target_index + VERTICAL_OFFSETS[cap]
                    style = CAP_STYLES[cap]
                    ax.plot(
                        [0, delta], [y, y],
                        color=style["color"], linewidth=0.8, alpha=0.85,
                        solid_capstyle="round", zorder=2,
                    )
                    ax.scatter(
                        delta, y, s=15, marker=style["marker"],
                        facecolor=style["color"], edgecolor="#202020",
                        linewidth=0.45, zorder=3,
                    )
                    if abs(delta) >= ANNOTATION_THRESHOLD:
                        align = "left" if delta >= 0 else "right"
                        offset = 0.10 if delta >= 0 else -0.10
                        ax.text(
                            delta + offset, y, f"{delta:+.2f}",
                            fontsize=5.2, ha=align, va="center",
                            color="#202020", clip_on=False,
                        )
            if row == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=7.5,
                             fontweight="bold", pad=5)
            if col == 0 and add_row_labels:
                ax.text(
                    -0.36, 0.5, scenario,
                    transform=ax.transAxes, rotation=90,
                    fontsize=7.5, fontweight="bold",
                    ha="center", va="center",
                )
    return x_limit


def add_deviation_legend(fig, y_anchor=0.015):
    handles = [
        Line2D(
            [0], [0], color=CAP_STYLES[cap]["color"], linewidth=0.8,
            marker=CAP_STYLES[cap]["marker"], markersize=4,
            markerfacecolor=CAP_STYLES[cap]["color"],
            markeredgecolor="#202020", markeredgewidth=0.45,
            label=CAP_STYLES[cap]["label"],
        )
        for cap in SENSITIVITY_CAPS
    ]
    handles.append(Line2D(
        [0], [0], color="#303030", linewidth=0.8,
        label="200 h baseline (zero)",
    ))
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, y_anchor), fontsize=6,
        handletextpad=0.45, columnspacing=1.2,
    )


def robustness_matrix(robustness):
    matrix = (
        robustness.pivot(
            index="scenario", columns="method", values="max_abs_deviation"
        )
        .reindex(index=SCENARIOS, columns=METHODS)
    )
    return matrix


def draw_heatmap(ax, robustness, add_colorbar=True):
    matrix = robustness_matrix(robustness)
    vmax = max(3.5, float(np.nanmax(matrix.to_numpy())))
    image = ax.imshow(
        matrix.to_numpy(), cmap="Blues", vmin=0, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in METHODS],
        fontsize=6.2, rotation=25, ha="right",
    )
    ax.set_yticks(np.arange(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=6.5)
    ax.tick_params(length=0)
    for i in range(len(SCENARIOS)):
        for j in range(len(METHODS)):
            value = float(matrix.iloc[i, j])
            color = "white" if value > 0.58 * vmax else "#202020"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color=color)
    ax.set_title("Worst-case absolute deviation", fontsize=7.5,
                 fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#B0B0B0")
    if add_colorbar:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
        colorbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
        colorbar.set_label("Percentage points", fontsize=6.2)
    return image


def pooled_method_summary(long):
    return (
        long.groupby("method")["abs_deviation"]
        .agg(["mean", "max"])
        .reindex(METHODS)
    )


def add_pooled_summary(ax, long):
    summary = pooled_method_summary(long)
    lines = ["Pooled robustness", "MAE / maximum (p.p.)"]
    for method in METHODS:
        lines.append(
            f"{METHOD_LABELS[method]}: "
            f"{summary.loc[method, 'mean']:.2f} / "
            f"{summary.loc[method, 'max']:.2f}"
        )
    ax.text(
        0.5, -0.28, "\n".join(lines), transform=ax.transAxes,
        ha="center", va="top", fontsize=5.8, linespacing=1.35,
    )


def draw_pooled_summary_panel(ax, long):
    """Draw the pooled MAE/maximum statistics in a dedicated axis."""
    summary = pooled_method_summary(long)
    lines = ["Pooled robustness", "MAE / maximum (p.p.)", ""]
    for method in METHODS:
        lines.append(
            f"{METHOD_LABELS[method]}\n"
            f"{summary.loc[method, 'mean']:.2f} / "
            f"{summary.loc[method, 'max']:.2f}"
        )
    ax.axis("off")
    ax.text(
        0.5, 0.53, "\n".join(lines), transform=ax.transAxes,
        ha="center", va="center", fontsize=6.2, linespacing=1.25,
    )


def save_figure(fig, stem, dpi=400):
    svg = RESULT_DIR / f"{stem}.svg"
    pdf = RESULT_DIR / f"{stem}.pdf"
    png = RESULT_DIR / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {svg}")
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def draw_deviation_figure(long):
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.6), sharex=True, sharey=True)
    populate_deviation_grid(axes, long)
    fig.supxlabel("Deviation from 200 h (percentage points)",
                  fontsize=7, y=0.064)
    fig.text(0.01, 0.975, "a", fontsize=9, fontweight="bold",
             ha="left", va="top")
    add_deviation_legend(fig, y_anchor=0.002)
    fig.subplots_adjust(
        left=0.13, right=0.985, bottom=0.12, top=0.95,
        hspace=0.20, wspace=0.12,
    )
    return fig


def draw_heatmap_figure(long, robustness):
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    draw_heatmap(ax, robustness, add_colorbar=True)
    add_pooled_summary(ax, long)
    fig.text(0.015, 0.97, "b", fontsize=9, fontweight="bold",
             ha="left", va="top")
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.31, top=0.88)
    return fig


def draw_combined_figure(long, robustness):
    fig = plt.figure(figsize=(7.2, 8.2))
    grid = fig.add_gridspec(
        6, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.52, 1.35],
        left=0.13, right=0.98, bottom=0.065, top=0.955,
        hspace=0.24, wspace=0.13,
    )
    axes = np.empty((4, 3), dtype=object)
    for row in range(4):
        for col in range(3):
            axes[row, col] = fig.add_subplot(
                grid[row, col],
                sharex=axes[0, 0] if row or col else None,
                sharey=axes[0, 0] if row or col else None,
            )
    populate_deviation_grid(axes, long)
    heat_ax = fig.add_subplot(grid[5, :2])
    draw_heatmap(heat_ax, robustness, add_colorbar=True)
    summary_ax = fig.add_subplot(grid[5, 2])
    draw_pooled_summary_panel(summary_ax, long)
    fig.text(0.5, 0.344, "Deviation from 200 h (percentage points)",
             ha="center", va="center", fontsize=7)
    fig.text(0.01, 0.975, "a", fontsize=9, fontweight="bold",
             ha="left", va="top")
    fig.text(0.02, 0.255, "b", fontsize=9, fontweight="bold",
             ha="left", va="top")
    add_deviation_legend(fig, y_anchor=0.306)
    return fig


def draw_scenario_integrated_figure(long):
    """Draw four scenario panels, each comparing all methods and both caps."""
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 4.8), sharex=True, sharey=True,
    )
    y_limit = deviation_limit(long)

    for index, (ax, scenario) in enumerate(zip(axes.flat, SCENARIOS)):
        ax.axhspan(
            -TOLERANCE_BAND, TOLERANCE_BAND,
            color="#ECECEC", alpha=0.75, linewidth=0, zorder=0,
        )
        ax.axhline(0, color="#303030", linewidth=0.75, zorder=1)
        scenario_data = long[long["scenario"].eq(scenario)]

        for method in METHODS:
            method_data = scenario_data[
                scenario_data["method"].eq(method)
            ]
            color = METHOD_STYLES[method]["color"]
            for cap in SENSITIVITY_CAPS:
                series = (
                    method_data[method_data["cap"].eq(cap)]
                    .sort_values("target")
                )
                cap_style = CAP_STYLES[cap]
                ax.plot(
                    series["target"], series["deviation"],
                    color=color,
                    marker=cap_style["marker"],
                    linestyle=cap_style["linestyle"],
                    linewidth=1.0,
                    markersize=4.0,
                    markerfacecolor="white" if cap == 150 else color,
                    markeredgecolor=color,
                    markeredgewidth=0.75,
                    zorder=3,
                )
                for _, point in series.iterrows():
                    delta = float(point["deviation"])
                    if abs(delta) >= ANNOTATION_THRESHOLD:
                        offset = 0.13 if delta >= 0 else -0.13
                        ax.text(
                            float(point["target"]), delta + offset,
                            f"{delta:+.2f}", fontsize=5.2,
                            ha="center",
                            va="bottom" if delta >= 0 else "top",
                            color=color, clip_on=False,
                        )

        ax.set_title(scenario, fontsize=8, fontweight="bold", pad=4)
        ax.text(
            -0.12, 1.04, "abcd"[index],
            transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom",
        )
        ax.set_xlim(58, 102)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks(TARGETS)
        ax.set_yticks(np.arange(-3, 4, 1))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, linestyle=":")
        ax.tick_params(axis="both", labelsize=6.2, length=2.5, width=0.55)

    fig.supxlabel(
        "Renewable-energy utilization target (%)", fontsize=7, y=0.105,
    )
    fig.supylabel(
        "Deviation from 200 h (percentage points)", fontsize=7, x=0.025,
    )

    method_handles = [
        Line2D(
            [0], [0], color=METHOD_STYLES[method]["color"],
            linewidth=1.4, label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    cap_handles = [
        Line2D(
            [0], [0], color="#303030", linewidth=1.0,
            linestyle=CAP_STYLES[cap]["linestyle"],
            marker=CAP_STYLES[cap]["marker"], markersize=4.0,
            markerfacecolor="white" if cap == 150 else "#303030",
            markeredgecolor="#303030", label=CAP_STYLES[cap]["label"],
        )
        for cap in SENSITIVITY_CAPS
    ]
    tolerance_handle = Line2D(
        [0], [0], color="#B8B8B8", linewidth=5,
        alpha=0.55, label="±1 p.p. stability band",
    )
    fig.legend(
        handles=method_handles + cap_handles + [tolerance_handle],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=6,
        fontsize=6.2, handlelength=2.2, columnspacing=1.3,
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.19, top=0.95,
        hspace=0.30, wspace=0.17,
    )
    return fig


def draw_method_integrated_figure(long, labeled=True):
    """Draw one panel per method, with all four scenarios overlaid."""
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True,
    )
    y_limit = deviation_limit(long)

    for index, (ax, method) in enumerate(zip(axes, METHODS)):
        ax.axhline(0, color="#303030", linewidth=0.75, zorder=1)
        method_data = long[long["method"].eq(method)]
        maximum_point = method_data.loc[method_data["abs_deviation"].idxmax()]

        for scenario in SCENARIOS:
            scenario_data = method_data[
                method_data["scenario"].eq(scenario)
            ]
            color = SCENARIO_STYLES[scenario]["color"]
            for cap in SENSITIVITY_CAPS:
                series = (
                    scenario_data[scenario_data["cap"].eq(cap)]
                    .sort_values("target")
                )
                cap_style = CAP_STYLES[cap]
                ax.plot(
                    series["target"], series["deviation"],
                    color=color,
                    marker=cap_style["marker"],
                    linestyle=cap_style["linestyle"],
                    linewidth=0.9,
                    markersize=3.6,
                    markerfacecolor="white" if cap == 150 else color,
                    markeredgecolor=color,
                    markeredgewidth=0.7,
                    zorder=3,
                )
                if labeled:
                    for _, point in series.iterrows():
                        is_maximum = (
                            scenario == maximum_point["scenario"]
                            and cap == int(maximum_point["cap"])
                            and int(point["target"]) == int(maximum_point["target"])
                        )
                        if not is_maximum:
                            continue
                        delta = float(point["deviation"])
                        offset = 0.13 if delta >= 0 else -0.13
                        ax.text(
                            float(point["target"]), delta + offset,
                            f"{delta:+.2f}", fontsize=4.8,
                            ha="center",
                            va="bottom" if delta >= 0 else "top",
                            color=color, clip_on=False,
                        )

        if labeled:
            ax.set_title(
                METHOD_LABELS[method], fontsize=7.5,
                fontweight="bold", pad=4,
            )
            ax.text(
                -0.13, 1.04, "abc"[index],
                transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", ha="left", va="bottom",
            )
        ax.set_xlim(58, 102)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks(TARGETS)
        ax.set_yticks(np.arange(-3, 4, 1))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.4, linestyle=":")
        ax.tick_params(
            axis="both", labelsize=6.0, length=2.4, width=0.55,
            labelbottom=labeled, labelleft=(labeled and index == 0),
        )

    if labeled:
        fig.supxlabel(
            "Renewable-energy utilization target (%)", fontsize=7, y=0.135,
        )
        fig.supylabel(
            "Deviation from 200 h (percentage points)", fontsize=7, x=0.02,
        )

        scenario_handles = [
            Line2D(
                [0], [0], color=SCENARIO_STYLES[scenario]["color"],
                linewidth=1.4, label=scenario,
            )
            for scenario in SCENARIOS
        ]
        cap_handles = [
            Line2D(
                [0], [0], color="#303030", linewidth=1.0,
                linestyle=CAP_STYLES[cap]["linestyle"],
                marker=CAP_STYLES[cap]["marker"], markersize=3.8,
                markerfacecolor="white" if cap == 150 else "#303030",
                markeredgecolor="#303030", label=CAP_STYLES[cap]["label"],
            )
            for cap in SENSITIVITY_CAPS
        ]
        fig.legend(
            handles=scenario_handles + cap_handles,
            loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=6,
            fontsize=5.9, handlelength=2.0, columnspacing=1.15,
        )
        fig.subplots_adjust(
            left=0.095, right=0.985, bottom=0.25, top=0.91,
            wspace=0.16,
        )
    else:
        fig.subplots_adjust(
            left=0.045, right=0.995, bottom=0.08, top=0.985,
            wspace=0.12,
        )
    return fig


def draw_all_combined_heatmap(long):
    """Draw all scenario-method-target-cap combinations in one heatmap."""
    row_keys = [(scenario, method) for scenario in SCENARIOS for method in METHODS]
    column_keys = [(target, cap) for target in TARGETS for cap in SENSITIVITY_CAPS]
    matrix = np.full((len(row_keys), len(column_keys)), np.nan)

    for row, (scenario, method) in enumerate(row_keys):
        for col, (target, cap) in enumerate(column_keys):
            point = long[
                long["scenario"].eq(scenario)
                & long["method"].eq(method)
                & long["target"].eq(target)
                & long["cap"].eq(cap)
            ]
            if not point.empty:
                matrix[row, col] = float(point["deviation"].iloc[0])

    limit = deviation_limit(long)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(
        matrix, cmap="RdBu_r", norm=norm,
        aspect="auto", interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(column_keys)))
    ax.set_xticklabels(
        [f"{target}%\n{cap} h" for target, cap in column_keys],
        fontsize=5.6,
    )
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels(
        [f"{scenario}  |  {METHOD_LABELS[method]}"
         for scenario, method in row_keys],
        fontsize=5.8,
    )
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xlabel("Utilization target and analysis upper limit", fontsize=7)
    ax.set_title(
        "Deviation from the 200 h baseline (percentage points)",
        fontsize=8, fontweight="bold", pad=8,
    )

    for boundary in range(2, len(column_keys), 2):
        ax.axvline(boundary - 0.5, color="white", linewidth=1.2)
    for boundary in range(3, len(row_keys), 3):
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value) and abs(value) >= TOLERANCE_BAND:
                text_color = "white" if abs(value) >= 1.8 else "#202020"
                ax.text(
                    col, row, f"{value:+.1f}",
                    ha="center", va="center", fontsize=4.8,
                    fontweight="bold", color=text_color,
                )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#A0A0A0")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
    colorbar.set_label("Percentage points", fontsize=6.5)
    fig.subplots_adjust(left=0.235, right=0.93, bottom=0.15, top=0.92)
    return fig


def export_source_tables(long, robustness):
    long_output = RESULT_DIR / "cap_sensitivity_deviation_from_200h.csv"
    robustness_output = RESULT_DIR / "cap_sensitivity_method_robustness.csv"
    long.to_csv(long_output, index=False, encoding="utf-8-sig",
                float_format="%.4f")
    robustness.to_csv(robustness_output, index=False, encoding="utf-8-sig",
                      float_format="%.4f")
    print(f"Saved: {long_output}")
    print(f"Saved: {robustness_output}")


def main():
    long, robustness = load_deviations()
    export_source_tables(long, robustness)

    clean_dir = RESULT_DIR / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        (
            draw_method_integrated_figure(long, labeled=True),
            RESULT_DIR / "cap_sensitivity_three_methods_all_scenarios.svg",
        ),
        (
            draw_method_integrated_figure(long, labeled=False),
            clean_dir / "cap_sensitivity_three_methods_all_scenarios.svg",
        ),
    ]
    for figure, output in figures:
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
