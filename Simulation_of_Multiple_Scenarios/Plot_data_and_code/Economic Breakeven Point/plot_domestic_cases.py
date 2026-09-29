"""Create Fig. 2: domestic storage-pathway economic regimes.

The four main panels show the cost difference between a battery-only (BESS)
pathway and an H2-integrated pathway along renewable-penetration/autonomy
curves. Open circles are calculated economic breakeven points and the solid
line through them is the decision boundary. Panel e compares each economic
threshold with the corresponding physical turning point from Fig. 1.

All axes and figure patches are transparent. No regime is represented by a
filled background area; pathway preference is encoded only by curve colour.
"""

from pathlib import Path
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import PchipInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "sensitivity"
RESULT_DIR = SCRIPT_DIR / "results"
CLEAN_DIR = RESULT_DIR / "clean" / "domestic"
PHYSICAL_THRESHOLD_FILE = (
    SCRIPT_DIR.parent
    / "Flexibility Turning Point"
    / "results"
    / "domestic_flexibility_knees_three_methods.xlsx"
)
RESULT_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("industrial", "Industrial"),
    ("commercial", "Commercial"),
    ("rural", "Rural"),
    ("island", "Island"),
]
SCENARIO_STYLES = {
    "industrial": {"color": "#2F6B9A", "marker": "o"},
    "commercial": {"color": "#E07A2D", "marker": "s"},
    "rural": {"color": "#2F8F4E", "marker": "^"},
    "island": {"color": "#8C63A8", "marker": "D"},
}

PEN_MIN = 0.0
PEN_MAX = 99.9
OFFGRID_MAX = 500.0
COST_LIMIT = 0.6
CONSUMPTION_TARGETS = [60, 70, 80, 90, 100]
TAIL_DROP_POINTS = {}
DROP_PEN_POINTS = {}
LABEL_OFFSETS = {
    ("industrial", 70): (10, 4),
    ("industrial", 80): (0, 13),
    ("commercial", 70): (10, 4),
    ("commercial", 80): (0, 13),
    ("rural", 60): (5, 4),
    ("rural", 70): (10, 4),
    ("rural", 80): (0, 13),
    ("island", 70): (10, 4),
    ("island", 80): (0, 13),
}

BESS_COLOR = "#2166AC"
H2_COLOR = "#B2182B"
BOUNDARY_COLOR = "#202020"
COST_CMAP = LinearSegmentedColormap.from_list(
    "bess_equal_h2_integrated", [BESS_COLOR, "#F7F7F7", H2_COLOR], N=256
)
COST_NORM = TwoSlopeNorm(vmin=-COST_LIMIT, vcenter=0.0, vmax=COST_LIMIT)

FIGURE_BASENAME = "Figure2_storage_pathway_economics"
FIGURE_TITLE = (
    "Renewable penetration, consumption constraints and autonomy jointly "
    "determine storage-pathway economics"
)
FIGURE_CAPTION = (
    "Fig. 2 | Renewable penetration, consumption constraints and autonomy "
    "jointly determine storage-pathway economics. Renewable penetration and "
    "minimum off-grid duration for industrial (a), commercial (b), rural (c) "
    "and island (d) microgrids at renewable-consumption targets of 60%, 70%, "
    "80%, 90% and 100%. Curve colour gives Delta LCOE (BESS minus the "
    "H2-integrated pathway): blue denotes the BESS-preferred regime and red "
    "the H2-integrated-pathway-preferred regime. Open circles denote "
    "calculated zero crossings; the solid connecting line is the economic "
    "breakeven boundary. For the island system at the 60% consumption target, "
    "Delta LCOE does not cross zero within the analysed penetration-autonomy "
    "domain; no breakeven point is reported in d and a cross at the top of the "
    "corresponding strip denotes no crossover in e. Panel e "
    "shows the signed separation between the economic "
    "breakeven penetration (p_e) and the physical turning-point penetration "
    "(p_p), Delta p = p_e - p_p. Positive values indicate that the physical "
    "transition precedes the economic crossover; negative values indicate an "
    "earlier economic crossover. Physical turning points use the "
    "maximum-curvature criterion reported with Fig. 1."
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.unicode_minus": False,
        "savefig.transparent": True,
    }
)


def make_transparent(fig, axes):
    """Remove figure/axes patch fills without removing plotted artists."""
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)


def load_matched(scenario_slug):
    """Load factor=1.0 data and match H2 LCOE by renewable penetration."""
    scenario_dir = DATA_DIR / scenario_slug
    columns = ["cons_target_%", "min_offgrid_hours", "energy_pen_%", "LCOE_USD_per_kWh"]
    battery_all = pd.read_excel(scenario_dir / "battery.xlsx")
    h2_all = pd.read_excel(scenario_dir / "hydrogen.xlsx")
    battery_mask = np.isclose(pd.to_numeric(battery_all.iloc[:, 0], errors="coerce"), 1.0)
    h2_mask = np.isclose(pd.to_numeric(h2_all.iloc[:, 0], errors="coerce"), 1.0)
    battery = battery_all.loc[battery_mask, columns].copy()
    h2 = h2_all.loc[h2_mask, columns].copy()
    if battery.empty or h2.empty:
        raise ValueError(f"{scenario_slug}: factor=1.0 is missing from battery or hydrogen data")

    matched_h2_lcoe = []
    for target in sorted(battery["cons_target_%"].unique()):
        battery_target = battery[battery["cons_target_%"] == target]
        h2_target = h2[h2["cons_target_%"] == target]
        h2_grouped = h2_target.groupby("energy_pen_%")["LCOE_USD_per_kWh"].mean().sort_index()
        if len(h2_grouped) < 2:
            matched_h2_lcoe.append(pd.Series(np.nan, index=battery_target.index))
            continue
        values = np.interp(
            battery_target["energy_pen_%"].to_numpy(dtype=float),
            h2_grouped.index.to_numpy(dtype=float),
            h2_grouped.to_numpy(dtype=float),
        )
        matched_h2_lcoe.append(pd.Series(values, index=battery_target.index))

    battery["h2_LCOE"] = pd.concat(matched_h2_lcoe).sort_index()
    battery["cost_diff"] = battery["LCOE_USD_per_kWh"] - battery["h2_LCOE"]
    return battery


def clip_curve_by_y_cap(x, y, z, y_cap):
    """End a curve at its first linearly interpolated crossing of y_cap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    above = np.flatnonzero(y >= y_cap)
    if len(above) == 0:
        return x, y, z
    idx = int(above[0])
    if idx == 0:
        if np.isclose(y[0], y_cap):
            return x[:1], np.array([y_cap]), z[:1]
        return np.array([]), np.array([]), np.array([])
    if np.isclose(y[idx], y_cap):
        return x[: idx + 1], y[: idx + 1], z[: idx + 1]
    fraction = (y_cap - y[idx - 1]) / (y[idx] - y[idx - 1])
    return (
        np.concatenate([x[:idx], [x[idx - 1] + fraction * (x[idx] - x[idx - 1])]]),
        np.concatenate([y[:idx], [y_cap]]),
        np.concatenate([z[:idx], [z[idx - 1] + fraction * (z[idx] - z[idx - 1])]]),
    )


def make_curve(subset, scenario_slug=None, target=None):
    """Return smoothed penetration, duration and cost-difference arrays."""
    subset = subset[subset["energy_pen_%"].between(PEN_MIN, PEN_MAX)].copy()
    subset = subset[np.isfinite(subset["cost_diff"])]
    if len(subset) < 3:
        return None
    grouped = (
        subset.groupby("energy_pen_%", as_index=False)
        .agg({"min_offgrid_hours": "mean", "cost_diff": "mean"})
        .sort_values("energy_pen_%")
    )
    drop_pens = DROP_PEN_POINTS.get((scenario_slug, target), [])
    if drop_pens:
        pen = grouped["energy_pen_%"].to_numpy(dtype=float)
        keep = np.ones(len(grouped), dtype=bool)
        for drop_pen in drop_pens:
            keep &= np.abs(pen - drop_pen) > 0.02
        grouped = grouped[keep].copy()
    tail_drop = TAIL_DROP_POINTS.get((scenario_slug, target), 0)
    if tail_drop:
        grouped = grouped.iloc[:-tail_drop].copy()
    if len(grouped) < 3:
        return None
    source_x = grouped["energy_pen_%"].to_numpy(dtype=float)
    source_y = grouped["min_offgrid_hours"].to_numpy(dtype=float)
    source_z = grouped["cost_diff"].to_numpy(dtype=float)
    if len(source_x) >= 4:
        x = np.linspace(source_x[0], source_x[-1], 500)
        y = PchipInterpolator(source_x, source_y)(x)
        z = PchipInterpolator(source_x, source_z)(x)
    else:
        x, y, z = source_x, source_y, source_z
    x, y, z = clip_curve_by_y_cap(x, y, z, OFFGRID_MAX)
    return None if len(x) < 2 else (x, y, z)


def find_breakevens(x, y, z):
    """Return all linearly interpolated Delta LCOE=0 crossings."""
    points = []
    for idx in range(1, len(z)):
        z0, z1 = z[idx - 1], z[idx]
        if np.isclose(z0, 0.0, atol=1e-12):
            points.append((x[idx - 1], y[idx - 1]))
        elif (z0 < 0 <= z1) or (z0 > 0 >= z1):
            delta = z1 - z0
            if abs(delta) < 1e-12:
                continue
            fraction = -z0 / delta
            points.append((x[idx - 1] + fraction * (x[idx] - x[idx - 1]), y[idx - 1] + fraction * (y[idx] - y[idx - 1])))
    if np.isclose(z[-1], 0.0, atol=1e-12):
        points.append((x[-1], y[-1]))
    return points


def add_colored_curve(ax, x, y, z):
    """Draw one curve with segment-wise colours from Delta LCOE."""
    dense_x = np.linspace(x[0], x[-1], 1200)
    dense_y = np.interp(dense_x, x, y)
    dense_z = np.interp(dense_x, x, z)
    points = np.column_stack([dense_x, dense_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_values = 0.5 * (dense_z[:-1] + dense_z[1:])
    collection = LineCollection(
        segments, cmap=COST_CMAP, norm=COST_NORM, linewidth=1.55,
        capstyle="round", joinstyle="round", zorder=3
    )
    collection.set_array(np.clip(segment_values, -COST_LIMIT, COST_LIMIT))
    ax.add_collection(collection)
    return collection


def label_curve_end(ax, x, y, target, scenario_slug):
    """Direct-label a target curve, staggering labels in narrow panels."""
    visible = (x <= PEN_MAX) & (y >= 0) & (y <= OFFGRID_MAX)
    if not visible.any():
        return
    visible_indices = np.flatnonzero(visible)
    # Curves reaching the upper cap are labelled inside the panel in two rows.
    # This remains legible after panel e is moved to the right-hand column.
    if target in {70, 80, 90, 100} and np.nanmax(y[visible]) >= 195:
        desired_y = {100: 190, 90: 170, 80: 150, 70: 130}[target]
        idx = visible_indices[np.argmin(np.abs(y[visible_indices] - desired_y))]
        dx, dy = (0, 2)
    else:
        idx = visible_indices[-1]
        dx, dy = LABEL_OFFSETS.get((scenario_slug, target), (0, 4))
    ax.annotate(
        f"{target}%", (x[idx], y[idx]), xytext=(dx, dy), textcoords="offset points",
        ha="center", va="bottom", fontsize=6, color="#222222", zorder=7, clip_on=False
    )


def plot_panel(ax, df, scenario_slug, scenario_label, clean=False, label_boundary=False):
    """Plot one scenario and return one source-data row per target."""
    all_breakevens = []
    source_rows = []
    no_crossover_targets = []
    for target in CONSUMPTION_TARGETS:
        curve = make_curve(df[df["cons_target_%"] == target], scenario_slug, target)
        if curve is None:
            continue
        x, y, z = curve
        add_colored_curve(ax, x, y, z)
        breakevens = find_breakevens(x, y, z)
        if breakevens:
            for penetration, duration in breakevens:
                all_breakevens.append((penetration, duration, target))
                source_rows.append({
                    "scenario": scenario_label,
                    "consumption_target_%": target,
                    "economic_breakeven_penetration_%": penetration,
                    "economic_breakeven_duration_h": duration,
                    "crossover_status": "crossover",
                })
            bx, by = zip(*breakevens)
            ax.scatter(bx, by, s=20, facecolors="none", edgecolors=BOUNDARY_COLOR, linewidths=0.75, zorder=6)
        else:
            no_crossover_targets.append(target)
            source_rows.append({
                "scenario": scenario_label,
                "consumption_target_%": target,
                "economic_breakeven_penetration_%": np.nan,
                "economic_breakeven_duration_h": np.nan,
                "crossover_status": "no crossover within analysed domain",
            })
        if not clean:
            label_curve_end(ax, x, y, target, scenario_slug)

    if len(all_breakevens) >= 2:
        boundary = sorted(all_breakevens, key=lambda point: point[0])
        bx = np.asarray([point[0] for point in boundary], dtype=float)
        by = np.asarray([point[1] for point in boundary], dtype=float)
        if len(bx) >= 3:
            smooth_x = np.linspace(bx[0], bx[-1], 250)
            smooth_y = PchipInterpolator(bx, by)(smooth_x)
        else:
            smooth_x, smooth_y = bx, by
        ax.plot(smooth_x, smooth_y, color=BOUNDARY_COLOR, linewidth=1.25, solid_capstyle="round", zorder=5)
        if label_boundary and not clean:
            mid = len(smooth_x) // 2
            ax.annotate(
                "Economic breakeven\nboundary", (smooth_x[mid], smooth_y[mid]),
                xytext=(-55, 20), textcoords="offset points", ha="left", va="bottom",
                fontsize=5.7, color=BOUNDARY_COLOR,
                arrowprops={"arrowstyle": "-", "color": BOUNDARY_COLOR, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 2},
                zorder=8,
            )

    if no_crossover_targets and not clean:
        targets = ", ".join(f"{target}%" for target in no_crossover_targets)
        ax.text(
            0.04, 0.94, f"{targets}: No crossover\nwithin analysed domain",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            color="#444444", linespacing=1.15, zorder=8
        )
    if not clean:
        ax.set_title(scenario_label, fontsize=7.2, fontweight="bold", pad=3, loc="left")
    ax.set_xlim(PEN_MIN, 100)
    ax.set_ylim(0, OFFGRID_MAX)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, OFFGRID_MAX + 1, 50))
    ax.grid(False)
    ax.tick_params(labelsize=5.8)
    return source_rows


def load_physical_thresholds():
    """Load the latest maximum-curvature turning points used in Fig. 1."""
    if not PHYSICAL_THRESHOLD_FILE.exists():
        raise FileNotFoundError(f"Physical turning-point results are required for panel e: {PHYSICAL_THRESHOLD_FILE}")
    thresholds = pd.read_excel(PHYSICAL_THRESHOLD_FILE, sheet_name="method_comparison")
    required = ["scenario", "consumption_target_%", "maximum_curvature_penetration_%"]
    missing = set(required).difference(thresholds.columns)
    if missing:
        raise ValueError(f"Physical-threshold file is missing columns: {sorted(missing)}")
    return thresholds[required].copy()


def build_summary_data(economic_rows):
    """Combine economic and physical thresholds and calculate signed separation."""
    economic = pd.DataFrame(economic_rows)
    physical = load_physical_thresholds()
    summary = economic.merge(physical, on=["scenario", "consumption_target_%"], how="left", validate="one_to_one")
    if summary["maximum_curvature_penetration_%"].isna().any():
        missing = summary.loc[summary["maximum_curvature_penetration_%"].isna(), ["scenario", "consumption_target_%"]]
        raise ValueError(f"Missing physical thresholds:\n{missing.to_string(index=False)}")
    summary = summary.rename(columns={"maximum_curvature_penetration_%": "physical_turning_penetration_%"})
    summary["signed_separation_percentage_points"] = (
        summary["economic_breakeven_penetration_%"] - summary["physical_turning_penetration_%"]
    )
    return summary


def plot_summary(axes, summary, clean=False):
    """Plot four vertically separated scenario strips in panel e."""
    # Categorical no-crossover marker; it is not a numerical Delta-p value.
    no_crossover_y = 14.0
    for index, (ax, (scenario_slug, scenario_label)) in enumerate(zip(axes, SCENARIOS)):
        group = summary[summary["scenario"] == scenario_label].sort_values("consumption_target_%")
        style = SCENARIO_STYLES[scenario_slug]
        x_values = group["consumption_target_%"].to_numpy(dtype=float)
        separation = group["signed_separation_percentage_points"].to_numpy(dtype=float)
        valid = np.isfinite(separation)
        ax.plot(
            x_values[valid],
            separation[valid],
            color=style["color"],
            linewidth=1.05,
            zorder=3,
        )
        positive = valid & (separation >= 0)
        negative = valid & (separation < 0)
        ax.scatter(
            x_values[positive],
            separation[positive],
            s=20,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
            clip_on=False,
        )
        ax.scatter(
            x_values[negative],
            separation[negative],
            s=22,
            marker=style["marker"],
            facecolor="none",
            edgecolor=style["color"],
            linewidth=0.9,
            zorder=4,
            clip_on=False,
        )
        if (~valid).any():
            no_cross_x = x_values[~valid]
            ax.scatter(
                no_cross_x,
                np.full((~valid).sum(), no_crossover_y),
                s=28,
                marker="x",
                color="#222222",
                linewidth=1.1,
                zorder=4,
                clip_on=False,
            )
            if not clean:
                ax.annotate(
                    "No crossover",
                    (no_cross_x[0], no_crossover_y),
                    xytext=(7, -2),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=4.8,
                    color="#222222",
                    zorder=5,
                )
        ax.axhline(0, color="#555555", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        if not clean:
            ax.text(
                0.98,
                0.95,
                scenario_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.6,
                fontweight="bold",
                color=style["color"],
            )
        ax.set_xlim(60, 100)
        ax.set_ylim(-12.5, 15.5)
        ax.set_yticks([-10, 0, 10])
        ax.tick_params(axis="y", labelsize=4.8, pad=1.5)
        ax.grid(False)
        if index < len(SCENARIOS) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xticks(CONSUMPTION_TARGETS)
            ax.tick_params(axis="x", labelsize=5.0, pad=2)


def build_figure():
    """Build a-d as a 2x2 grid with the summary panel e on the right."""
    fig = plt.figure(figsize=(7.2, 4.85), layout="constrained")
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.09],
        width_ratios=[1.0, 1.0, 0.92],
        hspace=0.10,
        wspace=0.12,
    )
    main_axes = [
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
    ]
    summary_subfig = fig.add_subfigure(grid[:2, 2])
    summary_grid = summary_subfig.add_gridspec(
        5,
        1,
        height_ratios=[0.45, 1.0, 1.0, 1.0, 1.0],
        hspace=0.14,
    )
    summary_axes = [summary_subfig.add_subplot(summary_grid[1, 0])]
    summary_axes.extend(
        summary_subfig.add_subplot(
            summary_grid[row, 0],
            sharex=summary_axes[0],
            sharey=summary_axes[0],
        )
        for row in range(2, 5)
    )
    colorbar_ax = fig.add_subplot(grid[2, :])

    all_economic_rows = []
    for index, (ax, (scenario_slug, scenario_label)) in enumerate(zip(main_axes, SCENARIOS)):
        print(f"Processing {scenario_label}...")
        df = load_matched(scenario_slug)
        all_economic_rows.extend(plot_panel(
            ax, df, scenario_slug, scenario_label, clean=False, label_boundary=(index == 0)
        ))
        ax.text(-0.105, 1.105, "abcd"[index], transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.set_xlabel("Renewable penetration (%)", fontsize=6.3)
        ax.set_ylabel("Minimum off-grid duration (h)", fontsize=6.3)

    summary = build_summary_data(all_economic_rows)
    plot_summary(summary_axes, summary)
    summary_subfig.text(
        0.06,
        0.995,
        "Signed separation between economic\nand physical thresholds",
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    summary_subfig.supxlabel("Consumption target (%)", fontsize=6.3, y=-0.015)
    summary_subfig.supylabel(
        r"$\Delta p=p_e-p_p$ (percentage points)",
        fontsize=5.5,
        x=-0.055,
    )
    summary_subfig.text(-0.13, 0.995, "e", fontsize=8, fontweight="bold", va="top")

    scalar_mappable = plt.cm.ScalarMappable(cmap=COST_CMAP, norm=COST_NORM)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-COST_LIMIT, -0.3, 0, 0.3, COST_LIMIT])
    colorbar.ax.tick_params(labelsize=5.3, width=0.5, length=2.3, pad=1.5)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label(r"$\Delta$LCOE (BESS $-$ H$_2$-integrated pathway; USD kWh$^{-1}$)", fontsize=6.1, labelpad=2)
    colorbar.ax.text(0.0, 2.6, "BESS-preferred", transform=colorbar.ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=BESS_COLOR)
    colorbar.ax.text(1.0, 2.6, r"H$_2$-integrated pathway preferred", transform=colorbar.ax.transAxes, ha="right", va="bottom", fontsize=5.8, fontweight="bold", color=H2_COLOR)
    summary_subfig.patch.set_facecolor("none")
    summary_subfig.patch.set_alpha(0.0)
    make_transparent(fig, main_axes + list(summary_axes) + [colorbar_ax])
    return fig, summary


def build_clean_scenario_panel(scenario_slug, scenario_label):
    """Create one text-free scenario panel for external figure assembly."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    df = load_matched(scenario_slug)
    plot_panel(
        ax,
        df,
        scenario_slug,
        scenario_label,
        clean=True,
        label_boundary=False,
    )
    ax.tick_params(axis="both", labelbottom=False, labelleft=False)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.99)
    make_transparent(fig, [ax])
    return fig


def build_clean_summary_panel(summary):
    """Create a text-free vertical four-strip version of panel e."""
    fig = plt.figure(figsize=(2.5, 4.6))
    grid = fig.add_gridspec(4, 1, hspace=0.14)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(
        fig.add_subplot(grid[row, 0], sharex=axes[0], sharey=axes[0])
        for row in range(1, 4)
    )
    plot_summary(axes, summary, clean=True)
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
        )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.05, top=0.99)
    make_transparent(fig, axes)
    return fig


def save_svg(fig, output_path):
    """Save one transparent SVG with editable text."""
    fig.savefig(output_path, bbox_inches="tight", transparent=True)


def save_clean_outputs(summary):
    """Save text-free a-e panels as SVG files only."""
    for scenario_slug, scenario_label in SCENARIOS:
        fig = build_clean_scenario_panel(scenario_slug, scenario_label)
        output_path = CLEAN_DIR / f"{scenario_slug}.svg"
        save_svg(fig, output_path)
        plt.close(fig)
        print(f"Saved clean: {output_path}")

    fig = build_clean_summary_panel(summary)
    output_path = CLEAN_DIR / "signed_separation.svg"
    save_svg(fig, output_path)
    plt.close(fig)
    print(f"Saved clean: {output_path}")


def main():
    fig, summary = build_figure()
    output_base = RESULT_DIR / FIGURE_BASENAME
    output_svg = output_base.with_suffix(".svg")
    save_svg(fig, output_svg)
    plt.close(fig)
    save_clean_outputs(summary)
    summary.to_csv(RESULT_DIR / f"{FIGURE_BASENAME}_source_data.csv", index=False, encoding="utf-8-sig")
    (RESULT_DIR / f"{FIGURE_BASENAME}_caption.txt").write_text(FIGURE_CAPTION + "\n", encoding="utf-8")

    positive = int((summary["signed_separation_percentage_points"] > 0).sum())
    negative = int((summary["signed_separation_percentage_points"] < 0).sum())
    no_crossover = int(summary["signed_separation_percentage_points"].isna().sum())
    print(f"Figure title: {FIGURE_TITLE}")
    print(f"Summary audit: positive={positive}, negative={negative}, no_crossover={no_crossover}")
    print(f"Saved: {output_svg}")


if __name__ == "__main__":
    main()
