"""
Nature-style 1 x 3 sensitivity figure for the three global cases.

1D-interpolation variant of gradient_costdiff_curves_sensitivity.py.
-------------------------------------------------------------------
The only difference vs the 2D version is in load_matched(): instead of
interpolating hydrogen LCOE on the 2D plane (min_offgrid_hours, energy_pen_%)
with scipy.interpolate.griddata, here we interpolate along a SINGLE axis
(energy_pen_%) with np.interp (linear, clamped at the H2 penetration range).

The global comparison contains Norway, USA, and Zambia.  All three datasets
contain the baseline cost factor (1.0), so the cases are arranged as panels
instead of arranging cost factors as panels.

Each panel uses battery and hydrogen results under the same overall cost
factor. Each curve is one consumption target. Curve color encodes
Delta cost = battery LCOE - hydrogen LCOE:
    blue  : battery storage has lower LCOE
    white : equal LCOE
    red   : hydrogen storage has lower LCOE
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DATA_DIR = SCRIPT_DIR / "data" / "global"
RESULT_DIR = SCRIPT_DIR / "results"
CLEAN_DIR = RESULT_DIR / "clean" / "global"
RESULT_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("Norway", "norway", "Norway"),
    ("USA", "usa", "USA"),
    ("Zambia", "zambia", "Zambia"),
]

BAT_FILE_CANDIDATES = [
    "battery.xlsx",
]
H2_FILE_CANDIDATES = [
    "hydrogen.xlsx",
]

DATA_COLS = ["cons_target_%", "min_offgrid_hours", "energy_pen_%", "LCOE_USD_per_kWh"]
PRICE_FACTOR = 1.0

PEN_MIN = 0.0
PEN_MAX = 99.9
OFFGRID_MAX = 200.0
COST_LIMIT = 0.6
CONSUMPTION_TARGETS = [60, 70, 80, 90, 100]

TARGET_STYLES = {target: "solid" for target in CONSUMPTION_TARGETS}

DROP_LAST_POINT_RULES = {
    ("rural", 0.9, 60),
}

COST_CMAP = LinearSegmentedColormap.from_list(
    "battery_equal_hydrogen",
    ["#2166AC", "#F7F7F7", "#B2182B"],
    N=256,
)
COST_NORM = TwoSlopeNorm(vmin=-COST_LIMIT, vcenter=0.0, vmax=COST_LIMIT)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.unicode_minus": False,
    }
)


def find_existing_file(data_dir, candidates):
    """Return the first matching source file in one scenario directory."""
    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"No source file found in {data_dir}; expected one of: {expected}")


def read_sources(data_dir):
    """Read result files and infer the first-column cost-factor names."""
    bat_file = find_existing_file(data_dir, BAT_FILE_CANDIDATES)
    h2_file = find_existing_file(data_dir, H2_FILE_CANDIDATES)
    bat_all = pd.read_excel(bat_file)
    h2_all = pd.read_excel(h2_file)
    bat_factor_col = bat_all.columns[0]
    h2_factor_col = h2_all.columns[0]
    return bat_all, h2_all, bat_factor_col, h2_factor_col


def load_matched(bat_all, h2_all, bat_factor_col, h2_factor_col, price_factor):
    """Load one factor pair and interpolate hydrogen LCOE to battery points.

    1D variant: H2 LCOE is interpolated as a function of a single variable
    (renewable penetration, energy_pen_%). For each consumption target the H2
    samples are collapsed to mean LCOE per penetration value (sorted), then
    linearly interpolated (np.interp) onto the battery points' penetration.
    Out-of-range battery penetrations are clamped to the H2 penetration
    endpoints (np.interp default), so no NaNs are produced.
    """
    bat = bat_all[bat_all[bat_factor_col] == price_factor][DATA_COLS].copy()
    h2 = h2_all[h2_all[h2_factor_col] == price_factor][DATA_COLS].copy()

    if bat.empty:
        raise ValueError(f"No battery rows found for {bat_factor_col}={price_factor}")
    if h2.empty:
        raise ValueError(f"No hydrogen rows found for {h2_factor_col}={price_factor}")

    matched = []
    for target in sorted(bat["cons_target_%"].unique()):
        b = bat[bat["cons_target_%"] == target]
        h = h2[h2["cons_target_%"] == target]

        # Collapse H2 to LCOE-vs-penetration: mean LCOE at each penetration.
        h_grouped = h.groupby("energy_pen_%")["LCOE_USD_per_kWh"].mean().sort_index()
        if len(h_grouped) < 2:
            # Need at least two distinct penetration values to define a 1D curve.
            matched.append(pd.Series(np.nan, index=b.index))
            continue

        h_pen = h_grouped.index.to_numpy(dtype=float)
        h_lcoe = h_grouped.to_numpy(dtype=float)
        b_pen = b["energy_pen_%"].to_numpy(dtype=float)

        # np.interp: linear, clamps to h_lcoe endpoints outside [h_pen[0], h_pen[-1]].
        b_h2_lcoe = np.interp(b_pen, h_pen, h_lcoe)
        matched.append(pd.Series(b_h2_lcoe, index=b.index))

    bat["h2_LCOE"] = pd.concat(matched).sort_index()
    bat["cost_diff"] = bat["LCOE_USD_per_kWh"] - bat["h2_LCOE"]
    return bat


def clip_curve_by_y_cap(x, y, z, y_cap):
    """End a curve at its first linearly interpolated crossing of ``y_cap``."""
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

    x1, y1, z1 = x[idx - 1], y[idx - 1], z[idx - 1]
    x2, y2, z2 = x[idx], y[idx], z[idx]
    fraction = (y_cap - y1) / (y2 - y1)
    x_cross = x1 + fraction * (x2 - x1)
    z_cross = z1 + fraction * (z2 - z1)
    return (
        np.concatenate([x[:idx], [x_cross]]),
        np.concatenate([y[:idx], [y_cap]]),
        np.concatenate([z[:idx], [z_cross]]),
    )


def make_curve(sub):
    """Return smoothed penetration, off-grid duration, and cost-difference arrays."""
    sub = sub[sub["energy_pen_%"].between(PEN_MIN, PEN_MAX)].copy()
    sub = sub[np.isfinite(sub["cost_diff"])]
    if len(sub) < 3:
        return None

    grouped = (
        sub.groupby("energy_pen_%", as_index=False)
        .agg({"min_offgrid_hours": "mean", "cost_diff": "mean"})
        .sort_values("energy_pen_%")
    )
    if len(grouped) < 3:
        return None

    x = grouped["energy_pen_%"].to_numpy(dtype=float)
    y = grouped["min_offgrid_hours"].to_numpy(dtype=float)
    z = grouped["cost_diff"].to_numpy(dtype=float)

    if len(x) >= 4:
        source_x = x
        x = np.linspace(source_x[0], source_x[-1], 500)
        y = PchipInterpolator(source_x, y)(x)
        z = PchipInterpolator(source_x, z)(x)

    x, y, z = clip_curve_by_y_cap(x, y, z, OFFGRID_MAX)
    return None if len(x) < 2 else (x, y, z)


def add_colored_curve(ax, x, y, z, linestyle="solid"):
    """Draw one curve with segment-wise colors from cost difference."""
    if len(x) >= 2:
        base_x, base_y, base_z = x, y, z
        x = np.linspace(base_x[0], base_x[-1], 1200)
        y = np.interp(x, base_x, base_y)
        z = np.interp(x, base_x, base_z)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_values = 0.5 * (z[:-1] + z[1:])
    lc = LineCollection(
        segments,
        cmap=COST_CMAP,
        norm=COST_NORM,
        linewidth=1.45,
        linestyles=linestyle,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    lc.set_array(np.clip(seg_values, -COST_LIMIT, COST_LIMIT))
    ax.add_collection(lc)
    return lc


def add_breakeven_points(ax, x, y, z):
    """Mark all Delta cost = 0 crossings on a curve."""
    bx, by = [], []
    for i in range(1, len(z)):
        if z[i - 1] == 0:
            bx.append(x[i - 1])
            by.append(y[i - 1])
        elif (z[i - 1] < 0 <= z[i]) or (z[i - 1] > 0 >= z[i]):
            dz = z[i] - z[i - 1]
            if abs(dz) < 1e-12:
                continue
            frac = -z[i - 1] / dz
            bx.append(x[i - 1] + frac * (x[i] - x[i - 1]))
            by.append(y[i - 1] + frac * (y[i] - y[i - 1]))

    if bx:
        ax.scatter(
            bx,
            by,
            s=18,
            facecolors="white",
            edgecolors="#252525",
            linewidths=0.55,
            zorder=5,
        )
    return list(zip(bx, by))


def label_curve_target(ax, x, y, target):
    """Place target labels at staggered low-height positions on the curves."""
    visible = (x >= PEN_MIN) & (x <= PEN_MAX) & (y >= 0) & (y <= OFFGRID_MAX)
    if not visible.any():
        return
    visible_idx = np.where(visible)[0]
    desired_y = {60: 8.0, 70: 18.0, 80: 28.0, 90: 38.0, 100: 48.0}[target]
    idx = visible_idx[np.argmin(np.abs(y[visible_idx] - desired_y))]
    x_offset = -3 if x[idx] > 96 else 3
    ax.annotate(
        f"{target}%",
        (x[idx], y[idx]),
        xytext=(x_offset, 0),
        textcoords="offset points",
        ha="right" if x_offset < 0 else "left",
        va="center",
        fontsize=5.2,
        color="#222222",
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.2},
    )


def apply_curve_adjustments(sub, scene_tag, price_factor, target):
    """Apply narrowly scoped point removals for known visual outliers."""
    if (scene_tag, round(price_factor, 1), target) not in DROP_LAST_POINT_RULES:
        return sub
    if sub.empty:
        return sub
    return sub.drop(index=sub["energy_pen_%"].idxmax())


def plot_panel(
    ax,
    df,
    price_factor,
    scene_tag=None,
    clean=False,
    title=None,
    show_breakeven_table=True,
):
    breakeven_labels = []
    breakeven_points = []
    for target in CONSUMPTION_TARGETS:
        sub = df[df["cons_target_%"] == target]
        sub = apply_curve_adjustments(sub, scene_tag, price_factor, target)
        curve = make_curve(sub)
        if curve is None:
            continue
        x, y, z = curve
        add_colored_curve(ax, x, y, z, TARGET_STYLES[target])
        breakevens = add_breakeven_points(ax, x, y, z)
        for bx, by in breakevens:
            breakeven_points.append((bx, by))
            breakeven_labels.append(f"{target:>3d}%     {bx:>5.1f}%     {by:>5.0f}h")
        if not clean:
            label_curve_target(ax, x, y, target)

    if len(breakeven_points) >= 2:
        points = (
            pd.DataFrame(breakeven_points, columns=["x", "y"])
            .groupby("x", as_index=False)
            .mean()
            .sort_values("x")
        )
        bx = points["x"].to_numpy(dtype=float)
        by = points["y"].to_numpy(dtype=float)
        if len(bx) >= 3:
            smooth_x = np.linspace(bx[0], bx[-1], 200)
            smooth_y = PchipInterpolator(bx, by)(smooth_x)
        else:
            smooth_x, smooth_y = bx, by
        ax.plot(
            smooth_x,
            smooth_y,
            color="#252525",
            linewidth=0.65,
            linestyle=(0, (2.0, 1.4)),
            zorder=4,
        )

    if not clean:
        panel_title = title or f"Cost factor = {price_factor:.1f}"
        ax.set_title(panel_title, fontsize=7.2, fontweight="bold", pad=3, loc="left")
    ax.set_xlim(PEN_MIN, 100)
    ax.set_ylim(0, OFFGRID_MAX)
    ax.set_xticks(np.arange(PEN_MIN, 101, 20))
    ax.set_yticks(np.arange(0, OFFGRID_MAX + 1, 50))
    ax.grid(False)
    ax.tick_params(labelsize=5.6)

    if not clean and show_breakeven_table and breakeven_labels:
        header = "Breakeven points\nCon(%)   Pen(%)   Offgrid(h)"
        ax.text(
            0.02,
            0.96,
            header + "\n" + "\n".join(breakeven_labels),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontfamily="monospace",
            color="#222222",
            linespacing=1.15,
            zorder=7,
        )


def build_figure(clean=False):
    """Build the 1 x 3 comparison across the global cases."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    panel_letters = "abc"

    for ax, (scene_name, scene_tag, scene_label), letter in zip(
        axes, SCENARIOS, panel_letters
    ):
        data_dir = ROOT_DATA_DIR / scene_name
        bat_all, h2_all, bat_factor_col, h2_factor_col = read_sources(data_dir)
        bat_factors = set(bat_all[bat_factor_col].dropna())
        h2_factors = set(h2_all[h2_factor_col].dropna())
        if PRICE_FACTOR not in bat_factors or PRICE_FACTOR not in h2_factors:
            raise ValueError(
                f"{scene_name} does not contain cost factor {PRICE_FACTOR:.1f} "
                "in both battery and hydrogen sources"
            )

        print(f"Processing {scene_name}, cost factor={PRICE_FACTOR}...")
        df = load_matched(
            bat_all,
            h2_all,
            bat_factor_col,
            h2_factor_col,
            PRICE_FACTOR,
        )
        plot_panel(
            ax,
            df,
            PRICE_FACTOR,
            scene_tag=scene_tag,
            clean=clean,
            title=scene_label,
            show_breakeven_table=False,
        )
        if not clean:
            ax.text(
                -0.14,
                1.06,
                letter,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
            )

    if not clean:
        for ax in axes:
            ax.set_xlabel("Renewable penetration (%)", fontsize=6.3)
        axes[0].set_ylabel("Min. off-grid duration (h)", fontsize=6.3)
    else:
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if not clean:
        breakeven_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#252525",
            markeredgewidth=0.55,
            markersize=4.0,
            label="Breakeven",
        )
        fig.legend(
            handles=[breakeven_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=1,
            fontsize=5.5,
            frameon=False,
            handlelength=2.4,
        )

    sm = plt.cm.ScalarMappable(cmap=COST_CMAP, norm=COST_NORM)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.20,
        aspect=45,
    )
    if not clean:
        cbar.set_label(r"$\Delta$Cost (battery - H$_2$, USD/kWh)", fontsize=6.2)
    cbar.set_ticks([-COST_LIMIT, -0.3, 0, 0.3, COST_LIMIT])
    if not clean:
        cbar.ax.tick_params(labelsize=5.4, width=0.5, length=2.5)
        cbar.ax.text(
            0.0,
            -1.65,
            "Battery advantage",
            transform=cbar.ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.2,
            color="#2166AC",
        )
        cbar.ax.text(
            1.0,
            -1.65,
            "H$_2$ advantage",
            transform=cbar.ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
            color="#B2182B",
        )
    else:
        cbar.ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        cbar.outline.set_visible(False)
    cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.28, wspace=0.14)
    return fig


def save_individual_panels(scene_name, scene_tag, data_dir, clean=True):
    """Save one standalone panel for a global case."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    bat_all, h2_all, bat_factor_col, h2_factor_col = read_sources(data_dir)
    print(f"Saving individual panel for {scene_name}, cost factor={PRICE_FACTOR}...")
    df = load_matched(
        bat_all,
        h2_all,
        bat_factor_col,
        h2_factor_col,
        PRICE_FACTOR,
    )
    pf, pa = plt.subplots(1, 1, figsize=(3.6, 2.8))
    plot_panel(
        pa,
        df,
        PRICE_FACTOR,
        scene_tag=scene_tag,
        clean=clean,
        title=scene_name,
        show_breakeven_table=False,
    )
    if clean:
        pa.set_xticklabels([])
        pa.set_yticklabels([])
    else:
        pa.set_xlabel("Renewable penetration (%)", fontsize=6.3)
        pa.set_ylabel("Min. off-grid duration (h)", fontsize=6.3)
    out_panel = CLEAN_DIR / scene_name
    pf.savefig(out_panel.with_suffix(".svg"), bbox_inches="tight")
    plt.close(pf)
    print(f"  Saved: {out_panel.relative_to(SCRIPT_DIR)}.svg")


def main():
    for scene_name, scene_tag, _ in SCENARIOS:
        data_dir = ROOT_DATA_DIR / scene_name
        save_individual_panels(scene_name, scene_tag, data_dir, clean=True)

    fig_labeled = build_figure(clean=False)
    out_labeled = RESULT_DIR / "global_comparison"
    fig_labeled.savefig(out_labeled.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig_labeled)
    print(f"Figure saved: {out_labeled.relative_to(SCRIPT_DIR)}.svg")


if __name__ == "__main__":
    main()
