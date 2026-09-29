"""Global storage-duration curves with maximum-curvature flexibility knees.

Reads:
  data/global/{Norway,USA,Zambia}/battery.xlsx

Writes one labeled comparison SVG, text-free panels, and a results CSV.
"""
import io
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Microsoft YaHei",
        "SimHei",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "sans-serif",
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


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "global"
RESULT_DIR = BASE_DIR / "results"
CLEAN_DIR = RESULT_DIR / "clean" / "global"
RESULT_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELED_SVG = RESULT_DIR / "global_comparison.svg"
OUTPUT_CLEAN_COMPARISON_SVG = RESULT_DIR / "clean" / "global_comparison.svg"
OUTPUT_KNEE_CSV = RESULT_DIR / "global_flexibility_knees_max_curvature.csv"

SCENARIOS = [
    ("Norway", DATA_DIR / "Norway" / "battery.xlsx"),
    ("USA", DATA_DIR / "USA" / "battery.xlsx"),
    ("Zambia", DATA_DIR / "Zambia" / "battery.xlsx"),
]

Y_COLUMN = "storage_duration_h"
MIN_STORAGE_HOURS = 1.0
Y_CAP = 200.0
Y_LABEL = "Storage duration (h)"

# Search only after the initial horizontal plateau, then use axis-normalized
# maximum curvature with the search range matching the displayed 200 h cap.
MIN_METHOD_POINTS = 7
METHOD_INTERPOLATION_POINTS = 300
SUSTAINED_GROWTH_SEGMENTS = 3
COMMON_SAVGOL_WINDOW = 101
COMMON_SAVGOL_POLYORDER = 3

KNOWN_SOURCE_COLUMNS = {
    "bat_price_factor",
    "PV_MW",
    "Wind_MW",
    "cons_target_%",
    "total_MWh",
    "total_MW",
    "grid_kWh",
    "grid_cost_USD",
    "curtail_kWh",
    "actual_cons_%",
    "energy_pen_%",
    "total_cost_USD",
    "LCOE_USD_per_kWh",
}

COLOR_MAP = {
    60: "#2166AC",
    70: "#4393C3",
    80: "#7FCDBB",
    90: "#41AB5D",
    100: "#006D2C",
}

MAX_CURVATURE_MARKER = {"marker": "o", "markersize": 4.5}


def normalize_percent(series):
    s = series.astype(str).str.replace("%", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")
    if s.max() <= 1.5:
        s = s * 100
    return s


def get_smooth_xy(x, y, num=300):
    """Interpolate every curve with the original shape-preserving PCHIP."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        y_sum = np.bincount(inverse, weights=y)
        y_count = np.bincount(inverse)
        x, y = unique_x, y_sum / y_count
    xs = np.linspace(x[0], x[-1], num)
    ys = PchipInterpolator(x, y, extrapolate=False)(xs)
    return xs, ys


def clip_curve_by_y_min(x, y, y_min):
    if len(x) == 0:
        return x, y
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = y >= y_min
    if keep.all():
        return x, y
    if not keep.any():
        return np.array([]), np.array([])

    idx = int(np.argmax(keep))
    if idx == 0:
        return x, y

    x1, y1, x2, y2 = x[idx - 1], y[idx - 1], x[idx], y[idx]
    xc = x2 if y2 == y1 else x1 + (y_min - y1) * (x2 - x1) / (y2 - y1)
    return np.concatenate([[xc], x[idx:]]), np.concatenate([[y_min], y[idx:]])


def clip_curve_by_y_cap(x, y, y_cap):
    if len(x) == 0:
        return x, y
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = y <= y_cap
    if keep.all():
        return x, y
    if not keep[0]:
        return np.array([]), np.array([])

    idx = int(np.argmax(~keep))
    x1, y1, x2, y2 = x[idx - 1], y[idx - 1], x[idx], y[idx]
    xc = x1 if y2 == y1 else x1 + (y_cap - y1) * (x2 - x1) / (y2 - y1)
    return np.concatenate([x[:idx], [xc]]), np.concatenate([y[:idx], [y_cap]])


def find_growth_start(x_curve, y_curve):
    """Return the end of the initial plateau before sustained positive growth."""
    x = np.asarray(x_curve, float)
    y = np.asarray(y_curve, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) <= SUSTAINED_GROWTH_SEGMENTS:
        return None

    # Scale-aware tolerance rejects floating-point jitter while retaining small
    # but genuine increases such as 0.01 h in the source data.
    y_scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
    tolerance = max(1e-9, 1e-8 * y_scale)
    positive = np.diff(y) > tolerance
    run = SUSTAINED_GROWTH_SEGMENTS
    for idx in range(len(positive) - run + 1):
        if np.all(positive[idx:idx + run]):
            return float(x[idx])
    return None


def prepare_method_interval(x, y, growth_start_x):
    """Remove the initial plateau, resample the growth region, and normalize."""
    if growth_start_x is None:
        return None
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= growth_start_x)
    x, y = x[valid], y[valid]
    if len(x) < 2 or x[-1] <= x[0]:
        return None

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        x, y = unique_x, sums / counts
    if len(x) < 2:
        return None

    # A fixed point count makes the curvature smoothing comparable even when a
    # country's growth region occupies only a narrow penetration interval.
    method_x = np.linspace(x[0], x[-1], METHOD_INTERPOLATION_POINTS)
    method_y = PchipInterpolator(x, y)(method_x)
    x, y = method_x, method_y
    if len(x) < MIN_METHOD_POINTS:
        return None
    if x[-1] <= x[0] or Y_CAP <= 0:
        return None

    # Curvature must use the common geometry visible in the figure.  Scaling
    # every post-plateau interval independently to 0--1 over-expands narrow
    # growth regions (most noticeably Norway) and pushes the selected point
    # too far up the steep tail.  The displayed axes are shared by all panels,
    # so normalize against those fixed spans instead.
    xn = x / 100.0
    yn = y / Y_CAP
    return x, y, xn, yn


def apply_common_savgol(xn, yn, deriv=0):
    """Apply the shared Savitzky-Golay model used by all knee methods."""
    window = min(
        COMMON_SAVGOL_WINDOW,
        len(yn) if len(yn) % 2 else len(yn) - 1,
    )
    min_window = COMMON_SAVGOL_POLYORDER + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(min_window, window)
    delta = float(np.mean(np.diff(xn)))
    return savgol_filter(
        yn,
        window_length=window,
        polyorder=COMMON_SAVGOL_POLYORDER,
        deriv=deriv,
        delta=delta,
        mode="interp",
    )


def find_max_curvature(x, y, growth_start_x):
    """Find maximum curvature using the common Savitzky-Golay derivatives."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    x, y, xn, yn = prepared
    first = apply_common_savgol(xn, yn, deriv=1)
    second = apply_common_savgol(xn, yn, deriv=2)

    # Savitzky-Golay derivatives within half a window of either boundary are
    # dominated by one-sided polynomial extrapolation. Excluding that guard
    # region prevents the plateau-to-growth boundary from becoming a false knee.
    edge = COMMON_SAVGOL_WINDOW // 2
    if len(xn) <= 2 * edge + 1:
        return None, None
    curvature = np.abs(second) / np.power(1.0 + first ** 2, 1.5)
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    if not np.isfinite(curvature).any():
        return None, None
    idx = int(np.nanargmax(curvature))
    return float(x[idx]), float(y[idx])


def curvature_search_bounds(x, y, growth_start_x):
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    interval_x = prepared[0]
    return float(interval_x[0]), float(interval_x[-1])


def find_first_prominent_curvature_peak(
        x, y, growth_start_x, return_info=False):
    """Return the leftmost prominent curvature peak in the growth region."""
    empty_info = {
        "明显曲率峰数量": 0,
        "有效明显曲率峰数量": 0,
        "曲率峰储能时长下限(h)": np.nan,
        "是否回退到最大曲率": False,
    }
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return (None, None, empty_info) if return_info else (None, None)

    x, y, xn, yn = prepared
    first = apply_common_savgol(xn, yn, deriv=1)
    second = apply_common_savgol(xn, yn, deriv=2)
    curvature = np.abs(second) / np.power(1.0 + first ** 2, 1.5)

    edge = max(
        COMMON_SAVGOL_WINDOW // 2 + 1,
        int(round(0.02 * len(x))),
    )
    if len(x) <= 2 * edge + 1:
        return (None, None, empty_info) if return_info else (None, None)
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    finite = np.isfinite(curvature)
    if not finite.any():
        return (None, None, empty_info) if return_info else (None, None)

    max_curvature = float(np.nanmax(curvature))
    peak_input = np.where(finite, curvature, -np.inf)
    min_distance = max(
        1, int(round(FIRST_PEAK_MIN_DISTANCE_FRACTION * len(x))))
    min_peak_height = np.nextafter(
        FIRST_PEAK_MIN_HEIGHT_RATIO * max_curvature, np.inf)
    peaks, _ = find_peaks(
        peak_input,
        height=min_peak_height,
        distance=min_distance,
    )

    min_peak_y = float(
        y[0] + FIRST_PEAK_MIN_RELATIVE_RISE * (y[-1] - y[0]))
    valid_peaks = peaks[(y[peaks] >= min_peak_y) & (first[peaks] > 0.0)]
    used_fallback = False
    if len(valid_peaks):
        idx = int(valid_peaks[0])
    else:
        eligible = np.flatnonzero(
            finite & (y >= min_peak_y) & (first > 0.0))
        if not len(eligible):
            info = dict(empty_info)
            info["明显曲率峰数量"] = int(len(peaks))
            info["曲率峰储能时长下限(h)"] = min_peak_y
            return (None, None, info) if return_info else (None, None)
        idx = int(eligible[np.argmax(curvature[eligible])])
        used_fallback = True

    info = {
        "明显曲率峰数量": int(len(peaks)),
        "有效明显曲率峰数量": int(len(valid_peaks)),
        "曲率峰储能时长下限(h)": min_peak_y,
        "是否回退到最大曲率": used_fallback,
    }
    point = (float(x[idx]), float(y[idx]))
    return (*point, info) if return_info else point


def find_storage_duration_column(df):
    candidates = []
    for col in df.columns:
        col_text = str(col).strip().lower()
        if any(token in col_text for token in ("hour", "duration", "时长", "小时")):
            candidates.append(col)
    if candidates:
        return candidates[0]

    extra_columns = [col for col in df.columns if col not in KNOWN_SOURCE_COLUMNS]
    numeric_extra_columns = [
        col for col in extra_columns
        if pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    if len(numeric_extra_columns) == 1:
        return numeric_extra_columns[0]

    raise ValueError(
        "Could not identify the storage-duration column. "
        f"Extra numeric columns: {numeric_extra_columns}"
    )


def load_curve_data(scenario_name, data_file):
    if not data_file.is_file():
        raise FileNotFoundError(f"Input file not found: {data_file}")

    df = pd.read_excel(data_file, sheet_name=0)
    df.columns = df.columns.astype(str).str.strip()
    duration_column = find_storage_duration_column(df)
    required_columns = ["cons_target_%", "energy_pen_%", duration_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{scenario_name}: missing required columns: {missing}")

    df = df[required_columns].copy()
    df["cons_target_%"] = normalize_percent(df["cons_target_%"])
    df["energy_pen_%"] = normalize_percent(df["energy_pen_%"])
    df[Y_COLUMN] = pd.to_numeric(df[duration_column], errors="coerce")
    if duration_column != Y_COLUMN:
        df = df.drop(columns=[duration_column])
    df = df.dropna(subset=["cons_target_%", "energy_pen_%", Y_COLUMN])

    plot_df = (
        df.groupby(["cons_target_%", "energy_pen_%"], as_index=False)[Y_COLUMN]
        .mean()
        .sort_values(["cons_target_%", "energy_pen_%"])
    )
    targets = sorted(
        int(round(t)) for t in plot_df["cons_target_%"].unique() if round(t) != 50
    )

    curve_data = {}
    raw_data = {}
    knee_data = {}
    knee_rows = []

    for target in targets:
        sub = plot_df[
            np.isclose(plot_df["cons_target_%"], target)
        ].sort_values("energy_pen_%")
        x_raw = sub["energy_pen_%"].to_numpy(float)
        y_raw = sub[Y_COLUMN].to_numpy(float)
        if len(x_raw) == 0:
            continue

        xs, ys = get_smooth_xy(x_raw, y_raw)
        x_line, y_line = clip_curve_by_y_min(xs, ys, MIN_STORAGE_HOURS)
        x_line, y_line = clip_curve_by_y_cap(x_line, y_line, Y_CAP)
        if len(x_line) == 0:
            continue

        growth_start_x = find_growth_start(x_raw, y_raw)
        if growth_start_x is None:
            print(f"  Warning: {target}% has no sustained growth after its initial plateau")
            continue
        knee_x, knee_y = find_max_curvature(x_line, y_line, growth_start_x)

        curve_data[target] = (x_line, y_line)
        raw_data[target] = (x_raw, y_raw, COLOR_MAP.get(target, "#767676"))
        knee_data[target] = (knee_x, knee_y)
        knee_rows.append({
            "场景": scenario_name,
            "消纳率目标(%)": target,
            "增长起点能量渗透率(%)": growth_start_x,
            "曲率最大_能量渗透率(%)": knee_x,
            "曲率最大_储能时长(h)": knee_y,
        })

    return {
        "scenario": scenario_name,
        "targets": sorted(curve_data),
        "curve_data": curve_data,
        "raw_data": raw_data,
        "knee_data": knee_data,
        "knee_rows": knee_rows,
    }


def draw_panel(ax, panel, labeled=False, panel_label=None):
    targets = panel["targets"]
    for target in targets:
        x_line, y_line = panel["curve_data"][target]
        x_raw, y_raw, color = panel["raw_data"][target]
        xk, yk = panel["knee_data"][target]
        raw_mask = (y_raw >= MIN_STORAGE_HOURS) & (y_raw <= Y_CAP)

        ax.scatter(
            x_raw[raw_mask],
            y_raw[raw_mask],
            s=4,
            color=color,
            edgecolors="black",
            alpha=0.28,
            zorder=3,
            linewidths=0.3,
        )
        ax.plot(x_line, y_line, color=color, linewidth=1.15, zorder=4)

        if xk is not None:
            ax.plot(
                xk,
                yk,
                linestyle="none",
                marker=MAX_CURVATURE_MARKER["marker"],
                markersize=MAX_CURVATURE_MARKER["markersize"],
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=10,
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y_CAP)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, Y_CAP + 0.1, 40))
    ax.tick_params(
        axis="both",
        which="major",
        length=2.5,
        width=0.6,
        labelsize=6,
        direction="out",
        pad=2,
    )
    if labeled:
        row_y = np.linspace(160, 80, len(targets))
        for target, y_row in zip(targets, row_y):
            xk, yk = panel["knee_data"][target]
            if xk is None:
                continue
            color = COLOR_MAP.get(target, "#767676")
            ax.plot([38, xk, xk], [y_row, y_row, yk], color=color,
                    linewidth=0.55, linestyle=(0, (2.0, 2.0)), alpha=0.75,
                    zorder=2)
        ax.set_title(panel["scenario"], fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("Energy penetration rate (%)", fontsize=7)
        ax.text(
            -0.09, 1.03, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom",
        )
        # Compact in-panel maximum-curvature coordinate block.
        ax.text(0.045, 0.955, "PEN.      DUR.", transform=ax.transAxes,
                fontsize=6.6, fontweight="bold", ha="left", va="top")
        for target, y_row in zip(targets, row_y):
            xk, yk = panel["knee_data"][target]
            if xk is None:
                continue
            color = COLOR_MAP.get(target, "#767676")
            ax.text(4.5, y_row, f"{xk:.1f}%", fontsize=6.2, color=color,
                    ha="left", va="center")
            ax.text(23.0, y_row, f"{yk:.1f} h", fontsize=6.2, color=color,
                    ha="left", va="center")
    else:
        # Pure clean version: retain axes/tick marks, but remove every text label.
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def draw_figure(panels, labeled=False):
    figure_height = 5.25 if labeled else 4.25
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.25 * len(panels), figure_height), sharey=True
    )
    if len(panels) == 1:
        axes = [axes]
    panel_labels = [chr(ord("a") + idx) for idx in range(len(panels))]
    for ax, panel, panel_label in zip(axes, panels, panel_labels):
        draw_panel(ax, panel, labeled=labeled, panel_label=panel_label)

    if labeled:
        axes[0].set_ylabel(Y_LABEL, fontsize=7)
        all_targets = sorted({target for panel in panels for target in panel["targets"]})
        target_handles = [
            Line2D([0], [0], color=COLOR_MAP.get(target, "#767676"), lw=1.4,
                   label=f"{target}%")
            for target in all_targets
        ]
        knee_handle = Line2D(
            [0], [0], linestyle="none", marker=MAX_CURVATURE_MARKER["marker"],
            markersize=MAX_CURVATURE_MARKER["markersize"], markerfacecolor="black",
            markeredgecolor="black", markeredgewidth=0.9,
            label="Maximum curvature",
        )
        fig.legend(
            handles=target_handles, loc="lower center", ncol=len(target_handles),
            fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.070),
            handletextpad=0.4, columnspacing=1.25,
            title="Self-consumption target", title_fontsize=6,
        )
        fig.legend(
            handles=[knee_handle], loc="lower center", ncol=1, fontsize=6,
            frameon=False, bbox_to_anchor=(0.5, -0.010), handletextpad=0.5,
        )
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.90, wspace=0.22)
    else:
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.98, wspace=0.10)
    return fig


def main():
    panels = []
    for scenario_name, data_file in SCENARIOS:
        print(f"Reading: {data_file}")
        panels.append(load_curve_data(scenario_name, data_file))

    knee_results = pd.DataFrame([
        row for panel in panels for row in panel["knee_rows"]
    ])
    knee_results.to_csv(OUTPUT_KNEE_CSV, index=False, encoding="utf-8-sig")
    print("\nMaximum-curvature results:")
    print(knee_results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved: {OUTPUT_KNEE_CSV}")

    fig_labeled = draw_figure(panels, labeled=True)
    fig_labeled.savefig(OUTPUT_LABELED_SVG, bbox_inches="tight")
    plt.close(fig_labeled)
    print(f"Saved: {OUTPUT_LABELED_SVG}")

    fig_clean_comparison = draw_figure(panels, labeled=False)
    fig_clean_comparison.savefig(OUTPUT_CLEAN_COMPARISON_SVG)
    plt.close(fig_clean_comparison)
    print(f"Saved: {OUTPUT_CLEAN_COMPARISON_SVG}")

    for panel in panels:
        fig_clean = draw_figure([panel], labeled=False)
        output_file = CLEAN_DIR / f"{panel['scenario']}.svg"
        # Keep the clean panel on its fixed 1:1 canvas for direct PPT placement.
        fig_clean.savefig(output_file)
        plt.close(fig_clean)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
