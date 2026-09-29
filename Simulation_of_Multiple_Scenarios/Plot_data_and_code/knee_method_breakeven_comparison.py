"""Plot three flexibility turning-point methods relative to economic breakeven.

Adapt the grouped dot-and-span layout of the original method-comparison plot.
Use the shared comparison module for all threshold loading and calculation.
The shared module uses baseline economic data (price factor 1.0).

Figure contract
---------------
Claim: most turning-point estimates precede economic breakeven; filled symbols
identify the exceptions, where breakeven occurs at a lower penetration.
Evidence: one quantitative comparison panel, four scenarios, five targets and
three methods. The main x axis is p_turning - p_breakeven, in percentage points.
The grey horizontal line is the range across methods, not an uncertainty bar.
When breakeven is missing, assume 100% for positioning only: x = p_turning - 100.
These markers have dashed outlines and a solid connector. The actual economic
threshold stays missing and these positions do not enter ordering statistics.
Small fixed vertical offsets distinguish coincident symbols without changing x.
Exports: 7.2 x 6.1 inch SVG/PDF with editable text, PNG preview, text-free SVG,
source CSV and caption. All graphics are generated with Python/matplotlib.

Run from this directory:
    python -B knee_method_breakeven_comparison.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import compare_knee_breakeven_order as comparison


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results" / "knee_breakeven_order"
CLEAN_DIR = RESULT_DIR / "clean"
BASENAME = "knee_method_breakeven_deviations"
OUTPUT_CSV = RESULT_DIR / f"{BASENAME}_plot_data.csv"
DEVIATION_COLUMN = "turning_minus_breakeven_pp"
HIGHLIGHT_COLUMN = "breakeven_precedes_turning_point"
TOLERANCE_PP = comparison.DEFAULT_TOLERANCE_PP
ASSUMED_BREAKEVEN_PERCENT = 100.0

# Preserve the original figure's scenario order, colours and marker shapes.
SCENARIOS = ["Rural", "Commercial", "Industrial", "Island"]
TARGETS = comparison.TARGETS
METHODS = list(comparison.METHODS)
DRAW_METHODS = ["maximum_curvature", "L_method", "kneedle"]
METHOD_LABELS = {
    "maximum_curvature": "Maximum curvature",
    "kneedle": "Kneedle",
    "L_method": "L_method",
}
METHOD_STYLES = {
    "maximum_curvature": {"marker": "o", "color": "#D9822B", "offset": 0.00, "size": 24},
    "kneedle": {"marker": "^", "color": "#377EB8", "offset": -0.13, "size": 34},
    "L_method": {"marker": "s", "color": "#6B7280", "offset": 0.13, "size": 20},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
    """Recalculate with the existing comparison code, then reverse its sign."""
    knees = comparison.load_knees()
    economics = comparison.calculate_economic_thresholds()
    df = comparison.compare_thresholds(knees, economics, TOLERANCE_PP)
    # The comparison script uses p_e - p_f; this figure needs p_f - p_e.
    df[DEVIATION_COLUMN] = -df["economic_minus_flexibility_pp"]
    df[HIGHLIGHT_COLUMN] = df["order"].eq("behind")
    no_breakeven = df["economic_crossover_status"].eq("no_crossover")
    df["breakeven_assumed_for_plot"] = no_breakeven
    df["plot_breakeven_penetration_%"] = df["economic_breakeven_penetration_%"].where(
        ~no_breakeven, ASSUMED_BREAKEVEN_PERCENT
    )
    df["plot_x_quantity"] = np.where(
        no_breakeven, "turning_minus_assumed_100_percent_pp", "turning_minus_breakeven_pp"
    )
    df["plot_x_value"] = (
        df["flexibility_knee_penetration_%"] - df["plot_breakeven_penetration_%"]
    )
    df["vertical_marker_offset"] = df["method"].map(
        {method: style["offset"] for method, style in METHOD_STYLES.items()}
    )
    expected = pd.MultiIndex.from_product(
        [SCENARIOS, TARGETS, METHODS],
        names=[*comparison.KEYS, "method"],
    )
    indexed = df.set_index([*comparison.KEYS, "method"])
    if not indexed.index.is_unique or set(indexed.index) != set(expected):
        raise ValueError("Expected exactly one record per scenario, target and method")
    df = indexed.reindex(expected).reset_index()
    if not np.array_equal(
        df[HIGHLIGHT_COLUMN].to_numpy(), (df[DEVIATION_COLUMN] > TOLERANCE_PP).to_numpy()
    ):
        raise ValueError("Breakeven-first flags disagree with plotted deviation signs")
    return df


def axis_limit(df):
    """Use symmetric limits with space for the numeric point annotations."""
    finite = df.loc[np.isfinite(df["plot_x_value"]), "plot_x_value"]
    if finite.empty:
        raise ValueError("No finite threshold differences are available to plot")
    return max(5.0, 5.0 * np.ceil((finite.abs().max() + 1.5) / 5.0))


def draw_figure(df, labeled=True):
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    group_gap = 1.15
    group_size = len(TARGETS)
    limit = axis_limit(df)
    lower_limit = (len(SCENARIOS) - 1) * (group_size + group_gap) + group_size - 0.45
    ax.set_xlim(-limit, limit)
    ax.set_ylim(lower_limit, -1.15)
    all_y, all_target_labels = [], []
    band_colors = ["#F6F8FA", "#FFFFFF", "#F6F8FA", "#FFFFFF"]

    for group_index, scenario in enumerate(SCENARIOS):
        sub = df[df["scenario"] == scenario]
        start = group_index * (group_size + group_gap)
        y_base = start + np.arange(group_size, dtype=float)
        all_y.extend(y_base)
        all_target_labels.extend([f"{target}%" for target in TARGETS])
        ax.axhspan(
            start - 0.48, start + group_size - 0.52,
            color=band_colors[group_index], zorder=-3,
        )

        # Solid spans also connect positions based on the assumed 100% breakeven.
        for target, y in zip(TARGETS, y_base):
            row = sub[sub["consumption_target_%"] == target]
            values = row["plot_x_value"].to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite) >= 2:
                span, = ax.plot(
                    [finite.min(), finite.max()], [y, y],
                    color="#B7BEC7", linewidth=2.2, linestyle="-",
                    solid_capstyle="round", zorder=1,
                )
                span.set_gid(f"method_span_{scenario}_{target}")
            if labeled and row["breakeven_assumed_for_plot"].all():
                ax.text(
                    0.54, y,
                    "No economic breakeven in analysed domain",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=5.8, color="#626974", fontstyle="italic",
                )
            elif labeled and row["order"].eq("missing_flexibility_knee").any():
                missing = row.loc[row["order"] == "missing_flexibility_knee", "method"]
                ax.text(
                    0.04, y + 0.36,
                    "Unavailable: " + ", ".join(METHOD_LABELS[m] for m in missing),
                    transform=ax.get_yaxis_transform(), fontsize=5.5, color="#626974",
                )

        for method in DRAW_METHODS:
            style = METHOD_STYLES[method]
            rows = sub[sub["method"] == method].set_index("consumption_target_%").reindex(TARGETS)
            x_values = rows["plot_x_value"].to_numpy(float)
            y_values = y_base + style["offset"]
            valid = np.isfinite(x_values)
            highlighted = rows[HIGHLIGHT_COLUMN].to_numpy(bool) & valid
            assumed = rows["breakeven_assumed_for_plot"].to_numpy(bool) & valid
            for mask, filled, dashed in [
                (valid & ~highlighted & ~assumed, False, False),
                (highlighted & ~assumed, True, False),
                (assumed, False, True),
            ]:
                if not mask.any():
                    continue
                points = ax.scatter(
                    x_values[mask], y_values[mask],
                    s=style["size"], marker=style["marker"],
                    facecolor=style["color"] if filled else "white",
                    edgecolor=style["color"], linewidth=1.15, zorder=4,
                    linestyles=(0, (1.2, 1.0)) if dashed else "solid",
                )
                kind = "assumed_breakeven" if dashed else ("breakeven_first" if filled else "other")
                points.set_gid(f"points_{scenario}_{method}_{kind}")

            if labeled:
                # Label both industrial symbols even when their x values coincide.
                dy = -11 if method == "L_method" else 8
                for x, y in zip(x_values[highlighted], y_values[highlighted]):
                    label = ax.annotate(
                        f"+{x:.2f}", (x, y), xytext=(0, dy),
                        textcoords="offset points", ha="center",
                        va="top" if dy < 0 else "bottom",
                        fontsize=5.8, fontweight="bold", color=style["color"],
                        zorder=6,
                    )
                    label.set_gid(f"highlight_label_{scenario}_{method}")

        if labeled:
            ax.text(
                -limit + 0.5, start - 0.70, scenario,
                fontsize=7.3, fontweight="bold", ha="left", va="center",
            )

    if labeled:
        n_highlight = int(df[HIGHLIGHT_COLUMN].sum())
        n_valid = int(np.isfinite(df[DEVIATION_COLUMN]).sum())
        ax.text(
            limit - 0.5, -0.70,
            f"Breakeven first: {n_highlight} / {n_valid} method estimates",
            fontsize=5.8, color="#525A65", ha="right", va="center",
        )

    ax.set_xticks(np.linspace(-limit, limit, 7))
    ax.set_yticks(all_y)
    ax.axvline(0, color="#40464F", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
    ax.grid(axis="x", color="#E3E7EB", linewidth=0.55, zorder=-1)
    ax.tick_params(axis="both", length=2.5, width=0.6, labelsize=6)

    if labeled:
        ax.set_yticklabels(all_target_labels)
        ax.set_xlabel(
            "Turning-point penetration - breakeven penetration (percentage points)",
            fontsize=7,
        )
        ax.set_ylabel("Self-consumption target", fontsize=7)
        ax.text(
            0.24, 1.014, "Turning point precedes breakeven",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.1, color="#525A65",
        )
        ax.text(
            0.76, 1.014, "Breakeven precedes turning point",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.1, color="#525A65",
        )
        method_handles = [
            Line2D(
                [0], [0], linestyle="none", marker=METHOD_STYLES[method]["marker"],
                markersize=5, markerfacecolor="white",
                markeredgecolor=METHOD_STYLES[method]["color"], markeredgewidth=1.15,
                label=METHOD_LABELS[method],
            ) for method in METHODS
        ]
        method_handles.append(Line2D(
            [0], [0], color="#B7BEC7", linewidth=2.2, label="Three-method span",
        ))
        fig.legend(
            handles=method_handles, loc="upper center", ncol=4, fontsize=6.2,
            bbox_to_anchor=(0.5, 0.995), handletextpad=0.5, columnspacing=1.4,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], linestyle="none", marker="o", markersize=4.5,
                       color="#40464F", label="Filled: breakeven precedes turning point"),
                Line2D([0], [0], color="#40464F", linewidth=1.0, linestyle=(0, (3, 2)),
                       label="Economic breakeven (zero)"),
            ],
            loc="upper center", ncol=2, fontsize=6,
            bbox_to_anchor=(0.5, 0.956), handletextpad=0.6, columnspacing=1.8,
        )
    else:
        ax.tick_params(labelbottom=False, labelleft=False)
    # Keep identical canvas geometry in the labelled and text-free versions.
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.10, top=0.89)
    return fig


def build_caption(df):
    n_valid = int(np.isfinite(df[DEVIATION_COLUMN]).sum())
    n_highlight = int(df[HIGHLIGHT_COLUMN].sum())
    lines = [
        "Flexibility turning-point deviations from economic breakeven.",
        "Symbols on the main axis show turning-point penetration minus economic breakeven penetration, "
        "in percentage points, for one method at the same scenario and consumption target. "
        "Negative values indicate an earlier flexibility turning point; positive values "
        "indicate an earlier economic breakeven. The dashed vertical line is zero deviation.",
        "Circle, triangle and square denote maximum curvature, Kneedle and L_method, "
        "respectively. Filled symbols and numeric labels identify economic breakeven "
        "preceding the turning point. Small vertical offsets separate the symbols within "
        "each target row; the x coordinates are unaltered. Grey lines connect the minimum "
        "and maximum estimates across methods; these are not confidence intervals.",
        f"The figure contains {n_valid} comparable method estimates; economic breakeven "
        f"precedes {n_highlight} of them. Estimates from different methods are not "
        "independent scenario replicates. No inferential test is performed.",
        "Thresholds use the existing comparison workflow: saved flexibility knees "
        "from the 200 h storage-duration analysis and economic crossings recomputed "
        "from baseline data with price factor 1.0, penetration 0--99.9%, and minimum "
        "off-grid duration capped at 500 h. The two duration limits retain their "
        "respective original definitions.",
        f"Differences within {TOLERANCE_PP:g} percentage points are treated as coincident. "
        "For a target without an economic crossing in the analysed domain, breakeven "
        "is assumed to be 100% for positioning only: x = turning-point penetration "
        "minus 100%. These three symbols have dashed outlines; their connector "
        "remains solid. The source economic threshold stays missing, and the assumed "
        "positions are excluded from the comparable count and ordering statistics.",
        "Highlighted estimates (turning point minus breakeven, percentage points):",
    ]
    for _, row in df[df[HIGHLIGHT_COLUMN]].iterrows():
        lines.append(
            f'{row["scenario"]}, {row["consumption_target_%"]:g}%, '
            f'{METHOD_LABELS[row["method"]]}: +{row[DEVIATION_COLUMN]:.4f}'
        )
    return "\n\n".join(lines) + "\n"


def main():
    df = load_data()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.10f")

    fig = draw_figure(df, labeled=True)
    for suffix in ("svg", "pdf", "png"):
        path = RESULT_DIR / f"{BASENAME}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)

    clean = draw_figure(df, labeled=False)
    clean_path = CLEAN_DIR / f"{BASENAME}.svg"
    clean.savefig(clean_path, facecolor="white")
    plt.close(clean)
    caption = build_caption(df)
    (RESULT_DIR / f"{BASENAME}_caption.txt").write_text(caption, encoding="utf-8")
    print(f"Saved: {clean_path}")
    print(f"Saved: {OUTPUT_CSV}")
    print(caption)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
