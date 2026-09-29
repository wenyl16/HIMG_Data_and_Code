"""Compare flexibility knees with economic crossovers along energy penetration.

Run from this directory:
    python -B compare_knee_breakeven_order.py

Dependencies: numpy, pandas, scipy and openpyxl (for Excel input).
Read the original three-method knee CSV used by knee_method_comparison.py.
Recalculate economic crossovers from factor=1.0 data in data/sensitivity.

Follow Economic Breakeven Point/plot_domestic_cases.py: match hydrogen LCOE
linearly by penetration, interpolate BESS-H2 cost differences and off-grid
duration with PCHIP (500 points), and detect segment-level zero crossings
within 0--99.9% penetration and 500 h off-grid duration. No points are removed.
Retain np.interp endpoint values outside the hydrogen penetration range.

Count each scenario-target-method pair once. Reject multiple distinct
economic crossovers for the same pair. Report missing crossovers separately
without inferring ordering outside the analysed domain. Save outputs in
results/knee_breakeven_order/ relative to this file.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


BASE_DIR = Path(__file__).resolve().parent
KNEE_FILE = (
    BASE_DIR / "Flexibility Turning Point" / "results"
    / "domestic_flexibility_knee_method_differences.csv"
)
DATA_DIR = BASE_DIR / "Economic Breakeven Point" / "data" / "sensitivity"
OUTPUT_DIR = BASE_DIR / "results" / "knee_breakeven_order"

SCENARIOS = {
    "Industrial": "Industrial", "Commercial": "Commercial", "Rural": "Rural", "Island": "Island",
}
TARGETS = [60, 70, 80, 90, 100]
METHODS = {
    "maximum_curvature": "Maximum curvature", "kneedle": "Kneedle", "L_method": "L_method",
}
KEYS = ["scenario", "consumption_target_%"]
PEN_MIN, PEN_MAX, OFFGRID_MAX = 0.0, 99.9, 500.0
PRICE_FACTOR = 1.0
DEFAULT_TOLERANCE_PP = 1e-4  # Source turning-point CSV stores four decimal places.
ORDERS = {
    "ahead": "Ahead", "behind": "Behind", "coincident": "Coincident",
    "no_economic_crossover": "No economic crossover in the analysed domain",
    "missing_flexibility_knee": "Missing flexibility knee",
}


def load_knees(path=KNEE_FILE):
    """Read the English turning-point schema and validate pairing keys."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    columns = KEYS + [f"{method}_penetration_%" for method in METHODS]
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{path}: Missing columns: {sorted(missing)}")
    if not df.columns.is_unique:
        raise ValueError(f"{path}: Duplicate column names")
    df = df[columns].copy()
    for column in columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: Duplicate scenario-target pairs prevent one-to-one matching")
    expected = pd.MultiIndex.from_product([SCENARIOS, TARGETS], names=KEYS)
    actual = pd.MultiIndex.from_frame(df[KEYS])
    if set(actual) != set(expected):
        raise ValueError(
            f"{path}: Unexpected scenario-target combinations; missing {list(expected.difference(actual))}; "
            f"extra {list(actual.difference(expected))}"
        )
    for column in columns[2:]:
        present = df[column].dropna()
        if not present.between(0, 100).all():
            raise ValueError(f"{path}: {column} must be a penetration percentage between 0 and 100")
    return df.set_index(KEYS).reindex(expected).reset_index()


def load_baseline(path, factor_column):
    """Keep price factor 1.0; reject missing or invalid input data."""
    raw = pd.read_excel(path)
    columns = ["cons_target_%", "min_offgrid_hours", "energy_pen_%", "LCOE_USD_per_kWh"]
    missing = set([factor_column] + columns).difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: Missing columns: {sorted(missing)}")
    factor = pd.to_numeric(raw[factor_column], errors="raise")
    df = raw.loc[np.isclose(factor, PRICE_FACTOR), columns].copy()
    if df.empty:
        raise ValueError(f"{path}: No data for factor={PRICE_FACTOR}")
    df = df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: Baseline data contain missing or infinite values")
    missing_targets = set(TARGETS).difference(df["cons_target_%"])
    if missing_targets:
        raise ValueError(f"{path}: Missing targets: {sorted(missing_targets)}")
    return df


def load_matched(scenario, data_dir=DATA_DIR):
    """Match H2 LCOE by target and penetration, as in the economic plotting script."""
    folder = data_dir / scenario.lower()
    battery = load_baseline(folder / "battery.xlsx", "bat_price_factor")
    hydrogen = load_baseline(folder / "hydrogen.xlsx", "storage_factor")
    battery = battery[battery["cons_target_%"].isin(TARGETS)].copy()
    battery["h2_LCOE"] = np.nan
    for target in TARGETS:
        mask = battery["cons_target_%"] == target
        h2 = (
            hydrogen.loc[hydrogen["cons_target_%"] == target]
            .groupby("energy_pen_%")["LCOE_USD_per_kWh"].mean().sort_index()
        )
        if len(h2) < 2:
            raise ValueError(f"{scenario}, {target}%: Insufficient H2 penetration points for matching")
        battery.loc[mask, "h2_LCOE"] = np.interp(
            battery.loc[mask, "energy_pen_%"].to_numpy(float),
            h2.index.to_numpy(float), h2.to_numpy(float),
        )
    battery["cost_diff"] = battery["LCOE_USD_per_kWh"] - battery["h2_LCOE"]
    return battery


def make_curve(subset):
    """Reproduce grouping, PCHIP interpolation and the 500 h cap."""
    subset = subset[subset["energy_pen_%"].between(PEN_MIN, PEN_MAX)]
    subset = subset[np.isfinite(subset["cost_diff"])]
    grouped = (
        subset.groupby("energy_pen_%", as_index=False)
        .agg({"min_offgrid_hours": "mean", "cost_diff": "mean"})
        .sort_values("energy_pen_%")
    )
    if len(grouped) < 3:
        raise ValueError("Fewer than three penetration points in the analysed domain")
    x = grouped["energy_pen_%"].to_numpy(float)
    y = grouped["min_offgrid_hours"].to_numpy(float)
    z = grouped["cost_diff"].to_numpy(float)
    if len(x) >= 4:
        dense_x = np.linspace(x[0], x[-1], 500)
        y, z = PchipInterpolator(x, y)(dense_x), PchipInterpolator(x, z)(dense_x)
        x = dense_x
    above = np.flatnonzero(y >= OFFGRID_MAX)
    if len(above):
        idx = int(above[0])
        if idx == 0:
            raise ValueError("Curve starts at the off-grid duration cap; insufficient analysed domain")
        if np.isclose(y[idx], OFFGRID_MAX):
            x, y, z = x[:idx + 1], y[:idx + 1], z[:idx + 1]
            y[-1] = OFFGRID_MAX
        else:
            fraction = (OFFGRID_MAX - y[idx - 1]) / (y[idx] - y[idx - 1])
            x = np.r_[x[:idx], x[idx - 1] + fraction * (x[idx] - x[idx - 1])]
            y = np.r_[y[:idx], OFFGRID_MAX]
            z = np.r_[z[:idx], z[idx - 1] + fraction * (z[idx] - z[idx - 1])]
    return x, y, z


def find_breakevens(x, y, z):
    """Detect zero crossings; deduplicate a zero shared by adjacent segments."""
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
            points.append((
                x[idx - 1] + fraction * (x[idx] - x[idx - 1]),
                y[idx - 1] + fraction * (y[idx] - y[idx - 1]),
            ))
    if np.isclose(z[-1], 0.0, atol=1e-12):
        points.append((x[-1], y[-1]))
    unique = []
    for point in points:
        if not unique or not np.isclose(point[0], unique[-1][0], atol=1e-9, rtol=0):
            unique.append(point)
    return unique


def calculate_economic_thresholds(data_dir=DATA_DIR):
    rows = []
    for scenario in SCENARIOS:
        matched = load_matched(scenario, data_dir)
        for target in TARGETS:
            try:
                curve = make_curve(matched[matched["cons_target_%"] == target])
            except ValueError as exc:
                raise ValueError(f"{scenario}, {target}%: {exc}") from exc
            points = find_breakevens(*curve)
            if len(points) > 1:
                raise ValueError(
                    f"{scenario}, {target}%: Multiple economic crossovers found: {points}; "
                    "define a selection rule before computing single-crossover statistics"
                )
            penetration, duration = points[0] if points else (np.nan, np.nan)
            rows.append({
                "scenario": scenario, "consumption_target_%": target,
                "economic_breakeven_penetration_%": penetration,
                "economic_breakeven_duration_h": duration,
                "economic_crossover_status": "crossover" if points else "no_crossover",
            })
    return pd.DataFrame(rows)


def compare_thresholds(knees, economics, tolerance_pp=DEFAULT_TOLERANCE_PP):
    """Delta p = economic crossover minus flexibility knee: positive means ahead."""
    if not np.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("Coincidence tolerance must be finite and nonnegative, in percentage points")
    merged = knees.merge(economics, on=KEYS, how="left", validate="one_to_one")
    if merged["economic_crossover_status"].isna().any():
        raise ValueError("Economic results are missing for some scenario-target pairs")
    rows = []
    for _, case in merged.iterrows():
        for method, label in METHODS.items():
            knee = case[f"{method}_penetration_%"]
            economic = case["economic_breakeven_penetration_%"]
            delta = economic - knee
            if not np.isfinite(knee):
                order = "missing_flexibility_knee"
            elif case["economic_crossover_status"] == "no_crossover":
                order = "no_economic_crossover"
            elif not np.isfinite(economic):
                raise ValueError("Economic status is crossover but its threshold is missing")
            elif abs(delta) <= tolerance_pp:
                order = "coincident"
            else:
                order = "ahead" if delta > 0 else "behind"
            rows.append({
                **{key: case[key] for key in KEYS},
                "scenario_label": SCENARIOS[case["scenario"]],
                "method": method, "method_label": label,
                "flexibility_knee_penetration_%": knee,
                "economic_breakeven_penetration_%": economic,
                "economic_breakeven_duration_h": case["economic_breakeven_duration_h"],
                "economic_minus_flexibility_pp": delta,
                "order": order, "order_label": ORDERS[order],
                "economic_crossover_status": case["economic_crossover_status"],
            })
    return pd.DataFrame(rows)


def summarize(details, group_columns):
    """Percentages use only cases with both thresholds available."""
    rows = []
    for key, group in details.groupby(group_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        counts = group["order"].value_counts()
        row = dict(zip(group_columns, key))
        row.update({f"{order}_count": int(counts.get(order, 0)) for order in ORDERS})
        valid = sum(row[f"{order}_count"] for order in ("ahead", "behind", "coincident"))
        row.update(total_cases=len(group), comparable_cases=valid)
        for order in ("ahead", "behind", "coincident"):
            row[f"{order}_percent_of_comparable"] = (
                100 * row[f"{order}_count"] / valid if valid else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(details, summary, tolerance_pp):
    lines = [
        "Ordering of flexibility knees and economic crossovers", "",
        f"Economic data: {DATA_DIR.relative_to(BASE_DIR).as_posix()}",
        "Use only bat_price_factor=1.0 in battery.xlsx and storage_factor=1.0 in hydrogen.xlsx.",
        f"Flexibility data: {KNEE_FILE.relative_to(BASE_DIR).as_posix()}",
        "Use the saved results from three knee methods with a 200 h storage-duration cap.",
        "Economic crossovers use the economic plotting algorithm, 0--99.9% penetration and a 500 h off-grid cap.",
        "Each duration cap retains its original threshold definition.", "",
        "Ordering is measured along the energy-penetration axis.",
        f"Delta p = economic crossover penetration - flexibility knee penetration; coincidence tolerance = {tolerance_pp:g} percentage points.",
        "Delta p > tolerance: flexibility knee ahead; Delta p < -tolerance: behind; otherwise coincident.",
        "Missing economic crossovers or flexibility knees are reported separately and excluded from ordering percentages.",
        "Each method uses comparable scenario-target pairs as its denominator; pooled methods count estimates, not independent scenarios.", "",
        "Method\tTotal cases\tComparable\tAhead\tBehind\tCoincident\tNo economic crossover\tMissing flexibility knee\tAhead (%)",
    ]
    for _, row in summary.iterrows():
        lines.append("\t".join(map(str, [
            METHODS[row["method"]], row["total_cases"], row["comparable_cases"],
            row["ahead_count"], row["behind_count"], row["coincident_count"],
            row["no_economic_crossover_count"], row["missing_flexibility_knee_count"],
            f'{row["ahead_percent_of_comparable"]:.2f}',
        ])))
    lines.extend(["", "Cases with a later flexibility knee (percentage points):"])
    behind = details[details["order"] == "behind"]
    if behind.empty:
        lines.append("None.")
    for _, row in behind.iterrows():
        lines.append(
            f'{row["scenario_label"]} / {row["consumption_target_%"]:g}% / {row["method_label"]}: '
            f'p_f={row["flexibility_knee_penetration_%"]:.4f}, '
            f'p_e={row["economic_breakeven_penetration_%"]:.4f}, '
            f'Delta p={row["economic_minus_flexibility_pp"]:.4f}'
        )
    no_cross = details.loc[details["economic_crossover_status"] == "no_crossover", KEYS].drop_duplicates()
    lines.extend(["", "Cases without an economic crossover in the analysed domain:"])
    if no_cross.empty:
        lines.append("None.")
    for _, row in no_cross.iterrows():
        lines.append(f'{SCENARIOS[row["scenario"]]} / {row["consumption_target_%"]:g}%')
    lines.extend([
        "", "Output files:",
        "comparison_details.csv: thresholds, differences and ordering by scenario, target and method.",
        "summary_by_method.csv: total, comparable and category counts and comparable-case percentages by method.",
        "summary_by_scenario_method.csv: the same statistics by scenario and method.",
        "summary.txt: analysis definitions and main results.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance-pp", type=float, default=DEFAULT_TOLERANCE_PP,
                        help="Absolute coincidence tolerance in percentage points (default: 0.0001)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_pp) or args.tolerance_pp < 0:
        parser.error("--tolerance-pp must be finite and nonnegative")
    knees = load_knees()
    economics = calculate_economic_thresholds()
    details = compare_thresholds(knees, economics, args.tolerance_pp)
    summary = summarize(details, ["method"])
    by_scenario = summarize(details, ["scenario", "method"])
    report = build_report(details, summary, args.tolerance_pp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [
        ("comparison_details.csv", details),
        ("summary_by_method.csv", summary),
        ("summary_by_scenario_method.csv", by_scenario),
    ]:
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig", float_format="%.10f")
    (args.output_dir / "summary.txt").write_text(report, encoding="utf-8-sig")
    print(report)
    print(f"Results saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
