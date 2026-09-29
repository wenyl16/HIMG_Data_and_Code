"""统计三种灵活性拐点相对经济交叉点的先后顺序（按能源渗透率比较）。

在本脚本所在目录运行：
    python -B compare_knee_breakeven_order.py

依赖：numpy、pandas、scipy、openpyxl（仅用于读取 Excel）。
灵活性拐点读取 knee_method_comparison.py 所用的原始拐点 CSV，
不是图中的相对三方法均值偏差。经济交叉点每次从 data/sensitivity
中的 factor=1.0 数据重算经济交叉点。

经济计算沿用 Economic Breakeven Point/plot_domestic_cases.py 的口径：
按渗透率线性匹配氢储能 LCOE；对 BESS-H2 成本差和离网时长作 PCHIP
插值（500 点）；限定渗透率 0--99.9%、离网时长不超过 500 h；在线段
内插值寻找成本差零点。原脚本的删点配置为空，本脚本同样不删点。
保留原 np.interp 的范围外端点取值行为，不另外引入外推模型。

每个“场景 × 消纳率目标 × 方法”只统计一次。若同一组合出现多个
不同经济交叉点，报错提示，避免擅自选点或重复计数。无交叉点单列，
不推断域外顺序。输出保存到本目录的 results/knee_breakeven_order/。
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
    "Industrial": "工业", "Commercial": "商业", "Rural": "乡村", "Island": "海岛",
}
TARGETS = [60, 70, 80, 90, 100]
METHODS = {
    "maximum_curvature": "最大曲率法", "kneedle": "Kneedle", "L_method": "L_method",
}
KEYS = ["scenario", "consumption_target_%"]
PEN_MIN, PEN_MAX, OFFGRID_MAX = 0.0, 99.9, 500.0
PRICE_FACTOR = 1.0
DEFAULT_TOLERANCE_PP = 1e-4  # 源拐点 CSV 精确至小数点后四位。
ORDERS = {
    "ahead": "领先", "behind": "滞后", "coincident": "重合",
    "no_economic_crossover": "分析域内无经济交叉点",
    "missing_flexibility_knee": "缺少灵活性拐点",
}


def load_knees(path=KNEE_FILE):
    """兼容现有英文数据及原绘图脚本中的中文字段，检查配对键。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={
        "场景": "scenario", "消纳率目标(%)": "consumption_target_%",
        "曲率最大_能量渗透率(%)": "maximum_curvature_penetration_%",
        "kneedle_能量渗透率(%)": "kneedle_penetration_%",
        "L_method_能量渗透率(%)": "L_method_penetration_%",
    })
    columns = KEYS + [f"{method}_penetration_%" for method in METHODS]
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{path}: 缺少列 {sorted(missing)}")
    if not df.columns.is_unique:
        raise ValueError(f"{path}: 存在重复列名")
    df = df[columns].copy()
    df["scenario"] = df["scenario"].replace({v: k for k, v in SCENARIOS.items()})
    for column in columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: 场景和消纳率目标存在重复，不能一一配对")
    expected = pd.MultiIndex.from_product([SCENARIOS, TARGETS], names=KEYS)
    actual = pd.MultiIndex.from_frame(df[KEYS])
    if set(actual) != set(expected):
        raise ValueError(
            f"{path}: 场景/目标组合不符；缺失 {list(expected.difference(actual))}；"
            f"多出 {list(actual.difference(expected))}"
        )
    for column in columns[2:]:
        present = df[column].dropna()
        if not present.between(0, 100).all():
            raise ValueError(f"{path}: {column} 应为 0--100 的渗透率百分数")
    return df.set_index(KEYS).reindex(expected).reset_index()


def load_baseline(path, factor_column):
    """仅保留价格因子 1.0；缺失/坏数据报错，不伪装成无交叉点。"""
    raw = pd.read_excel(path)
    columns = ["cons_target_%", "min_offgrid_hours", "energy_pen_%", "LCOE_USD_per_kWh"]
    missing = set([factor_column] + columns).difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: 缺少列 {sorted(missing)}")
    factor = pd.to_numeric(raw[factor_column], errors="raise")
    df = raw.loc[np.isclose(factor, PRICE_FACTOR), columns].copy()
    if df.empty:
        raise ValueError(f"{path}: 缺少 factor={PRICE_FACTOR} 数据")
    df = df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: 基准数据包含缺失值或无穷值")
    missing_targets = set(TARGETS).difference(df["cons_target_%"])
    if missing_targets:
        raise ValueError(f"{path}: 缺少目标 {sorted(missing_targets)}")
    return df


def load_matched(scenario, data_dir=DATA_DIR):
    """与原经济绘图脚本相同：按目标、渗透率匹配 H2 LCOE。"""
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
            raise ValueError(f"{scenario}, {target}%: H2 渗透率点不足，无法匹配")
        battery.loc[mask, "h2_LCOE"] = np.interp(
            battery.loc[mask, "energy_pen_%"].to_numpy(float),
            h2.index.to_numpy(float), h2.to_numpy(float),
        )
    battery["cost_diff"] = battery["LCOE_USD_per_kWh"] - battery["h2_LCOE"]
    return battery


def make_curve(subset):
    """复现原脚本的分组、PCHIP 和 500 h 截断，保留数值口径。"""
    subset = subset[subset["energy_pen_%"].between(PEN_MIN, PEN_MAX)]
    subset = subset[np.isfinite(subset["cost_diff"])]
    grouped = (
        subset.groupby("energy_pen_%", as_index=False)
        .agg({"min_offgrid_hours": "mean", "cost_diff": "mean"})
        .sort_values("energy_pen_%")
    )
    if len(grouped) < 3:
        raise ValueError("分析域内不足三个渗透率点，不能计算经济交叉点")
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
            raise ValueError("曲线起点已达到离网时长上限，分析域内曲线不足")
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
    """原脚本的零点检测；仅去除相邻线段共享的同一个零点。"""
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
                    f"{scenario}, {target}%: 存在多个经济交叉点 {points}；"
                    "请明确选点规则后再统计，不能按单一交叉点重复计数"
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
    """Delta p = 经济交叉点 - 灵活性拐点：正值领先，负值滞后。"""
    if not np.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("重合容差必须是非负有限数，单位为百分点")
    merged = knees.merge(economics, on=KEYS, how="left", validate="one_to_one")
    if merged["economic_crossover_status"].isna().any():
        raise ValueError("部分场景/目标缺少经济计算结果")
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
                raise ValueError("经济状态为 crossover，但交叉点数值缺失")
            elif abs(delta) <= tolerance_pp:
                order = "coincident"
            else:
                order = "ahead" if delta > 0 else "behind"
            rows.append({
                **{key: case[key] for key in KEYS},
                "scenario_zh": SCENARIOS[case["scenario"]],
                "method": method, "method_label": label,
                "flexibility_knee_penetration_%": knee,
                "economic_breakeven_penetration_%": economic,
                "economic_breakeven_duration_h": case["economic_breakeven_duration_h"],
                "economic_minus_flexibility_pp": delta,
                "order": order, "order_zh": ORDERS[order],
                "economic_crossover_status": case["economic_crossover_status"],
            })
    return pd.DataFrame(rows)


def summarize(details, group_columns):
    """占比的分母仅包含两类阈值均存在的有效比较。"""
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
        "灵活性拐点与经济交叉点的先后顺序", "",
        f"经济数据：{DATA_DIR}",
        "仅使用 battery.xlsx 的 bat_price_factor=1.0 和 hydrogen.xlsx 的 storage_factor=1.0。",
        f"灵活性数据：{KNEE_FILE}",
        "灵活性拐点沿用已保存的三方法结果，不重新识别；该结果使用 200 h 储能时长上限。",
        "经济计算沿用原经济绘图算法，渗透率 0--99.9%，离网时长上限 500 h。",
        "两处时长上限属于各自原有阈值定义，不在本次比较中改动。", "",
        "比较轴为能源渗透率，不是时间或离网时长。",
        f"Delta p = 经济交叉点渗透率 - 灵活性拐点渗透率；重合容差 = {tolerance_pp:g} 个百分点。",
        "Delta p > 容差：灵活性拐点领先；Delta p < -容差：滞后；绝对值不超过容差：重合。",
        "无经济交叉点或缺少灵活性拐点单独列出，均不纳入领先/滞后/重合及其占比分母。",
        "每种方法的分母是有效的场景×目标组合数；跨方法合计是方法判断次数，不是独立场景数。", "",
        "方法\t总组合\t有效比较\t领先\t滞后\t重合\t无经济交叉点\t缺少灵活性拐点\t领先占比(%)",
    ]
    for _, row in summary.iterrows():
        lines.append("\t".join(map(str, [
            METHODS[row["method"]], row["total_cases"], row["comparable_cases"],
            row["ahead_count"], row["behind_count"], row["coincident_count"],
            row["no_economic_crossover_count"], row["missing_flexibility_knee_count"],
            f'{row["ahead_percent_of_comparable"]:.2f}',
        ])))
    lines.extend(["", "滞后明细（百分点）："])
    behind = details[details["order"] == "behind"]
    if behind.empty:
        lines.append("无。")
    for _, row in behind.iterrows():
        lines.append(
            f'{row["scenario_zh"]} / {row["consumption_target_%"]:g}% / {row["method_label"]}: '
            f'p_f={row["flexibility_knee_penetration_%"]:.4f}, '
            f'p_e={row["economic_breakeven_penetration_%"]:.4f}, '
            f'Delta p={row["economic_minus_flexibility_pp"]:.4f}'
        )
    no_cross = details.loc[details["economic_crossover_status"] == "no_crossover", KEYS].drop_duplicates()
    lines.extend(["", "分析域内无经济交叉点的组合："])
    if no_cross.empty:
        lines.append("无。")
    for _, row in no_cross.iterrows():
        lines.append(f'{SCENARIOS[row["scenario"]]} / {row["consumption_target_%"]:g}%')
    lines.extend([
        "", "输出文件：",
        "comparison_details.csv：逐场景、目标、方法的阈值、差值、先后顺序。",
        "summary_by_method.csv：各方法的总数、有效数、各类数量及有效比较中的占比。",
        "summary_by_scenario_method.csv：按场景及方法细分的相同统计。",
        "summary.txt：本说明与主要结果。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance-pp", type=float, default=DEFAULT_TOLERANCE_PP,
                        help="判定重合的绝对容差（百分点，默认 0.0001）")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果目录")
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_pp) or args.tolerance_pp < 0:
        parser.error("--tolerance-pp 必须是非负有限数")
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
    print(f"结果已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
