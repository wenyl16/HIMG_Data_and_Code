#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_h2_offgrid_duration.py
===========================

基于 sensitivity_h2_sizing_results.xlsx 中的每组 PV + BESS + H2 容量配置，
利用 Annual_8760.xlsx 的全年8760h 数据，仿真计算微网离网运行时长。

系统组成：PV + 原有BESS(2.5MW/5MWh) + 氢储能(电解槽+燃料电池+储氢罐)
调度规则：
  富余: PV→负荷 → BESS充电 → 电解槽制氢 → 弃光
  缺口: PV→负荷 → BESS放电 → 燃料电池发电 → 失败
"""

import os
import sys
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ╔══════════════════════════════════════════════════════════════╗
# ║                       参 数 设 置                           ║
# ╚══════════════════════════════════════════════════════════════╝

PV_BASE_CAPACITY_MW = 1
BASE_BESS_POWER_KW = 2500
BASE_BESS_ENERGY_KWH = 5000
BESS_CHARGE_EFF = 0.95
BESS_DISCHARGE_EFF = 0.95
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90

H2_PRODUCTION_KG_PER_KWH = 0.0205
FUEL_CELL_KWH_PER_KG = 18.3315
H2_SOC_MIN = 0.05
H2_SOC_MAX = 0.98

ALPHA = 1.0
INITIAL_BESS_SOC = 0.9
INITIAL_H2_SOC = 0.9
START_HOURS_OF_DAY = [0, 6, 12, 18]

INPUT_8760 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annual_8760.xlsx")
INPUT_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_h2_sizing_joint_results.xlsx")
FIGURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offgrid_h2_duration_figures")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     数 据 读 取                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[数据] 读取 {filepath} ...")
    df_pv = pd.read_excel(filepath, sheet_name="PV", header=None)
    pv_raw = pd.to_numeric(df_pv.iloc[:, 1], errors="coerce").values.astype(float)
    df_load = pd.read_excel(filepath, sheet_name="Load", header=None)
    load = pd.to_numeric(df_load.iloc[:, 1], errors="coerce").values.astype(float)

    assert len(pv_raw) == 8760
    assert len(load) == 8760

    for arr, name in [(pv_raw, "PV"), (load, "Load")]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            s = pd.Series(arr)
            s.interpolate(inplace=True)
            s.bfill(inplace=True)
            arr[:] = s.values

    print(f"  PV: 均值={np.mean(pv_raw):.1f} kW, 峰值={np.max(pv_raw):.1f} kW")
    print(f"  Load: 均值={np.mean(load):.1f} kW, 峰值={np.max(load):.1f} kW")
    return pv_raw, load


def load_sensitivity(filepath=INPUT_SENSITIVITY):
    print(f"[数据] 读取 {filepath} ...")
    df = pd.read_excel(filepath, sheet_name="Results")
    print(f"  {len(df)} 行配置")
    return df


def generate_start_info(n_hours=8760):
    n_days = n_hours // 24
    start_hours, start_days, start_hod = [], [], []
    for d in range(n_days):
        for h in START_HOURS_OF_DAY:
            sh = d * 24 + h
            if sh < n_hours:
                start_hours.append(sh)
                start_days.append(d + 1)
                start_hod.append(h)
    print(f"[起点] 共 {len(start_hours)} 个断网起点")
    return np.array(start_hours), np.array(start_days), np.array(start_hod)


# ╔══════════════════════════════════════════════════════════════╗
# ║          向 量 化 仿 真 核 心 (BESS + H2)                   ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(pv_avail, load, elec_kw, fc_kw, h2_tank_kg,
                        start_hours, alpha=ALPHA,
                        initial_bess_soc=INITIAL_BESS_SOC,
                        initial_h2_soc=INITIAL_H2_SOC,
                        n_hours=8760):
    """
    对所有断网起点同时仿真 PV + BESS + H2 离网运行。

    调度规则:
      富余: PV→load, 剩余→BESS充电, 再剩余→电解槽制氢, 再剩余→弃光
      缺口: PV→load, 缺口→BESS放电, 再缺口→燃料电池(H2), 再缺口→失败
    """
    n_events = len(start_hours)

    # 状态
    bess_soc = np.full(n_events, initial_bess_soc, dtype=np.float64)
    h2_kg = np.full(n_events, initial_h2_soc * h2_tank_kg, dtype=np.float64)
    alive = np.ones(n_events, dtype=bool)
    autonomy = np.full(n_events, float(n_hours), dtype=np.float64)
    fail_reason = np.array([""] * n_events, dtype=object)

    for rel_h in range(n_hours):
        if not alive.any():
            break

        t_indices = (start_hours + rel_h) % n_hours
        pv_now = pv_avail[t_indices]
        load_now = alpha * load[t_indices]

        # PV → 负荷
        pv_to_load = np.minimum(pv_now, load_now)
        deficit = load_now - pv_to_load

        # ═══ 缺口处理 ═══
        need_discharge = alive & (deficit > 0)
        if need_discharge.any():

            # Step 1: BESS放电
            bess_avail_energy = np.maximum(0.0, (bess_soc - BESS_SOC_MIN) * BASE_BESS_ENERGY_KWH)
            bess_avail_output = bess_avail_energy * BESS_DISCHARGE_EFF
            bess_max_discharge = np.minimum(BASE_BESS_POWER_KW, bess_avail_output)
            bess_discharge = np.where(need_discharge, np.minimum(deficit, bess_max_discharge), 0.0)
            bess_soc -= np.where(need_discharge, bess_discharge / BESS_DISCHARGE_EFF / BASE_BESS_ENERGY_KWH, 0.0)
            bess_soc = np.maximum(bess_soc, BESS_SOC_MIN)
            deficit_after_bess = deficit - bess_discharge

            # Step 2: 燃料电池
            remaining_deficit = np.where(need_discharge, np.maximum(0.0, deficit_after_bess), 0.0)
            need_fc = need_discharge & (remaining_deficit > 1e-6)

            if need_fc.any():
                avail_h2_kg = np.maximum(0.0, h2_kg - H2_SOC_MIN * h2_tank_kg)
                avail_fc_energy = avail_h2_kg * FUEL_CELL_KWH_PER_KG
                fc_max_output = np.minimum(fc_kw, avail_fc_energy)
                fc_output = np.where(need_fc, np.minimum(remaining_deficit, fc_max_output), 0.0)
                h2_kg -= np.where(need_fc & (fc_output > 0), fc_output / FUEL_CELL_KWH_PER_KG, 0.0)
                h2_kg = np.maximum(h2_kg, 0.0)
                final_deficit = remaining_deficit - fc_output

                # 检查失败
                failed = need_fc & (final_deficit > 1e-6)
                if failed.any():
                    idx = np.where(failed)[0]
                    for i in idx:
                        if alive[i]:
                            alive[i] = False
                            autonomy[i] = rel_h
                            ae = avail_fc_energy[i]
                            rd = remaining_deficit[i]   # FC 出力前的缺口(修复: 原误用 final_deficit 导致标签失真)
                            if ae < rd - 1e-6:
                                fail_reason[i] = "h2_depleted"              # H2 能量不足以覆盖该小时缺口
                            elif fc_kw < rd - 1e-6:
                                fail_reason[i] = "fuel_cell_power_limited"  # FC 功率受限(H2 够但功率不够)
                            else:
                                fail_reason[i] = "pv_bess_h2_insufficient"  # 兜底
            else:
                # BESS放电后仍不足，但无FC或不需要FC
                failed = need_discharge & (deficit_after_bess > 1e-6) & (~need_fc | (remaining_deficit > 1e-6))
                # Recheck: cases where deficit_after_bess > 0 and fc didn't cover
                failed_no_fc = need_discharge & (deficit_after_bess > 1e-6)
                # This path: BESS discharged but still deficit, FC not engaged (h2 tank may be empty or fc_kw=0)
                if failed_no_fc.any():
                    idx2 = np.where(failed_no_fc)[0]
                    for i in idx2:
                        if alive[i]:
                            alive[i] = False
                            autonomy[i] = rel_h
                            if fc_kw <= 1e-6 and h2_tank_kg <= 1e-6:
                                fail_reason[i] = "bess_energy_depleted"
                            else:
                                avail_h2 = max(0.0, h2_kg[i] - H2_SOC_MIN * h2_tank_kg)
                                if avail_h2 <= 1e-6:
                                    fail_reason[i] = "h2_depleted"
                                else:
                                    fail_reason[i] = "fuel_cell_power_limited"

        # ═══ 富余处理 ═══
        surplus = pv_now - pv_to_load
        can_charge = alive & (surplus > 0)
        if can_charge.any():

            # Step 1: BESS充电
            bess_remain_cap = np.maximum(0.0, (BESS_SOC_MAX - bess_soc) * BASE_BESS_ENERGY_KWH)
            bess_max_charge_input = bess_remain_cap / BESS_CHARGE_EFF
            bess_max_charge = np.minimum(BASE_BESS_POWER_KW, bess_max_charge_input)
            bess_charge = np.where(can_charge, np.minimum(surplus, bess_max_charge), 0.0)
            bess_soc += np.where(can_charge, bess_charge * BESS_CHARGE_EFF / BASE_BESS_ENERGY_KWH, 0.0)
            bess_soc = np.minimum(bess_soc, BESS_SOC_MAX)
            surplus_after_bess = surplus - bess_charge

            # Step 2: 电解槽制氢
            can_electrolyze = can_charge & (surplus_after_bess > 1e-6) & (elec_kw > 0) & (h2_tank_kg > 0)
            if can_electrolyze.any():
                h2_headroom = np.maximum(0.0, H2_SOC_MAX * h2_tank_kg - h2_kg)
                max_elec_input = h2_headroom / H2_PRODUCTION_KG_PER_KWH
                elec_max = np.minimum(elec_kw, max_elec_input)
                elec_power = np.where(can_electrolyze, np.minimum(surplus_after_bess, elec_max), 0.0)
                h2_kg += np.where(can_electrolyze & (elec_power > 0), elec_power * H2_PRODUCTION_KG_PER_KWH, 0.0)
                h2_kg = np.minimum(h2_kg, H2_SOC_MAX * h2_tank_kg)

    return {
        "autonomy_hours": autonomy,
        "failure_reasons": fail_reason,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                    图 形 绘 制                               ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)
    cons_vals = sorted(df['cons_target_%'].unique())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(cons_vals) - 1, 1)) for i in range(len(cons_vals))]

    # 去除撑满全年8760h的结果
    df_plot = df[df['min_offgrid_hours'] < 8760].copy()
    print(f"  [绘图] 去除撑满全年配置后: {len(df_plot)} / {len(df)} 行")

    # ── 图1: 6子图 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for i, ct in enumerate(cons_vals):
        ax = axes_flat[i]
        sub = df_plot[df_plot['cons_target_%'] == ct].sort_values('energy_pen_%')
        ax.plot(sub['energy_pen_%'], sub['min_offgrid_hours'],
                'o-', color='steelblue', markersize=3, linewidth=1.2, alpha=0.85)
        ax.set_title(f'cons_target = {ct}%', fontsize=12, fontweight='bold')
        ax.set_xlabel('Energy Penetration (%)', fontsize=10)
        ax.set_ylabel('Min Off-grid Hours', fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
    fig.suptitle('H2 System: Min Off-grid Duration vs Energy Penetration (survived < 8760h)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_energy_pen_6panels.png"), dpi=200)
    plt.close(fig)
    print(f"  [图] offgrid_vs_energy_pen_6panels.png")

    # ── 图2: 综合曲线 ──
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(cons_vals):
        sub = df_plot[df_plot['cons_target_%'] == ct].sort_values('energy_pen_%')
        ax.plot(sub['energy_pen_%'], sub['min_offgrid_hours'],
                'o-', color=colors[i], markersize=3, linewidth=1.2, alpha=0.85,
                label=f'cons_target = {ct}%')
    ax.set_xlabel('Energy Penetration (%)', fontsize=12)
    ax.set_ylabel('Min Off-grid Duration (hours)', fontsize=12)
    ax.set_title('H2 System: Min Off-grid Duration vs Energy Penetration (survived < 8760h)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_energy_pen_by_cons.png"), dpi=200)
    plt.close(fig)
    print(f"  [图] offgrid_vs_energy_pen_by_cons.png")

    # ── 图3: 离网时长 vs PV散点图 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ct in enumerate(cons_vals):
        sub = df_plot[df_plot['cons_target_%'] == ct]
        ax.scatter(sub['PV_MW'], sub['min_offgrid_hours'],
                   c=[colors[i]], label=f'{ct}%', s=30, alpha=0.7, edgecolors='none')
    ax.set_xlabel('PV Capacity (MW)', fontsize=12)
    ax.set_ylabel('Min Off-grid Duration (hours)', fontsize=12)
    ax.set_title('H2 System: Off-grid Duration vs PV Capacity', fontsize=13, fontweight='bold')
    ax.legend(title='cons_target', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_pv_by_cons.png"), dpi=200)
    plt.close(fig)
    print(f"  [图] offgrid_vs_pv_by_cons.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      主 程 序                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  微网离网运行时长分析（氢储能版本）")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # ── 1. 读取数据 ──
    pv_raw, load = load_8760_data()
    df_sens = load_sensitivity()
    n_hours = 8760

    # ── 2. 断网起点 ──
    start_hours, start_days, start_hod = generate_start_info(n_hours)

    # ── 3. 逐行仿真 ──
    results = {
        "min_offgrid_hours": [],
        "mean_offgrid_hours": [],
        "p10_offgrid_hours": [],
        "max_offgrid_hours": [],
        "survival_rate": [],
        "worst_start_day": [],
        "worst_start_hour": [],
        "worst_failure_reason": [],
    }

    for idx in tqdm(range(len(df_sens)), desc="仿真进度"):
        row = df_sens.iloc[idx]
        pv_mw = row["PV_MW"]
        elec_kw = row["elec_cap_kW"]
        fc_kw_val = row["fc_cap_kW"] + 1.0  # +1kW余量: 确保FC严格>峰值负荷, 避免浮点精度导致恰好差一点
        h2_tank = row["h2_tank_kg"]

        pv_avail = pv_raw * (pv_mw / PV_BASE_CAPACITY_MW)

        sim = simulate_vectorized(
            pv_avail, load, elec_kw, fc_kw_val, h2_tank,
            start_hours, alpha=ALPHA,
            initial_bess_soc=INITIAL_BESS_SOC,
            initial_h2_soc=INITIAL_H2_SOC,
            n_hours=n_hours
        )

        autonomies = sim["autonomy_hours"]
        reasons = sim["failure_reasons"]
        worst_idx = np.argmin(autonomies)

        results["min_offgrid_hours"].append(float(autonomies[worst_idx]))
        results["mean_offgrid_hours"].append(float(np.mean(autonomies)))
        results["p10_offgrid_hours"].append(float(np.percentile(autonomies, 10)))
        results["max_offgrid_hours"].append(float(np.max(autonomies)))
        results["survival_rate"].append(float(np.mean(autonomies >= n_hours)))
        results["worst_start_day"].append(int(start_days[worst_idx]))
        results["worst_start_hour"].append(int(start_hod[worst_idx]))
        results["worst_failure_reason"].append(str(reasons[worst_idx]))

    # ── 4. 写回Excel ──
    for col, vals in results.items():
        df_sens[col] = vals

    backup_path = INPUT_SENSITIVITY.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        shutil.copy2(INPUT_SENSITIVITY, backup_path)
        print(f"\n[备份] 原文件已备份至: {backup_path}")

    with pd.ExcelWriter(INPUT_SENSITIVITY, engine="openpyxl") as writer:
        df_sens.to_excel(writer, sheet_name="Results", index=False)
    print(f"\n[输出] 结果已写回: {INPUT_SENSITIVITY}")

    # ── 5. 绘图 ──
    print("\n[图表] 生成可视化 ...")
    plot_results(df_sens, FIGURES_FOLDER)

    # ── 6. 摘要 ──
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  运 行 摘 要")
    print(f"{'='*70}")
    print(f"  运行耗时: {elapsed:.0f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  配置总数: {len(df_sens)}")
    print(f"  min_offgrid_hours 统计:")
    print(f"    最小值: {df_sens['min_offgrid_hours'].min():.1f} h")
    print(f"    最大值: {df_sens['min_offgrid_hours'].max():.1f} h ({df_sens['min_offgrid_hours'].max()/24:.1f} 天)")
    print(f"    中位数: {df_sens['min_offgrid_hours'].median():.1f} h")
    print(f"    均值:   {df_sens['min_offgrid_hours'].mean():.1f} h")
    print(f"  能撑到年末: {(df_sens['survival_rate'] == 1.0).sum()} / {len(df_sens)}")
    print(f"  失败原因分布:")
    print(df_sens['worst_failure_reason'].value_counts().to_string())
    print(f"\n  Excel: {INPUT_SENSITIVITY}")
    print(f"  图片: {FIGURES_FOLDER}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
