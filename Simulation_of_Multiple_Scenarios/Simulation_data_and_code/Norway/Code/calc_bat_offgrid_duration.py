#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_bat_offgrid_duration.py
===========================
基于 风+光+储(BESS) 容量配置，仿真计算微网离网运行时长。
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
# ║                     参 数 设 置                              ║
# ╚══════════════════════════════════════════════════════════════╝

PV_BASE_CAPACITY_MW = 0.08     # Norway PV 实测峰值~80kW
WIND_BASE_CAPACITY_MW = 0.225  # Norway 风电实测峰值~225kW

BASE_BESS_POWER_KW = 0
BASE_BESS_ENERGY_KWH = 0
BESS_CHARGE_EFF = 0.95
BESS_DISCHARGE_EFF = 0.95
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90

ALPHA = 1.0
INITIAL_BESS_SOC = 0.9
START_HOURS_OF_DAY = [0, 6, 12, 18]

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_8760 = os.path.join(_HERE, "Norway.xlsx")
INPUT_SENSITIVITY = os.path.join(_HERE, "sensitivity_wind_pv_bat_sizing_merged_results.xlsx")
FIGURES_FOLDER = os.path.join(_HERE, "offgrid_bat_duration_figures")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     数 据 读 取                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[数据] 读取 {filepath} 相关数据 ...")
    
    def _read_series(sheet_name):
        """高鲁棒性读取函数：兼容 Excel Sheet 和分离的 CSV 文件"""
        # 1. 尝试读 Excel Sheet
        if os.path.exists(filepath):
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                for col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() > 8000:
                        return s.values.astype(float)
            except Exception:
                pass
        
        # 2. 尝试读 CSV 备用文件 (对应你上传的文件命名格式)
        csv_path = f"{filepath} - {sheet_name}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None)
            for col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().sum() > 8000:
                    return s.values.astype(float)
                    
        raise FileNotFoundError(f"无法在 {filepath} 的 Sheet '{sheet_name}' 或 '{csv_path}' 中找到有效数据。")

    # Norway: 单 sheet 'train' 三列(col1=pv_production, col2=wind_production, col3=consumption)
    # 跳表头, 截前8760h; wind 含负值(待机/坏点)钳位>=0 (与 sizing 一致)
    _df = pd.read_excel(filepath, sheet_name="train", header=None, skiprows=1)
    pv_raw   = pd.to_numeric(_df.iloc[:, 1], errors="coerce").values.astype(float)
    wind_raw = pd.to_numeric(_df.iloc[:, 2], errors="coerce").values.astype(float)
    wind_raw = np.maximum(wind_raw, 0.0)
    load     = pd.to_numeric(_df.iloc[:, 3], errors="coerce").values.astype(float)
    _T = 8760
    pv_raw, wind_raw, load = pv_raw[:_T], wind_raw[:_T], load[:_T]

    assert len(pv_raw) == 8760, "光伏数据长度必须为8760"
    assert len(wind_raw) == 8760, "风电数据长度必须为8760"
    assert len(load) == 8760, "负荷数据长度必须为8760"

    # 空值填补
    for arr, name in [(pv_raw, "PV"), (wind_raw, "Wind"), (load, "Load")]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            s = pd.Series(arr)
            s.interpolate(inplace=True)
            s.bfill(inplace=True)
            arr[:] = s.values
        print(f"  {name}: 均值={np.mean(arr):.1f} kW, 峰值={np.max(arr):.1f} kW")
        
    return pv_raw, wind_raw, load


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
# ║         向 量 化 仿 真 核 心 (纯 BESS 版本)                  ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(re_avail, load, start_hours, alpha=ALPHA,
                        initial_bess_soc=INITIAL_BESS_SOC,
                        n_hours=8760,
                        bess_power=BASE_BESS_POWER_KW,
                        bess_energy=BASE_BESS_ENERGY_KWH):
    """
    对所有断网起点同时仿真 RE(风+光) + BESS 离网运行。
    调度规则:
      富余: RE→load, 剩余→BESS充电, 再剩余→弃电
      缺口: RE→load, 缺口→BESS放电, 再缺口→失败
    """
    n_events = len(start_hours)
    bess_soc = np.full(n_events, initial_bess_soc, dtype=np.float64)
    alive = np.ones(n_events, dtype=bool)
    autonomy = np.full(n_events, float(n_hours), dtype=np.float64)
    fail_reason = np.array([""] * n_events, dtype=object)

    for rel_h in range(n_hours):
        if not alive.any():
            break

        t_indices = (start_hours + rel_h) % n_hours
        re_now = re_avail[t_indices]  # 使用总新能源出力
        load_now = alpha * load[t_indices]

        # RE → 负荷
        re_to_load = np.minimum(re_now, load_now)
        deficit = load_now - re_to_load

        # ═══ 缺口处理 ═══
        need_discharge = alive & (deficit > 0)
        if need_discharge.any():
            bess_avail_energy = np.maximum(0.0, (bess_soc - BESS_SOC_MIN) * bess_energy)
            bess_avail_output = bess_avail_energy * BESS_DISCHARGE_EFF
            bess_max_discharge = np.minimum(bess_power, bess_avail_output)
            bess_discharge = np.where(need_discharge, np.minimum(deficit, bess_max_discharge), 0.0)
            
            bess_soc -= np.where(need_discharge, bess_discharge / BESS_DISCHARGE_EFF / bess_energy, 0.0)
            bess_soc = np.maximum(bess_soc, BESS_SOC_MIN)
            
            deficit_after_bess = deficit - bess_discharge

            # BESS放电后仍不足 -> 失败
            failed = need_discharge & (deficit_after_bess > 1e-6)
            if failed.any():
                idx = np.where(failed)[0]
                for i in idx:
                    if alive[i]:
                        alive[i] = False
                        autonomy[i] = rel_h
                        fail_reason[i] = "bess_energy_depleted"

        # ═══ 富余处理 ═══
        surplus = re_now - re_to_load
        can_charge = alive & (surplus > 0)
        if can_charge.any():
            bess_remain_cap = np.maximum(0.0, (BESS_SOC_MAX - bess_soc) * bess_energy)
            bess_max_charge_input = bess_remain_cap / BESS_CHARGE_EFF
            bess_max_charge = np.minimum(bess_power, bess_max_charge_input)
            bess_charge = np.where(can_charge, np.minimum(surplus, bess_max_charge), 0.0)
            
            bess_soc += np.where(can_charge, bess_charge * BESS_CHARGE_EFF / bess_energy, 0.0)
            bess_soc = np.minimum(bess_soc, BESS_SOC_MAX)

    return {
        "autonomy_hours": autonomy,
        "failure_reasons": fail_reason,
    }

# 绘图函数保持原样即可 (略作标签修改以适应风光)
def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)
    if 'cons_target_%' not in df.columns:
        return # 容错
    cons_vals = sorted(df['cons_target_%'].unique())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(cons_vals) - 1, 1)) for i in range(len(cons_vals))]

    df_plot = df[df['min_offgrid_hours'] < 8760].copy()

    # 图2: 综合曲线
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(cons_vals):
        sub = df_plot[df_plot['cons_target_%'] == ct].sort_values('energy_pen_%')
        ax.plot(sub['energy_pen_%'], sub['min_offgrid_hours'],
                'o-', color=colors[i], markersize=3, linewidth=1.2, alpha=0.85,
                label=f'cons_target = {ct}%')
    ax.set_xlabel('Energy Penetration (%)', fontsize=12)
    ax.set_ylabel('Min Off-grid Duration (hours)', fontsize=12)
    ax.set_title('Bat System: Min Off-grid Duration vs Energy Penetration', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_energy_pen_by_cons.png"), dpi=200)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════╗
# ║                      主 程 序                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  微网离网运行时长分析（纯电池版本 - 风光联合）")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # ── 1. 读取数据 ──
    pv_raw, wind_raw, load = load_8760_data()
    try:
        df_sens = load_sensitivity()
    except FileNotFoundError:
        print(f"找不到 {INPUT_SENSITIVITY}，请确保敏感性配置文件存在。")
        return
        
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
        
        # 提取光伏和风机容量（merged 结果含 PV_MW/Wind_MW/total_MW/total_MWh 全部列，严格索引）
        pv_mw = row["PV_MW"]
        wind_mw = row["Wind_MW"]
        # 扫描输出 total_MW/total_MWh(电池总容量, 已是总量), 直接用 (MW→kW)
        total_bess_power  = row["total_MW"]  * 1000
        total_bess_energy = row["total_MWh"] * 1000

        # 叠加风光出力
        pv_avail = pv_raw * (pv_mw / PV_BASE_CAPACITY_MW)
        wind_avail = wind_raw * (wind_mw / WIND_BASE_CAPACITY_MW)
        re_avail = pv_avail + wind_avail

        sim = simulate_vectorized(
            re_avail, load, start_hours, alpha=ALPHA,
            initial_bess_soc=INITIAL_BESS_SOC,
            n_hours=n_hours,
            bess_power=total_bess_power,
            bess_energy=total_bess_energy,
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
    plot_results(df_sens, FIGURES_FOLDER)

if __name__ == "__main__":
    main()