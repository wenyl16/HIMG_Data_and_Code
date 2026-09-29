#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_h2_offgrid_duration.py
===========================
Simulate microgrid off-grid duration for wind + PV + BESS + hydrogen capacity configurations.
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
#                       Parameter settings
# ╚══════════════════════════════════════════════════════════════╝

PV_BASE_CAPACITY_MW = 0.08     # Norway measured PV peak: approximately 80 kW
WIND_BASE_CAPACITY_MW = 0.225  # Norway measured wind peak: approximately 225 kW

BASE_BESS_POWER_KW = 250    # Norway existing battery: 250 kW, matching sizing P_BAT_MAX.
BASE_BESS_ENERGY_KWH = 500   # Norway existing battery: 500 kWh, matching sizing E_BAT_MAX.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_8760 = os.path.join(_HERE, "Norway.xlsx")
INPUT_SENSITIVITY = os.path.join(_HERE, "sensitivity_wind_pv_h2_sizing_joint_results.xlsx")
FIGURES_FOLDER = os.path.join(_HERE, "offgrid_h2_duration_figures")

# ╔══════════════════════════════════════════════════════════════╗
#                       Data loading
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[Data] Reading {filepath} time-series data ...")
    
    def _read_series(sheet_name):
        # 1. Try the Excel sheet.
        if os.path.exists(filepath):
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
                for col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() > 8000:
                        return s.values.astype(float)
            except Exception:
                pass
        
        # 2. Try the fallback CSV file.
        csv_path = f"{filepath} - {sheet_name}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None)
            for col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().sum() > 8000:
                    return s.values.astype(float)
                    
        raise FileNotFoundError(f"No valid data found in {filepath}, sheet '{sheet_name}', or '{csv_path}'.")

    # Norway: three columns in sheet 'train' (col1=pv_production, col2=wind_production, col3=consumption).
    # Skip the header and retain the first 8760 hours; clip negative wind values (standby/bad data) to zero, as in sizing.
    _df = pd.read_excel(filepath, sheet_name="train", header=None, skiprows=1)
    pv_raw   = pd.to_numeric(_df.iloc[:, 1], errors="coerce").values.astype(float)
    wind_raw = pd.to_numeric(_df.iloc[:, 2], errors="coerce").values.astype(float)
    wind_raw = np.maximum(wind_raw, 0.0)
    load     = pd.to_numeric(_df.iloc[:, 3], errors="coerce").values.astype(float)
    _T = 8760
    pv_raw, wind_raw, load = pv_raw[:_T], wind_raw[:_T], load[:_T]

    # Fill missing values.
    for arr, name in [(pv_raw, "PV"), (wind_raw, "Wind"), (load, "Load")]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            s = pd.Series(arr)
            s.interpolate(inplace=True)
            s.bfill(inplace=True)
            arr[:] = s.values
        print(f"  {name}: mean={np.mean(arr):.1f} kW, peak={np.max(arr):.1f} kW")
        
    return pv_raw, wind_raw, load

def load_sensitivity(filepath=INPUT_SENSITIVITY):
    print(f"[Data] Reading {filepath} ...")
    df = pd.read_excel(filepath, sheet_name="Results")
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
    return np.array(start_hours), np.array(start_days), np.array(start_hod)

# ╔══════════════════════════════════════════════════════════════╗
#                       Vectorized simulation (BESS + H2)
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(re_avail, load, elec_kw, fc_kw, h2_tank_kg,
                        start_hours, alpha=ALPHA,
                        initial_bess_soc=INITIAL_BESS_SOC,
                        initial_h2_soc=INITIAL_H2_SOC,
                        n_hours=8760):
    """
    Simulate RE (wind + PV) + BESS + H2 off-grid operation for all outage starting times.
    Dispatch rules:
      Surplus: RE -> load, then BESS charging, hydrogen production, and curtailment.
      Deficit: RE -> load, then BESS discharge, fuel cell generation, and supply failure.
    """
    n_events = len(start_hours)
    bess_soc = np.full(n_events, initial_bess_soc, dtype=np.float64)
    h2_kg = np.full(n_events, initial_h2_soc * h2_tank_kg, dtype=np.float64)
    alive = np.ones(n_events, dtype=bool)
    autonomy = np.full(n_events, float(n_hours), dtype=np.float64)
    fail_reason = np.array([""] * n_events, dtype=object)

    for rel_h in range(n_hours):
        if not alive.any():
            break

        t_indices = (start_hours + rel_h) % n_hours
        re_now = re_avail[t_indices]
        load_now = alpha * load[t_indices]

        # RE -> load
        re_to_load = np.minimum(re_now, load_now)
        deficit = load_now - re_to_load

        # Handle generation deficits.
        need_discharge = alive & (deficit > 0)
        if need_discharge.any():
            # Step 1: Discharge the BESS.
            bess_avail_energy = np.maximum(0.0, (bess_soc - BESS_SOC_MIN) * BASE_BESS_ENERGY_KWH)
            bess_avail_output = bess_avail_energy * BESS_DISCHARGE_EFF
            bess_max_discharge = np.minimum(BASE_BESS_POWER_KW, bess_avail_output)
            bess_discharge = np.where(need_discharge, np.minimum(deficit, bess_max_discharge), 0.0)
            
            bess_soc -= np.where(need_discharge, bess_discharge / BESS_DISCHARGE_EFF / BASE_BESS_ENERGY_KWH, 0.0)
            bess_soc = np.maximum(bess_soc, BESS_SOC_MIN)
            deficit_after_bess = deficit - bess_discharge

            # Step 2: Generate power with the fuel cell.
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

                # Check for supply failure.
                failed = need_fc & (final_deficit > 1e-6)
                if failed.any():
                    idx = np.where(failed)[0]
                    for i in idx:
                        if alive[i]:
                            alive[i] = False
                            autonomy[i] = rel_h
                            ae = avail_fc_energy[i]
                            rd = remaining_deficit[i]   # Deficit before FC output; using final_deficit here previously misclassified failures.
                            if ae < rd - 1e-6:
                                fail_reason[i] = "h2_depleted"
                            elif fc_kw < rd - 1e-6:
                                fail_reason[i] = "fuel_cell_power_limited"
                            else:
                                fail_reason[i] = "re_bess_h2_insufficient"
            else:
                # A deficit remains after BESS discharge, but no FC is available.
                failed_no_fc = need_discharge & (deficit_after_bess > 1e-6)
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

        # Handle surplus generation.
        surplus = re_now - re_to_load
        can_charge = alive & (surplus > 0)
        if can_charge.any():
            # Step 1: Charge the BESS.
            bess_remain_cap = np.maximum(0.0, (BESS_SOC_MAX - bess_soc) * BASE_BESS_ENERGY_KWH)
            bess_max_charge_input = bess_remain_cap / BESS_CHARGE_EFF
            bess_max_charge = np.minimum(BASE_BESS_POWER_KW, bess_max_charge_input)
            bess_charge = np.where(can_charge, np.minimum(surplus, bess_max_charge), 0.0)
            
            bess_soc += np.where(can_charge, bess_charge * BESS_CHARGE_EFF / BASE_BESS_ENERGY_KWH, 0.0)
            bess_soc = np.minimum(bess_soc, BESS_SOC_MAX)
            surplus_after_bess = surplus - bess_charge

            # Step 2: Produce hydrogen with the electrolyzer.
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


def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)
    if 'cons_target_%' not in df.columns:
        return
    cons_vals = sorted(df['cons_target_%'].unique())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(cons_vals) - 1, 1)) for i in range(len(cons_vals))]

    df_plot = df[df['min_offgrid_hours'] < 8760].copy()

    # Figure 2: Combined curves
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(cons_vals):
        sub = df_plot[df_plot['cons_target_%'] == ct].sort_values('energy_pen_%')
        ax.plot(sub['energy_pen_%'], sub['min_offgrid_hours'],
                'o-', color=colors[i], markersize=3, linewidth=1.2, alpha=0.85,
                label=f'cons_target = {ct}%')
    ax.set_xlabel('Energy Penetration (%)', fontsize=12)
    ax.set_ylabel('Min Off-grid Duration (hours)', fontsize=12)
    ax.set_title('H2 System: Min Off-grid Duration vs Energy Penetration', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_energy_pen_by_cons.png"), dpi=200)
    plt.close(fig)

# ╔══════════════════════════════════════════════════════════════╗
#                       Main program
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  Microgrid off-grid duration analysis (wind + PV + hydrogen storage)")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # 1. Load data.
    pv_raw, wind_raw, load = load_8760_data()
    try:
        df_sens = load_sensitivity()
    except FileNotFoundError:
        print(f"Cannot find {INPUT_SENSITIVITY}; check that the file exists.")
        return
    n_hours = 8760

    # 2. Generate outage starting times.
    start_hours, start_days, start_hod = generate_start_info(n_hours)

    # 3. Simulate each configuration.
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

    for idx in tqdm(range(len(df_sens)), desc="Simulation progress"):
        row = df_sens.iloc[idx]
        
        # Read equipment capacities.
        pv_mw = row.get("PV_MW", 0.0)
        wind_mw = row.get("Wind_MW", 0.0)
        elec_kw = row.get("elec_cap_kW", 0.0)
        fc_kw_val = row.get("fc_cap_kW", 0.0) + 1.0  # Add 1 kW, as in USA/Zambia, so FC capacity strictly exceeds peak load.
        h2_tank = row.get("h2_tank_kg", 0.0)

        # Combine wind and PV generation.
        pv_avail = pv_raw * (pv_mw / PV_BASE_CAPACITY_MW)
        wind_avail = wind_raw * (wind_mw / WIND_BASE_CAPACITY_MW)
        re_avail = pv_avail + wind_avail

        sim = simulate_vectorized(
            re_avail, load, elec_kw, fc_kw_val, h2_tank,
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

    # 4. Write results back to Excel.
    for col, vals in results.items():
        df_sens[col] = vals

    backup_path = INPUT_SENSITIVITY.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        shutil.copy2(INPUT_SENSITIVITY, backup_path)
        print(f"\n[Backup] Original file saved to: {backup_path}")

    with pd.ExcelWriter(INPUT_SENSITIVITY, engine="openpyxl") as writer:
        df_sens.to_excel(writer, sheet_name="Results", index=False)
    print(f"\n[Output] Results written to: {INPUT_SENSITIVITY}")

    # 5. Plot results.
    plot_results(df_sens, FIGURES_FOLDER)

if __name__ == "__main__":
    main()