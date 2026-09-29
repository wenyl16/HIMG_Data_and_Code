#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_h2_offgrid_duration.py
===========================

For each PV + BESS + H2 capacity configuration in sensitivity_h2_sizing_results.xlsx,
simulate microgrid off-grid operation using the full 8760-hour series in Annual_8760.xlsx.

System: PV + existing BESS (2.5 MW/5 MWh) + hydrogen storage (electrolyzer, fuel cell, tank).
Dispatch rules:
  Surplus: PV supplies load, then charges BESS, then produces hydrogen; curtail the remainder.
  Deficit: PV supplies load, then BESS discharges, then the fuel cell generates; fail if load remains unmet.
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
# |                       PARAMETERS                           |
# ╚══════════════════════════════════════════════════════════════╝

PV_BASE_CAPACITY_MW = 1
BASE_BESS_POWER_KW = 500
BASE_BESS_ENERGY_KWH = 2000
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
# |                       DATA LOADING                           |
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[Data] Reading {filepath} ...")
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

    print(f"  PV: mean={np.mean(pv_raw):.1f} kW, peak={np.max(pv_raw):.1f} kW")
    print(f"  Load: mean={np.mean(load):.1f} kW, peak={np.max(load):.1f} kW")
    return pv_raw, load


def load_sensitivity(filepath=INPUT_SENSITIVITY):
    print(f"[Data] Reading {filepath} ...")
    df = pd.read_excel(filepath, sheet_name="Results")
    print(f"  {len(df)} configuration rows")
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
    print(f"[Start times] {len(start_hours)} outage start times")
    return np.array(start_hours), np.array(start_days), np.array(start_hod)


# ╔══════════════════════════════════════════════════════════════╗
# |             VECTORIZED SIMULATION (BESS + H2)               |
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(pv_avail, load, elec_kw, fc_kw, h2_tank_kg,
                        start_hours, alpha=ALPHA,
                        initial_bess_soc=INITIAL_BESS_SOC,
                        initial_h2_soc=INITIAL_H2_SOC,
                        n_hours=8760):
    """
    Simulate PV + BESS + H2 off-grid operation for all outage start times simultaneously.

    Dispatch rules:
      Surplus: PV supplies load, then charges BESS, then powers the electrolyzer; curtail any remainder.
      Deficit: PV supplies load, then BESS discharges, then the fuel cell uses H2; fail if load remains unmet.
    """
    n_events = len(start_hours)

    # State variables.
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

        # PV supplies the load.
        pv_to_load = np.minimum(pv_now, load_now)
        deficit = load_now - pv_to_load

        # === Handle deficits ===
        need_discharge = alive & (deficit > 0)
        if need_discharge.any():

            # Step 1: Discharge BESS.
            bess_avail_energy = np.maximum(0.0, (bess_soc - BESS_SOC_MIN) * BASE_BESS_ENERGY_KWH)
            bess_avail_output = bess_avail_energy * BESS_DISCHARGE_EFF
            bess_max_discharge = np.minimum(BASE_BESS_POWER_KW, bess_avail_output)
            bess_discharge = np.where(need_discharge, np.minimum(deficit, bess_max_discharge), 0.0)
            bess_soc -= np.where(need_discharge, bess_discharge / BESS_DISCHARGE_EFF / BASE_BESS_ENERGY_KWH, 0.0)
            bess_soc = np.maximum(bess_soc, BESS_SOC_MIN)
            deficit_after_bess = deficit - bess_discharge

            # Step 2: Fuel cell.
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

                # Check for failure.
                failed = need_fc & (final_deficit > 1e-6)
                if failed.any():
                    idx = np.where(failed)[0]
                    for i in idx:
                        if alive[i]:
                            alive[i] = False
                            autonomy[i] = rel_h
                            ae = avail_fc_energy[i]
                            rd = remaining_deficit[i]   # Deficit before FC output; using final_deficit previously misclassified failures.
                            if ae < rd - 1e-6:
                                fail_reason[i] = "h2_depleted"              # H2 energy cannot cover this hour's deficit.
                            elif fc_kw < rd - 1e-6:
                                fail_reason[i] = "fuel_cell_power_limited"  # Sufficient H2, but insufficient FC power.
                            else:
                                fail_reason[i] = "pv_bess_h2_insufficient"  # Fallback reason.
            else:
                # BESS discharge is insufficient, but no FC is available or required.
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

        # === Handle surplus ===
        surplus = pv_now - pv_to_load
        can_charge = alive & (surplus > 0)
        if can_charge.any():

            # Step 1: Charge BESS.
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


# ╔══════════════════════════════════════════════════════════════╗
# |                          PLOTTING                            |
# ╚══════════════════════════════════════════════════════════════╝

def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)
    cons_vals = sorted(df['cons_target_%'].unique())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(cons_vals) - 1, 1)) for i in range(len(cons_vals))]

    # Exclude configurations that survive all 8760 hours.
    df_plot = df[df['min_offgrid_hours'] < 8760].copy()
    print(f"  [Plotting] After excluding configurations that survive the full year: {len(df_plot)} / {len(df)} rows")

    # -- Figure 1: six panels --
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
    print(f"  [Figure] offgrid_vs_energy_pen_6panels.png")

    # -- Figure 2: combined curves --
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
    print(f"  [Figure] offgrid_vs_energy_pen_by_cons.png")

    # -- Figure 3: off-grid duration vs PV scatter plot --
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
    print(f"  [Figure] offgrid_vs_pv_by_cons.png")


# ╔══════════════════════════════════════════════════════════════╗
# |                         MAIN PROGRAM                         |
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  Microgrid off-grid duration analysis (hydrogen storage)")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # -- 1. Load data --
    pv_raw, load = load_8760_data()
    df_sens = load_sensitivity()
    n_hours = 8760

    # -- 2. Outage start times --
    start_hours, start_days, start_hod = generate_start_info(n_hours)

    # -- 3. Simulate each configuration row --
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
        pv_mw = row["PV_MW"]
        elec_kw = row["elec_cap_kW"]
        fc_kw_val = row["fc_cap_kW"] + 1.0  # Add a 1 kW margin to keep FC power above peak load despite floating-point error.
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

    # -- 4. Write results back to Excel --
    for col, vals in results.items():
        df_sens[col] = vals

    backup_path = INPUT_SENSITIVITY.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        shutil.copy2(INPUT_SENSITIVITY, backup_path)
        print(f"\n[Backup] Original file backed up to: {backup_path}")

    with pd.ExcelWriter(INPUT_SENSITIVITY, engine="openpyxl") as writer:
        df_sens.to_excel(writer, sheet_name="Results", index=False)
    print(f"\n[Output] Results written back to: {INPUT_SENSITIVITY}")

    # -- 5. Plot results --
    print("\n[Figures] Generating plots ...")
    plot_results(df_sens, FIGURES_FOLDER)

    # -- 6. Summary --
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  RUN SUMMARY")
    print(f"{'='*70}")
    print(f"  Elapsed time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Total configurations: {len(df_sens)}")
    print(f"  min_offgrid_hours statistics:")
    print(f"    Minimum: {df_sens['min_offgrid_hours'].min():.1f} h")
    print(f"    Maximum: {df_sens['min_offgrid_hours'].max():.1f} h ({df_sens['min_offgrid_hours'].max()/24:.1f} days)")
    print(f"    Median: {df_sens['min_offgrid_hours'].median():.1f} h")
    print(f"    Mean:   {df_sens['min_offgrid_hours'].mean():.1f} h")
    print(f"  Configurations surviving the full year: {(df_sens['survival_rate'] == 1.0).sum()} / {len(df_sens)}")
    print(f"  Failure reason distribution:")
    print(df_sens['worst_failure_reason'].value_counts().to_string())
    print(f"\n  Excel: {INPUT_SENSITIVITY}")
    print(f"  Figures: {FIGURES_FOLDER}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
