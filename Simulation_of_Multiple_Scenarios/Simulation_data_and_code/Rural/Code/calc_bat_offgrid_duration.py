#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_bat_offgrid_duration.py
========================

For each PV + BESS capacity configuration in sensitivity_bat_sizing_merged_results.xlsx,
simulate microgrid off-grid operation using the full 8760-hour series in Annual_8760.xlsx.

Method:
  - 365 days x 4 times (0/6/12/18) = 1460 outage start times.
  - Simulate each start time hourly from SOC=0.9 (PV first, then charging or discharging).
  - Use the minimum duration across the 1460 scenarios as the off-grid capability metric.
  - Write the results back to the original Excel file as additional columns.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ╔══════════════════════════════════════════════════════════════╗
# |                       PARAMETERS                             |
# ╚══════════════════════════════════════════════════════════════╝

PV_BASE_CAPACITY_MW = 1
BASE_BESS_POWER_KW = 2500
BASE_BESS_ENERGY_KWH = 5000
BESS_CHARGE_EFF = 0.95
BESS_DISCHARGE_EFF = 0.95
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90
ALPHA = 1.0
INITIAL_SOC = 0.9
START_HOURS_OF_DAY = [0, 6, 12, 18]

INPUT_8760 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annual_8760.xlsx")
INPUT_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_bat_sizing_merged_results.xlsx")
FIGURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offgrid_duration_figures")


# ╔══════════════════════════════════════════════════════════════╗
# |                       DATA LOADING                           |
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[Data] Reading {filepath} ...")
    df_pv = pd.read_excel(filepath, sheet_name="PV", header=None)
    pv_raw = pd.to_numeric(df_pv.iloc[:, 1], errors="coerce").values.astype(float)
    df_load = pd.read_excel(filepath, sheet_name="Load", header=None)
    load = pd.to_numeric(df_load.iloc[:, 1], errors="coerce").values.astype(float)

    assert len(pv_raw) == 8760, f"PV series length {len(pv_raw)} != 8760"
    assert len(load) == 8760, f"Load series length {len(load)} != 8760"

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
    """Generate indices and metadata for the 1460 outage start times."""
    n_days = n_hours // 24
    start_hours = []
    start_days = []
    start_hod = []
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
# |                  VECTORIZED SIMULATION                       |
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(pv_avail, load, total_energy_kwh, total_power_kw,
                        start_hours, alpha=ALPHA, initial_soc=INITIAL_SOC,
                        n_hours=8760):
    """
    Simulate all outage start times hourly in parallel using NumPy arrays.

    Returns a dict:
      autonomy_hours: (n_events,) off-grid operating duration for each scenario
      failure_reasons: (n_events,) reason for failure
    """
    n_events = len(start_hours)
    max_sim = n_hours  # Simulate for up to one full year.

    # State of each scenario.
    soc = np.full(n_events, initial_soc, dtype=np.float64)
    alive = np.ones(n_events, dtype=bool)  # Scenarios still operating.
    autonomy = np.full(n_events, max_sim, dtype=np.float64)  # Default: survive the full simulation year.
    fail_reason = np.array([""] * n_events, dtype=object)

    for rel_h in range(max_sim):
        if not alive.any():
            break

        # Absolute index of the current hour, wrapping around the annual series.
        t_indices = (start_hours + rel_h) % n_hours

        # Get the current PV output and load for active scenarios only.
        pv_now = pv_avail[t_indices]
        load_now = alpha * load[t_indices]

        # Supply the load from PV first.
        pv_to_load = np.minimum(pv_now, load_now)
        deficit = load_now - pv_to_load

        # --- Scenarios requiring battery discharge ---
        need_discharge = alive & (deficit > 0)
        if need_discharge.any():
            avail_energy = np.maximum(0.0, (soc - BESS_SOC_MIN) * total_energy_kwh)
            avail_output = avail_energy * BESS_DISCHARGE_EFF
            max_discharge = np.minimum(total_power_kw, avail_output)
            actual_discharge = np.minimum(deficit, max_discharge)
            actual_discharge = np.where(need_discharge, actual_discharge, 0.0)

            # Update SOC.
            soc -= np.where(need_discharge, actual_discharge / BESS_DISCHARGE_EFF / total_energy_kwh, 0.0)
            soc = np.maximum(soc, BESS_SOC_MIN)

            # Check for an unmet load.
            unserved = np.where(need_discharge, deficit - actual_discharge, 0.0)
            failed = need_discharge & (unserved > 1e-6)
            if failed.any():
                idx = np.where(failed)[0]
                for i in idx:
                    if alive[i]:
                        alive[i] = False
                        autonomy[i] = rel_h
                        ae = avail_energy[i]
                        if ae <= 1e-6:
                            fail_reason[i] = "battery_energy_depleted"
                        elif max_discharge[i] < deficit[i]:
                            fail_reason[i] = "battery_power_limited"
                        else:
                            fail_reason[i] = "pv_and_bess_insufficient"

        # --- Scenarios with surplus PV available for charging ---
        surplus = pv_now - pv_to_load
        can_charge = alive & (surplus > 0)
        if can_charge.any():
            remain_cap = np.maximum(0.0, (BESS_SOC_MAX - soc) * total_energy_kwh)
            max_charge_input = remain_cap / BESS_CHARGE_EFF
            max_charge = np.minimum(total_power_kw, max_charge_input)
            actual_charge = np.minimum(surplus, max_charge)
            actual_charge = np.where(can_charge, actual_charge, 0.0)

            soc += np.where(can_charge, actual_charge * BESS_CHARGE_EFF / total_energy_kwh, 0.0)
            soc = np.minimum(soc, BESS_SOC_MAX)

    return {
        "autonomy_hours": autonomy,
        "failure_reasons": fail_reason,
    }


# ╔══════════════════════════════════════════════════════════════╗
# |                          PLOTTING                            |
# ╚══════════════════════════════════════════════════════════════╝

def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # -- Figure 1: off-grid duration vs PV capacity --
    ax = axes[0, 0]
    cons_vals = sorted(df["cons_target_%"].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(cons_vals) - 1, 1)) for i in range(len(cons_vals))]
    for i, cv in enumerate(cons_vals):
        sub = df[df["cons_target_%"] == cv]
        ax.scatter(sub["PV_MW"], sub["min_offgrid_hours"],
                   c=[colors[i]], label=f"{cv}%", s=25, alpha=0.7, edgecolors="none")
    ax.set_xlabel("PV Capacity (MW)")
    ax.set_ylabel("Min Off-grid Duration (hours)")
    ax.set_title("Min Off-grid Duration vs PV Capacity")
    ax.legend(title="cons_target", fontsize=8)
    ax.grid(alpha=0.3)

    # -- Figure 2: off-grid duration vs total BESS capacity --
    ax = axes[0, 1]
    df_plot = df.copy()
    df_plot["total_bess_mwh"] = df_plot["total_MWh"]
    for i, cv in enumerate(cons_vals):
        sub = df_plot[df_plot["cons_target_%"] == cv]
        ax.scatter(sub["total_bess_mwh"], sub["min_offgrid_hours"],
                   c=[colors[i]], label=f"{cv}%", s=25, alpha=0.7, edgecolors="none")
    ax.set_xlabel("Total BESS Energy (MWh)")
    ax.set_ylabel("Min Off-grid Duration (hours)")
    ax.set_title("Min Off-grid Duration vs Total BESS Capacity")
    ax.legend(title="cons_target", fontsize=8)
    ax.grid(alpha=0.3)

    # -- Figure 3: histogram of off-grid durations --
    ax = axes[1, 0]
    ax.hist(df["min_offgrid_hours"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Min Off-grid Duration (hours)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Min Off-grid Duration across {len(df)} Configurations")
    ax.axvline(df["min_offgrid_hours"].median(), color="red", ls="--", label=f"Median={df['min_offgrid_hours'].median():.0f}h")
    ax.legend()
    ax.grid(alpha=0.3)

    # -- Figure 4: off-grid duration heatmap (pivot on cons_target_% x energy_pen_%) --
    ax = axes[1, 1]
    try:
        pivot = df.pivot_table(
            values="min_offgrid_hours",
            index="cons_target_%",
            columns="energy_pen_%",
            aggfunc="mean"
        )
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", origin="lower")
        ax.set_xticks(range(0, pivot.shape[1], max(1, pivot.shape[1] // 10)))
        ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns[::max(1, pivot.shape[1] // 10)]],
                           fontsize=7)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
        ax.set_xlabel("Energy Penetration (%)")
        ax.set_ylabel("Consumption Target (%)")
        ax.set_title("Min Off-grid Hours (mean per cell)")
        plt.colorbar(im, ax=ax, label="Hours")
    except Exception:
        ax.text(0.5, 0.5, "Heatmap data insufficient", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("Off-grid Duration Analysis for PV+BESS Microgrid", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(figures_folder, "offgrid_duration_overview.png"), dpi=200)
    plt.close(fig)
    print(f"[Figures] Saved: {figures_folder}/offgrid_duration_overview.png")

    # -- Additional figure: off-grid duration vs PV, grouped by BESS capacity --
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.copy()
    df_plot["total_bess_mwh"] = df_plot["total_MWh"]
    bess_bins = [0, 5, 20, 100, 720]
    bess_labels = ["0-5 MWh", "5-20 MWh", "20-100 MWh", "100+ MWh"]
    bess_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    df_plot["bess_bucket"] = pd.cut(df_plot["total_bess_mwh"], bins=bess_bins, labels=bess_labels)
    for label, color in zip(bess_labels, bess_colors):
        sub = df_plot[df_plot["bess_bucket"] == label]
        if len(sub) > 0:
            ax.scatter(sub["PV_MW"], sub["min_offgrid_hours"],
                       c=color, label=label, s=30, alpha=0.7, edgecolors="none")
    ax.set_xlabel("PV Capacity (MW)")
    ax.set_ylabel("Min Off-grid Duration (hours)")
    ax.set_title("Off-grid Duration vs PV Capacity (by BESS Size)")
    ax.legend(title="Total BESS")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, "offgrid_vs_pv_by_bess.png"), dpi=200)
    plt.close(fig)
    print(f"[Figures] Saved: {figures_folder}/offgrid_vs_pv_by_bess.png")


# ╔══════════════════════════════════════════════════════════════╗
# |                         MAIN PROGRAM                         |
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  Microgrid off-grid duration analysis")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # -- 1. Load data --
    pv_raw, load = load_8760_data()
    df_sens = load_sensitivity()
    n_hours = 8760

    # -- 2. Generate outage start times --
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
        total_mwh = row["total_MWh"]
        total_mw = row["total_MW"]

        # Scale the PV profile.
        pv_avail = pv_raw * (pv_mw / PV_BASE_CAPACITY_MW)

        # Total BESS capacity, taken directly from the results file.
        total_energy_kwh = total_mwh * 1000.0
        total_power_kw = total_mw * 1000.0

        # Run the simulation.
        sim = simulate_vectorized(
            pv_avail, load, total_energy_kwh, total_power_kw,
            start_hours, alpha=ALPHA, initial_soc=INITIAL_SOC, n_hours=n_hours
        )

        autonomies = sim["autonomy_hours"]
        reasons = sim["failure_reasons"]

        # Index of the worst-case scenario.
        worst_idx = np.argmin(autonomies)

        results["min_offgrid_hours"].append(float(autonomies[worst_idx]))
        results["mean_offgrid_hours"].append(float(np.mean(autonomies)))
        results["p10_offgrid_hours"].append(float(np.percentile(autonomies, 10)))
        results["max_offgrid_hours"].append(float(np.max(autonomies)))
        results["survival_rate"].append(float(np.mean(autonomies >= n_hours)))
        results["worst_start_day"].append(int(start_days[worst_idx]))
        results["worst_start_hour"].append(int(start_hod[worst_idx]))
        results["worst_failure_reason"].append(str(reasons[worst_idx]))

    # -- 4. Merge results and write them back to Excel --
    for col, vals in results.items():
        df_sens[col] = vals

    # Add columns to the original file.
    output_path = INPUT_SENSITIVITY
    # Back up the original file.
    backup_path = INPUT_SENSITIVITY.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(INPUT_SENSITIVITY, backup_path)
        print(f"\n[Backup] Original file backed up to: {backup_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_sens.to_excel(writer, sheet_name="Results", index=False)

    print(f"\n[Output] Results written back to: {output_path}")

    # -- 5. Plot results --
    print("\n[Figures] Generating plots ...")
    plot_results(df_sens, FIGURES_FOLDER)

    # -- 6. Console summary --
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  RUN SUMMARY")
    print(f"{'='*70}")
    print(f"  Elapsed time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Total configurations: {len(df_sens)}")
    print(f"  min_offgrid_hours statistics:")
    print(f"    Minimum: {df_sens['min_offgrid_hours'].min():.1f} h")
    print(f"    Maximum: {df_sens['min_offgrid_hours'].max():.1f} h")
    print(f"    Median: {df_sens['min_offgrid_hours'].median():.1f} h")
    print(f"    Mean:   {df_sens['min_offgrid_hours'].mean():.1f} h")
    print(f"  Configurations surviving the full year (survival_rate=100%): "
          f"{(df_sens['survival_rate'] == 1.0).sum()} / {len(df_sens)}")
    print(f"\n  Excel: {output_path}")
    print(f"  Figures: {FIGURES_FOLDER}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
