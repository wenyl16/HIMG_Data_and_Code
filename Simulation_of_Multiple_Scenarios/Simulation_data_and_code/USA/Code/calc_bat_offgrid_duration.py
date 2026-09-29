#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_bat_offgrid_duration.py
========================

For each PV + BESS capacity configuration in sensitivity_bat_sizing_merged_results.xlsx,
simulate microgrid off-grid duration using the annual 8760-hour data in Annual_8760.xlsx.

Core procedure:
  - 365 days x 4 times per day (0/6/12/18) = 1460 outage starting times.
  - Start each event at SOC=0.9 and simulate hourly: PV supply first, followed by battery charging or discharge.
  - Use the minimum duration across 1460 events as the off-grid capability metric.
  - Add the results as new columns in the original Excel file.
"""

import os
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

PV_BASE_CAPACITY_MW = 0.046  # USA measured peak reference: 46 kW, matching sizing PV_RATED_ORIGINAL.
BASE_BESS_POWER_KW = 0
BASE_BESS_ENERGY_KWH = 0
BESS_CHARGE_EFF = 0.95
BESS_DISCHARGE_EFF = 0.95
BESS_SOC_MIN = 0.10
BESS_SOC_MAX = 0.90
ALPHA = 1.0
INITIAL_SOC = 0.9
START_HOURS_OF_DAY = [0, 6, 12, 18]

INPUT_8760 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annual_8760_clean.xlsx")
INPUT_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_bat_sizing_merged_results.xlsx")
FIGURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offgrid_duration_figures")


# ╔══════════════════════════════════════════════════════════════╗
#                       Data loading
# ╚══════════════════════════════════════════════════════════════╝

def load_8760_data(filepath=INPUT_8760):
    print(f"[Data] Reading {filepath} ...")
    df_pv = pd.read_excel(filepath, sheet_name="pv_kw", header=None, skiprows=1)
    pv_raw = pd.to_numeric(df_pv.iloc[:, 1], errors="coerce").values.astype(float)
    df_load = pd.read_excel(filepath, sheet_name="load_kw", header=None, skiprows=1)
    load = pd.to_numeric(df_load.iloc[:, 1], errors="coerce").values.astype(float)

    assert len(pv_raw) == 8760, f"PV length {len(pv_raw)} != 8760"
    assert len(load) == 8760, f"Load length {len(load)} != 8760"

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
    print(f"  {len(df)} configurations")
    return df


def generate_start_info(n_hours=8760):
    """Generate indices and metadata for 1460 outage starting times."""
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
    print(f"[Outage starts] {len(start_hours)} starting times")
    return np.array(start_hours), np.array(start_days), np.array(start_hod)


# ╔══════════════════════════════════════════════════════════════╗
#                       Vectorized simulation
# ╚══════════════════════════════════════════════════════════════╝

def simulate_vectorized(pv_avail, load, total_energy_kwh, total_power_kw,
                        start_hours, alpha=ALPHA, initial_soc=INITIAL_SOC,
                        n_hours=8760):
    """
    Simulate all outage starting times simultaneously, hour by hour, using NumPy vectorization.

    Return a dict with:
      autonomy_hours: (n_events,) off-grid duration of each event.
      failure_reasons: (n_events,) cause of failure for each event.
    """
    n_events = len(start_hours)
    max_sim = n_hours  # Maximum simulation horizon: one full year.

    # State of each outage event
    soc = np.full(n_events, initial_soc, dtype=np.float64)
    alive = np.ones(n_events, dtype=bool)  # Events still operating
    autonomy = np.full(n_events, max_sim, dtype=np.float64)  # Default to survival through the full simulation horizon.
    fail_reason = np.array([""] * n_events, dtype=object)

    for rel_h in range(max_sim):
        if not alive.any():
            break

        # Absolute hourly indices, wrapping around the annual series.
        t_indices = (start_hours + rel_h) % n_hours

        # Read PV and load for the current hour; update only active events.
        pv_now = pv_avail[t_indices]
        load_now = alpha * load[t_indices]

        # Supply load from PV first.
        pv_to_load = np.minimum(pv_now, load_now)
        deficit = load_now - pv_to_load

        # Events requiring battery discharge
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

            # Check for unmet load.
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

        # Events with surplus PV available for charging
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
#                       Plotting
# ╚══════════════════════════════════════════════════════════════╝

def plot_results(df, figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Figure 1: Off-grid duration vs PV capacity scatter plot
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

    # Figure 2: Off-grid duration vs total BESS capacity scatter plot
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

    # Figure 3: Histogram of off-grid duration
    ax = axes[1, 0]
    ax.hist(df["min_offgrid_hours"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Min Off-grid Duration (hours)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Min Off-grid Duration across {len(df)} Configurations")
    ax.axvline(df["min_offgrid_hours"].median(), color="red", ls="--", label=f"Median={df['min_offgrid_hours'].median():.0f}h")
    ax.legend()
    ax.grid(alpha=0.3)

    # Figure 4: Off-grid duration heatmap, pivoted on cons_target_% x energy_pen_%
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

    # Additional figure: Off-grid duration vs PV capacity, grouped by BESS size.
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
#                       Main program
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  Microgrid off-grid duration analysis")
    print("=" * 70)
    start_time = datetime.datetime.now()

    # 1. Load data.
    pv_raw, load = load_8760_data()
    df_sens = load_sensitivity()
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
        pv_mw = row["PV_MW"]
        total_mwh = row["total_MWh"]
        total_mw = row["total_MW"]

        # Scale PV generation.
        pv_avail = pv_raw * (pv_mw / PV_BASE_CAPACITY_MW)

        # Total BESS capacity, read directly from the results file.
        total_energy_kwh = total_mwh * 1000.0
        total_power_kw = total_mw * 1000.0

        # Run the simulation.
        sim = simulate_vectorized(
            pv_avail, load, total_energy_kwh, total_power_kw,
            start_hours, alpha=ALPHA, initial_soc=INITIAL_SOC, n_hours=n_hours
        )

        autonomies = sim["autonomy_hours"]
        reasons = sim["failure_reasons"]

        # Index of the worst-case event
        worst_idx = np.argmin(autonomies)

        results["min_offgrid_hours"].append(float(autonomies[worst_idx]))
        results["mean_offgrid_hours"].append(float(np.mean(autonomies)))
        results["p10_offgrid_hours"].append(float(np.percentile(autonomies, 10)))
        results["max_offgrid_hours"].append(float(np.max(autonomies)))
        results["survival_rate"].append(float(np.mean(autonomies >= n_hours)))
        results["worst_start_day"].append(int(start_days[worst_idx]))
        results["worst_start_hour"].append(int(start_hod[worst_idx]))
        results["worst_failure_reason"].append(str(reasons[worst_idx]))

    # 4. Merge results and write them back to Excel.
    for col, vals in results.items():
        df_sens[col] = vals

    # Write new columns to the original file.
    output_path = INPUT_SENSITIVITY
    # Back up the original file.
    backup_path = INPUT_SENSITIVITY.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(INPUT_SENSITIVITY, backup_path)
        print(f"\n[Backup] Original file saved to: {backup_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_sens.to_excel(writer, sheet_name="Results", index=False)

    print(f"\n[Output] Results written to: {output_path}")

    # 5. Plot results.
    print("\n[Figures] Creating plots ...")
    plot_results(df_sens, FIGURES_FOLDER)

    # 6. Console summary
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  Run summary")
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
