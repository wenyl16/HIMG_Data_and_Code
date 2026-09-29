"""
Visualize a 168-hour off-grid green-hydrogen microgrid test: power balance (Nature style).
H2+BESS configuration with PV generation and no wind power.

Three panels show power balance, storage states on dual y axes, and daily charging/discharging shares.

Data sources (in the same directory as this script):seven_day_offgrid_H2_BESS.xlsx

Four power curves:
- Renewable Gen = PV (positive)
- Storage Output = battery discharge minus battery charge (positive/negative)
- Fuel Cell Output/H2 Production = fuel cell output minus electrolyzer input (positive/negative)
- Load = load (negative)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from pathlib import Path

# ==================== Configuration ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "seven_day_offgrid_H2_BESS.xlsx"
OUTPUT_PATH = SCRIPT_DIR / "figures"

# Colours for the four curves
COLORS = {
    'renewable_gen': '#FFA500',
    'storage_output': '#5287BC',
    'fuelcell_h2': '#0F7057',
    'load': '#B0B0B0',
    'soc': '#5287BC',
    'h2_soc': '#0F7057',
}


# ==================== Main plotting function ====================

def plot_figure():
    """Create Figure 1: power balance, storage states and daily charging/discharging shares."""

    # ==================== Publication figure settings ====================
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.8,
        'lines.linewidth': 1.8,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.framealpha': 0.95,
        'legend.fontsize': 7.5,
        'legend.edgecolor': '#666666',
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5.25,
        'ytick.major.size': 5.25,
        'axes.unicode_minus': False,
        'axes.edgecolor': '#333333',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'axes.labelcolor': '#000000',
    })

    print("Reading data...")
    df = pd.read_excel(DATA_PATH, sheet_name='SimulationResults', header=0)
    N_POINTS = len(df)
    t = df['Time_h'].values

    # ========== Calculate the four curves ==========
    # 1. Renewable Gen = PV (positive)
    renewable_gen = df['PV_kW'].values

    # 2. Storage Output = -Battery_kW(battery power is positive for charging; negation makes discharge positive)
    storage_output = -df['Battery_kW'].values

    # 3. Fuel Cell Output/H2 Production = FuelCell - Electrolyzer
    fuelcell_h2 = df['FuelCell_kW'].values - df['Electrolyzer_kW'].values

    # 4. Load = -Load_kW(negative)
    load_total = -df['Load_kW'].values

    # Storage states
    battery_soc = df['Battery_SOC_pct'].values
    h2_soc = df['H2_SOC_pct'].values

    print(f"  Data loaded: {N_POINTS} time points")

    # ==================== Create the figure ====================
    print("Generating Figure 1 (Nature style)...")

    fig = plt.figure(figsize=(15.4, 9.5), dpi=300)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.6, 0.5, 0.5], hspace=0.30,
                           left=0.09, right=0.72, top=0.94, bottom=0.10)

    # ========== Panel (a): power balance ==========
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t, renewable_gen, color=COLORS['renewable_gen'], linewidth=3.6,
             label='Renewable Gen', solid_capstyle='round')
    ax1.plot(t, storage_output, color=COLORS['storage_output'], linewidth=3.6,
             label='Storage Output', solid_capstyle='round')
    ax1.plot(t, fuelcell_h2, color=COLORS['fuelcell_h2'], linewidth=3.6,
             label='Fuel Cell Output/H$_2$ Production', solid_capstyle='round')
    ax1.plot(t, load_total, color=COLORS['load'], linewidth=3.6, linestyle='--',
             label='Load', solid_capstyle='round')

    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=3)

    ax1.set_ylabel('Power (kW)', fontweight='medium')
    ax1.set_xlim(0, 168)
    ax1.set_ylim(-3000, 4000)
    ax1.set_yticks(np.arange(-3000, 4001, 1000))
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels([])
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', width=1.2, length=5.25, labelsize=9, color='#333333')

    ax1.legend(bbox_to_anchor=(1.10, 0.5), loc='center left', fontsize=7,
               frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.9)

    # ========== Panel (b): comparison of storage states ==========
    ax2 = fig.add_subplot(gs[1])

    # Battery SOC
    soc_color = COLORS['soc']
    ax2.plot(t, battery_soc, color=soc_color, linewidth=3.6,
             label='Battery SOC', solid_capstyle='round')

    # H2 SOC
    h2_color = COLORS['h2_soc']
    ax2.plot(t, h2_soc, color=h2_color, linewidth=3.6,
             label='H$_2$ SOC', solid_capstyle='round')

    ax2.set_ylabel('SOC (%)', fontweight='medium')
    ax2.set_xlim(0, 168)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    ax2.tick_params(axis='x', labelsize=9, width=1.2, length=5.25)
    ax2.tick_params(axis='y', labelsize=9, width=1.2, length=5.25)
    ax2.grid(False)

    ax2.set_xticks(np.arange(0, 169, 24))
    ax2.set_xticklabels([])

    ax2.legend(bbox_to_anchor=(1.10, 0.5), loc='center left', fontsize=7,
               frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.9)

    # ========== Panel (c): daily charging/discharging shares ==========
    ax3 = fig.add_subplot(gs[2])

    battery_kw = df['Battery_kW'].values
    fc_kw = df['FuelCell_kW'].values
    elec_kw = df['Electrolyzer_kW'].values

    # Set power values below 1 to zero
    battery_kw = np.where(np.abs(battery_kw) < 1, 0, battery_kw)
    fc_kw = np.where(np.abs(fc_kw) < 1, 0, fc_kw)
    elec_kw = np.where(np.abs(elec_kw) < 1, 0, elec_kw)

    # Battery_kW > 0 means charging; < 0 means discharging
    batt_charge = np.where(battery_kw > 0, battery_kw, 0)
    batt_discharge = np.where(battery_kw < 0, -battery_kw, 0)

    pts_per_day = 48  # 0.5 h intervals, 48 points per day
    n_days = N_POINTS // pts_per_day

    daily_batt_c, daily_batt_d, daily_h2, daily_fc = [], [], [], []
    for day in range(n_days):
        s, e = day * pts_per_day, (day + 1) * pts_per_day
        daily_batt_c.append(np.sum(batt_charge[s:e]) * 0.5)
        daily_batt_d.append(np.sum(batt_discharge[s:e]) * 0.5)
        daily_fc.append(np.sum(fc_kw[s:e]) * 0.5)
        daily_h2.append(np.sum(elec_kw[s:e]) * 0.5)

    # Charging side: battery charging plus hydrogen production
    c_total = [b + h for b, h in zip(daily_batt_c, daily_h2)]
    # Discharging side: battery discharge plus fuel cell output
    d_total = [b + f for b, f in zip(daily_batt_d, daily_fc)]

    c_batt_pct = [b / t * 100 if t > 0 else 0 for b, t in zip(daily_batt_c, c_total)]
    c_h2_pct = [h / t * 100 if t > 0 else 0 for h, t in zip(daily_h2, c_total)]
    d_batt_pct = [b / t * 100 if t > 0 else 0 for b, t in zip(daily_batt_d, d_total)]
    d_fc_pct = [f / t * 100 if t > 0 else 0 for f, t in zip(daily_fc, d_total)]

    days = np.arange(n_days)
    w = 0.3
    gap = 0.08

    # Charging bars (solid, shifted left)
    ax3.bar(days - w/2 - gap/2, c_batt_pct, width=w, color=COLORS['storage_output'],
            edgecolor='white', linewidth=0.5, label='Battery (Charge)')
    ax3.bar(days - w/2 - gap/2, c_h2_pct, width=w, bottom=c_batt_pct, color=COLORS['fuelcell_h2'],
            edgecolor='white', linewidth=0.5, label='H$_2$ Prod. (Charge)')

    # Discharging bars (hatched, shifted right)
    ax3.bar(days + w/2 + gap/2, d_batt_pct, width=w, color=COLORS['storage_output'],
            hatch='//', edgecolor='white', linewidth=0.5, label='Battery (Discharge)')
    ax3.bar(days + w/2 + gap/2, d_fc_pct, width=w, bottom=d_batt_pct, color=COLORS['fuelcell_h2'],
            hatch='//', edgecolor='white', linewidth=0.5, label='Fuel Cell (Discharge)')

    ax3.set_xlim(-0.5, n_days - 0.5)
    ax3.set_xticks(np.arange(-0.5, n_days + 0.5, 1))
    ax3.set_xticklabels([])
    ax3.set_ylabel('Ratio (%)', fontweight='medium')
    ax3.set_ylim(0, 100)
    ax3.set_yticks([0, 50, 100])
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', width=1.2, length=5.25, labelsize=9, color='#333333')

    ax3.legend(bbox_to_anchor=(1.10, 0.5), loc='center left', fontsize=7,
               frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.9)

    # Print daily shares
    print(f"\n========== Daily charging/discharging shares ==========")
    for i in range(n_days):
        print(f"  Day {i+1} discharging: Battery={daily_batt_d[i]:.1f} kWh ({d_batt_pct[i]:.1f}%), "
              f"Fuel Cell={daily_fc[i]:.1f} kWh ({d_fc_pct[i]:.1f}%)")
        print(f"  Day {i+1} charging: Battery={daily_batt_c[i]:.1f} kWh ({c_batt_pct[i]:.1f}%), "
              f"H2 Prod.={daily_h2[i]:.1f} kWh ({c_h2_pct[i]:.1f}%)")

    # Save the figure
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.savefig(OUTPUT_PATH / "figure1_power_balance_H2_BESS.svg", bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / "figure1_power_balance_H2_BESS.png", dpi=300, bbox_inches='tight')

    print(f"\nFigure 1 saved to {OUTPUT_PATH}/")
    print("  - figure1_power_balance_H2_BESS.svg")
    print("  - figure1_power_balance_H2_BESS.png")

    plt.close()


if __name__ == "__main__":
    plot_figure()
    print("\nCompleted!")
