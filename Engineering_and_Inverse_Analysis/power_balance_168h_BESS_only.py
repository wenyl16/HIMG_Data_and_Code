"""
Visualize a 168-hour off-grid test of a battery-only microgrid (Nature style).
Compare two BESS configurations: enough_BESS (sufficient capacity) and original_BESS (original capacity).

Two panels show power balance and battery SOC, without hydrogen-storage curves.

Data sources (in the same directory as this script):
  enough_BESS.xlsx
  original_BESS.xlsx

Three power curves:
- Renewable Gen = PV (positive)
- Storage Output = battery discharge minus battery charge (positive/negative)
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
OUTPUT_PATH = SCRIPT_DIR / "figures"

FILES = {
    'enough': SCRIPT_DIR / "enough_BESS.xlsx",
    'original': SCRIPT_DIR / "original_BESS.xlsx",
}

COLORS = {
    'renewable_gen': '#FFA500',
    'storage_output': '#5287BC',
    'load': '#B0B0B0',
    'soc': '#5287BC',
}

LABELS = {
    'enough': 'Enough BESS',
    'original': 'Original BESS',
}


# ==================== Plotting function ====================

def plot_bess_figure(key):
    """Plot power balance and battery SOC for one BESS configuration."""

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

    print(f"\n{'='*50}")
    print(f"Reading {LABELS[key]} data...")
    df = pd.read_excel(FILES[key], sheet_name='SimulationResults', header=0)
    N_DATA = len(df)

    # If fewer than 336 points (168 h) are available, pad power with zeros and retain the final SOC.
    N_FULL = 336
    if N_DATA < N_FULL:
        pad_len = N_FULL - N_DATA
        pad_time = np.arange(N_DATA, N_FULL) * 0.5
        last_soc = df['Battery_SOC_pct'].iloc[-1] if N_DATA > 0 else 0
        pad_df = pd.DataFrame({
            'Time_h': pad_time,
            'PV_kW': np.zeros(pad_len),
            'Load_kW': np.zeros(pad_len),
            'Battery_kW': np.zeros(pad_len),
            'Battery_SOC_pct': np.full(pad_len, last_soc),
        })
        df = pd.concat([df, pad_df], ignore_index=True)
        print(f"  Original {N_DATA} points; padded to {N_FULL} points (168 h); "
              f"power padded with zeros and SOC held at {last_soc:.6f}%")
    else:
        print(f"  {N_DATA} time points covering the full 168 hours")

    t = df['Time_h'].values
    T_MAX = 168.0

    # ========== Calculate the three curves ==========
    renewable_gen = df['PV_kW'].values
    storage_output = -df['Battery_kW'].values
    load_total = -df['Load_kW'].values
    battery_soc = df['Battery_SOC_pct'].values

    # ==================== Create the figure ====================
    print(f"Generating the {LABELS[key]} figure...")

    fig = plt.figure(figsize=(15.4, 6.5), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.6, 0.6], hspace=0.28,
                           left=0.09, right=0.72, top=0.94, bottom=0.10)

    # ========== Panel (a): power balance ==========
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t, renewable_gen, color=COLORS['renewable_gen'], linewidth=3.6,
             label='Renewable Gen', solid_capstyle='round')
    ax1.plot(t, storage_output, color=COLORS['storage_output'], linewidth=3.6,
             label='Storage Output', solid_capstyle='round')
    ax1.plot(t, load_total, color=COLORS['load'], linewidth=3.6, linestyle='--',
             label='Load', solid_capstyle='round')

    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=3)

    ax1.set_ylabel('Power (kW)', fontweight='medium')
    ax1.set_xlim(0, T_MAX)
    ax1.set_ylim(-3000, 4000)
    ax1.set_yticks(np.arange(-3000, 4001, 1000))
    ax1.set_xticks(np.arange(0, int(T_MAX) + 1, 24))
    ax1.set_xticklabels([])
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', width=1.2, length=5.25, labelsize=9, color='#333333')

    ax1.legend(bbox_to_anchor=(1.10, 0.5), loc='center left', fontsize=7,
               frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.9)

    # ========== Panel (b): battery SOC ==========
    ax2 = fig.add_subplot(gs[1])

    soc_color = COLORS['soc']
    ax2.plot(t, battery_soc, color=soc_color, linewidth=3.6,
             label='Battery SOC', solid_capstyle='round')

    ax2.set_ylabel('Battery SOC (%)', color=soc_color, fontweight='medium')
    ax2.set_xlim(0, T_MAX)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    ax2.set_xticks(np.arange(0, int(T_MAX) + 1, 24))
    ax2.set_xticklabels([])
    ax2.tick_params(axis='both', which='major', width=1.2, length=5.25, labelsize=9, color='#333333')
    ax2.grid(False)

    ax2.legend(bbox_to_anchor=(1.10, 0.5), loc='center left', fontsize=7,
               frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.9)

    # ========== Save ==========
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    outname_svg = f"figure_power_balance_{key}_BESS.svg"
    outname_png = f"figure_power_balance_{key}_BESS.png"
    plt.savefig(OUTPUT_PATH / outname_svg, bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / outname_png, dpi=300, bbox_inches='tight')

    print(f"  Saved to {OUTPUT_PATH}/{outname_svg}")
    print(f"  Saved to {OUTPUT_PATH}/{outname_png}")
    plt.close()


if __name__ == "__main__":
    for key in ['enough', 'original']:
        plot_bess_figure(key)
    print("\nAll figures completed!")
