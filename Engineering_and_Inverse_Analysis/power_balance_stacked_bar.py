"""
绿氢微电网一天运行数据可视化 - 功率平衡堆叠柱状图 (Nature Style)
晴天和阴天每小时各资源功率堆叠柱状图（正向=供给，负向=需求）
每个功率图下方附带储能状态百分比图（双y轴）

数据源（与本脚本位于同一目录）：typical_sunny_day.xlsx,
typical_cloudy_day.xlsx（半小时间隔，48个点）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import CubicSpline
import os
from pathlib import Path

# ==================== 配置 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR
OUTPUT_PATH = SCRIPT_DIR / "figures"

COLORS = {
    'renewable_gen': '#FFA500',
    'grid': '#4A4A4A',
    'storage_output': '#5287BC',
    'fuelcell_h2': '#0F7057',
    'load': '#B0B0B0',
    'soc': '#5287BC',
    'h_soc': '#0F7057',
}


def safe_spline(x, y, x_new):
    y_clean = np.nan_to_num(y, nan=0.0)
    return CubicSpline(x, y_clean)(x_new)


def add_panel_label(ax, label, x=-0.04, y=1.02, fontsize=12,
                    fontweight='bold', color='black'):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight, color=color,
            ha='left', va='bottom')


def read_daily_data(filename):
    try:
        df = pd.read_excel(DATA_PATH / filename)
        return df
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")
        return None


def compute_hourly_data(df):
    n = min(len(df), 48)

    pv = df['pv'].values[:n]
    grid_purchase = df.get('购电功率', pd.Series([0]*len(df))).values[:n]
    battery_discharge = df['电化学储能放电'].values[:n]
    battery_charge = df['电化学储能充电'].values[:n]
    fuel_cell = df['燃料电池放电'].values[:n]
    h2_production = df['制氢功率'].values[:n]
    charging_station = df['充电桩负荷'].values[:n]
    dc_station_load = df['直流和站用负荷'].values[:n]
    ess_soc = df['ess_soc'].values[:n]
    h_soc = df['h_soc'].values[:n]
    h_load = df['h_load'].values[:n]
    pv_grid_feed = df.get('光伏上网', pd.Series([0]*len(df))).values[:n]

    renewable_gen = pv
    storage_output = battery_discharge - battery_charge
    fuelcell_h2 = fuel_cell - h2_production
    load_total = -(charging_station + dc_station_load)

    # 虚拟百分比：不加氢 = h_soc + cumsum(h_load)，再除以450
    h_soc_virtual = h_soc + np.cumsum(h_load)

    return {
        'power': {
            'renewable_gen': renewable_gen,
            'grid': grid_purchase,
            'storage_output': storage_output,
            'fuelcell_h2': fuelcell_h2,
            'load': load_total,
            'pv_grid_feed': pv_grid_feed,
        },
        'ess_soc_pct': ess_soc / 6000 * 100,
        'h_soc_pct': h_soc / 450 * 100,
        'h_soc_virtual_pct': h_soc_virtual / 450 * 100,
        'h_load': h_load,
    }, n


def setup_xaxis(ax, show_labels=False):
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xticks(np.arange(0, 25, 1), minor=True)
    ax.tick_params(axis='x', which='minor', length=2.5, width=0.5)
    ax.tick_params(axis='x', which='major', length=4.5, width=0.7)
    if show_labels:
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 4)], fontsize=8.5)
        ax.set_xlabel('Time', fontweight='medium')
    else:
        ax.set_xticklabels([])


def plot_stacked_bars(ax, data, n_points, panel_label):
    bar_centers = np.linspace(0, 24, n_points, endpoint=False) + 24 / n_points / 2
    bw = 0.4
    d = data

    pos_ren = d['renewable_gen'].copy()
    pos_grid = d['grid'].copy()
    pos_storage = np.maximum(d['storage_output'], 0)
    pos_fc = np.maximum(d['fuelcell_h2'], 0)

    neg_load = d['load'].copy()
    neg_storage = np.minimum(d['storage_output'], 0)
    neg_fc = np.minimum(d['fuelcell_h2'], 0)
    neg_pv_feed = -d.get('pv_grid_feed', np.zeros(n_points))

    # 正向堆叠 (supply)
    ax.bar(bar_centers, pos_ren, bw, color=COLORS['renewable_gen'],
           label='Renewable Gen', edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, pos_grid, bw, bottom=pos_ren, color=COLORS['grid'],
           label='Grid Purchase', edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, pos_storage, bw, bottom=pos_ren + pos_grid,
           color=COLORS['storage_output'], label='Storage Output',
           edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, pos_fc, bw, bottom=pos_ren + pos_grid + pos_storage,
           color=COLORS['fuelcell_h2'], label='Fuel Cell / H$_2$ Prod.',
           edgecolor='white', linewidth=0.3)

    # 负向堆叠 (demand)
    ax.bar(bar_centers, neg_load, bw, color=COLORS['load'],
           label='Load', edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, neg_pv_feed, bw, bottom=neg_load,
           color=COLORS['grid'], label='PV Grid Feed-in',
           edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, neg_storage, bw, bottom=neg_load + neg_pv_feed,
           color=COLORS['storage_output'], edgecolor='white', linewidth=0.3)
    ax.bar(bar_centers, neg_fc, bw, bottom=neg_load + neg_pv_feed + neg_storage,
           color=COLORS['fuelcell_h2'], edgecolor='white', linewidth=0.3)

    ax.axhline(y=0, color='black', linewidth=0.8, zorder=5)
    ax.set_ylabel('Power (kW)', fontweight='medium')
    ax.set_ylim(-4000, 4000)
    ax.set_yticks(np.arange(-4000, 4001, 1000))
    setup_xaxis(ax, show_labels=False)

    add_panel_label(ax, panel_label)


def plot_battery_soc(ax, ess_soc_pct, n_points, panel_label,
                     show_xlabel=False):
    bar_centers = np.linspace(0, 24, n_points, endpoint=False) + 24 / n_points / 2
    t_smooth = np.linspace(0, 24, 300)

    soc_color = COLORS['soc']

    y_soc_smooth = safe_spline(bar_centers, ess_soc_pct, t_smooth)
    ax.fill_between(t_smooth, 0, y_soc_smooth, color=soc_color, alpha=0.12)
    ax.plot(t_smooth, y_soc_smooth, color=soc_color, linewidth=2.2,
            label='Battery SOC', solid_capstyle='round')
    ax.set_ylabel('Battery SOC (%)', color=soc_color, fontweight='medium')
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis='y', labelcolor=soc_color, labelsize=9,
                   width=0.8, length=4)

    setup_xaxis(ax, show_labels=show_xlabel)
    add_panel_label(ax, panel_label)


def plot_h2_soc(ax, h_soc_pct, h_load, n_points, panel_label,
                show_xlabel=False):
    bar_centers = np.linspace(0, 24, n_points, endpoint=False) + 24 / n_points / 2
    t_smooth = np.linspace(0, 24, 300)

    h2_color = '#0F7057'
    refuel_color = '#7FBDA8'

    # 左轴：实际储氢百分比 — 实线 + 半透明填充
    y_h2_smooth = safe_spline(bar_centers, h_soc_pct, t_smooth)
    ax.plot(t_smooth, y_h2_smooth, color=h2_color, linewidth=2.0,
            label='H$_2$ SOC', solid_capstyle='round')
    ax.fill_between(t_smooth, 0, y_h2_smooth, color=h2_color, alpha=0.10)

    ax.set_ylabel('H$_2$ Storage (%)', color=h2_color, fontweight='medium')
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis='y', labelcolor=h2_color, labelsize=9,
                   width=0.8, length=4)

    # 右轴：加氢量柱子
    ax_right = ax.twinx()
    refuel_mask = h_load > 0
    if np.any(refuel_mask):
        refuel_centers = bar_centers[refuel_mask]
        refuel_values = h_load[refuel_mask]
        ax_right.bar(refuel_centers, refuel_values, width=0.4, color=refuel_color,
                     alpha=0.85, label='Refuel', edgecolor='darkgreen', linewidth=0.8)

    ax_right.set_ylabel('Refuel (kg)', color=refuel_color, fontweight='medium')
    ax_right.set_ylim(0, 20)
    ax_right.set_yticks([0, 10, 20])
    ax_right.tick_params(axis='y', labelcolor=refuel_color, labelsize=9,
                        width=0.8, length=4)

    # 100% 参考线
    ax.axhline(y=100, color='#888888', linestyle='--', linewidth=1.0, zorder=1)

    setup_xaxis(ax, show_labels=show_xlabel)
    add_panel_label(ax, panel_label)


def main():
    # ── Nature 出版级 rcParams ──────────────────────────────────────────────
    # MANDATORY: editable SVG text
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'

    # Layout & style
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.08
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['legend.frameon'] = False

    # Ticks
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.major.width'] = 0.72
    plt.rcParams['ytick.major.width'] = 0.72
    plt.rcParams['xtick.major.size'] = 5.25
    plt.rcParams['ytick.major.size'] = 5.25

    # ── 读取数据 ────────────────────────────────────────────────────────────
    df_sunny = read_daily_data("typical_sunny_day.xlsx")
    df_cloudy = read_daily_data("typical_cloudy_day.xlsx")

    if df_sunny is None or df_cloudy is None:
        print("数据读取失败！")
        return

    data_sunny, n_h1 = compute_hourly_data(df_sunny)
    data_cloudy, n_h2 = compute_hourly_data(df_cloudy)

    print(f"晴天：{n_h1} 个点，阴天：{n_h2} 个点")

    # ── 创建图表 (3×2: 左列晴天, 右列阴天) ─────────────────────────────────
    fig = plt.figure(figsize=(32, 11.2), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.6, 0.4, 0.5],
                           hspace=0.30, wspace=0.25,
                           left=0.07, right=0.66, top=0.95, bottom=0.06)

    ax_p1 = fig.add_subplot(gs[0, 0])   # 晴天功率
    ax_b1 = fig.add_subplot(gs[1, 0])   # 晴天Battery SOC
    ax_h1 = fig.add_subplot(gs[2, 0])   # 晴天H2 SOC
    ax_p2 = fig.add_subplot(gs[0, 1])   # 阴天功率
    ax_b2 = fig.add_subplot(gs[1, 1])   # 阴天Battery SOC
    ax_h2 = fig.add_subplot(gs[2, 1])   # 阴天H2 SOC

    # ---- 晴天 (左列) ----
    plot_stacked_bars(ax_p1, data_sunny['power'], n_h1, 'a')
    plot_battery_soc(ax_b1, data_sunny['ess_soc_pct'], n_h1, 'b')
    plot_h2_soc(ax_h1, data_sunny['h_soc_pct'],
                data_sunny['h_load'], n_h1, 'c', show_xlabel=True)

    # ---- 阴天 (右列) ----
    plot_stacked_bars(ax_p2, data_cloudy['power'], n_h2, 'd')
    plot_battery_soc(ax_b2, data_cloudy['ess_soc_pct'], n_h2, 'e')
    plot_h2_soc(ax_h2, data_cloudy['h_soc_pct'],
                data_cloudy['h_load'], n_h2, 'f', show_xlabel=True)

    # ---- 统一图例（功率 + SOC 合并，去重）----
    all_handles, all_labels = [], []
    for ax in [ax_p1, ax_b1, ax_h1]:
        h, l = ax.get_legend_handles_labels()
        all_handles += list(h)
        all_labels += list(l)
    by_label = dict(zip(all_labels, all_handles))
    fig.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(0.68, 0.5), loc='center left', fontsize=9,
               frameon=False)

    # ── 导出 ────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    base = OUTPUT_PATH / "figure_stacked_bar_hourly"
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"图表已保存到 {OUTPUT_PATH}/")
    print(f"  - {base}.svg")
    print(f"  - {base}.png")


if __name__ == "__main__":
    main()
    print("\n完成!")
