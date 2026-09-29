"""
纯电池储能微电网168小时离网测试数据可视化 (Nature风格)
两种BESS配置对比：enough_BESS（充足电池）和 original_BESS（原始电池）

包含2个子图：功率平衡曲线图、电池SOC状态图（无氢储能相关曲线）

数据源（与本脚本位于同一目录）：
  enough_BESS.xlsx
  original_BESS.xlsx

功率曲线（3条）：
- Renewable Gen = 光伏（正值）
- Storage Output = 电池放电 - 电池充电（正/负）
- Load = 负荷（负值）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from pathlib import Path

# ==================== 配置 ====================
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


# ==================== 绘图函数 ====================

def plot_bess_figure(key):
    """为单个BESS配置生成图：功率平衡 + 电池SOC"""

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
    print(f"读取 {LABELS[key]} 数据...")
    df = pd.read_excel(FILES[key], sheet_name='SimulationResults', header=0)
    N_DATA = len(df)

    # 不足336点（168h）时，功率补零，SOC延续原始数据的最后一个值
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
        print(f"  原始 {N_DATA} 点，补全至 {N_FULL} 点（168h）；"
              f"功率补零，补全段SOC保持为 {last_soc:.6f}%")
    else:
        print(f"  共 {N_DATA} 个时间点，完整168小时")

    t = df['Time_h'].values
    T_MAX = 168.0

    # ========== 计算3条曲线 ==========
    renewable_gen = df['PV_kW'].values
    storage_output = -df['Battery_kW'].values
    load_total = -df['Load_kW'].values
    battery_soc = df['Battery_SOC_pct'].values

    # ==================== 创建图表 ====================
    print(f"生成 {LABELS[key]} 图表...")

    fig = plt.figure(figsize=(15.4, 6.5), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.6, 0.6], hspace=0.28,
                           left=0.09, right=0.72, top=0.94, bottom=0.10)

    # ========== 子图(a): 功率平衡曲线图 ==========
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

    # ========== 子图(b): 电池SOC ==========
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

    # ========== 保存 ==========
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    outname_svg = f"figure_power_balance_{key}_BESS.svg"
    outname_png = f"figure_power_balance_{key}_BESS.png"
    plt.savefig(OUTPUT_PATH / outname_svg, bbox_inches='tight')
    plt.savefig(OUTPUT_PATH / outname_png, dpi=300, bbox_inches='tight')

    print(f"  已保存到 {OUTPUT_PATH}/{outname_svg}")
    print(f"  已保存到 {OUTPUT_PATH}/{outname_png}")
    plt.close()


if __name__ == "__main__":
    for key in ['enough', 'original']:
        plot_bess_figure(key)
    print("\n全部完成!")
