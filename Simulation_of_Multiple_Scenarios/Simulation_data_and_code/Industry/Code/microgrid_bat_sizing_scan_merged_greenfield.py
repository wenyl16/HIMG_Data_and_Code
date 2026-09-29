"""
光储微电网 PV容量扫描 + 电池储能容量优化 (单电池合并版)
扫描PV容量 × 光伏消纳率目标(60%~100%)
单一电池储能: 总容量 E_TOTAL 为决策变量, 功率 = E_TOTAL/2 (能量:功率=2:1)
Greenfield 版: 无原有电池, 从0开始选型 (纯弃光方案), E_BAT_MAX = 0
用绝对能量 E_bat(t) 跟踪SOC(避免变量相除的双线性, 全部线性约束)
单组充放电互斥约束(MILP): 作为整体禁止同时充放电
目标: 最小化全系统年化总成本; 求解器 Gurobi
"""

import os
import openpyxl
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 系统参数
# ============================================================

# --- 光伏 ---
PV_RATED_ORIGINAL = 1000     # kW (1MW)
PV_CAPEX = 530               # $/kW
PV_LIFE = 25
PV_OM_RATE = 0.015

# --- PV扫描 ---
PV_SCAN_MIN = 10            # kW
PV_SCAN_MAX = 30000           # kW
PV_SCAN_STEP = 100            # kW

# 壁边(可行性墙)加密: 60%消纳率在~30.45MW处有≥200h窄峰(30.34-30.40MW), 100kW步长会跨过;
# 在 [WALL_LO, WALL_HI] 内追加 20kW 细步长以捕获该窄峰。build_pv_values() 由本模块与并行版共用。
WALL_LO, WALL_HI, WALL_STEP = 30000, 30600, 20   # kW

def build_pv_values():
    coarse = list(range(PV_SCAN_MIN, PV_SCAN_MAX + PV_SCAN_STEP, PV_SCAN_STEP))
    fine   = list(range(WALL_LO, WALL_HI + WALL_STEP, WALL_STEP))
    return sorted(set(coarse + fine))

# --- 单一电池储能 ---
E_BAT_MAX = 0                 # kWh —— Greenfield: 无原有电池下界, 从0优化
BAT_POWER_RATIO = 2           # 能量:功率 = 2:1, 功率 = E_TOTAL / 2
BAT_CAPEX = 250               # $/kWh
BAT_LIFE = 8
BAT_OM_RATE = 0.01
BAT_PRICE_FACTORS = [1.0]             # Greenfield: 仅基准成本因子

ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95
SOC_BAT_MIN = 0.1
SOC_BAT_MAX = 0.9
SOC_BAT_INIT = 0.5

# --- 电价 (RMB/kWh, 使用时 /7 转 USD) ---
PRICE_VALLEY      = 0.451 / 7    # 低谷
PRICE_FLAT        = 0.738 / 7    # 平段
PRICE_PEAK        = 1.144 / 7    # 高峰
PRICE_SHARP       = 1.24 / 7     # 尖峰 (仅夏冬季)
PRICE_DEEP_VALLEY = 0.355 / 7    # 深谷 (时段表未分配, 备用)

# --- 经济参数 ---
DISCOUNT_RATE = 0.05

# --- 消纳率目标 ---
CONSUMPTION_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]
CONSUMPTION_TOL = 0.0005       # 0.05%  (所有消纳率目标统一)

T = 8760


# ============================================================
# 2. 工具函数
# ============================================================

def crf(rate, years):
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def load_raw_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annual_8760.xlsx")
    wb = openpyxl.load_workbook(filepath, read_only=True)
    pv_raw = [row[1] for row in wb["PV"].iter_rows(min_row=1, max_row=T, max_col=2, values_only=True)]
    load = [row[1] for row in wb["Load"].iter_rows(min_row=1, max_row=T, max_col=2, values_only=True)]
    wb.close()
    return pv_raw, load


# 各月天数(非闰年)及起始小时, 用于由全年小时序号反查月份 (hour 0 = 1月1日00:00)
_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_START = [0]
for _d in _DAYS:
    _MONTH_START.append(_MONTH_START[-1] + _d * 24)
_SPRING_AUTUMN = {2, 3, 4, 5, 6, 9, 10, 11}   # 其余 {1, 7, 8, 12} 为夏冬季


def _month_of(hour_idx):
    for _m in range(12):
        if _MONTH_START[_m] <= hour_idx < _MONTH_START[_m + 1]:
            return _m + 1
    return 12


def build_prices():
    # 季节性分时电价: 春秋季(月2-6,9-11)与夏冬季(月1,7-8,12)时段不同
    prices = []
    for i in range(T):
        h = i % 24
        if _month_of(i) in _SPRING_AUTUMN:                        # 春秋季
            if   16 <= h < 23:                                    p = PRICE_PEAK    # 高峰 16-23
            elif (7 <= h < 11) or (14 <= h < 16) or (h == 23):    p = PRICE_FLAT    # 平段
            else:                                                 p = PRICE_VALLEY  # 低谷 0-7, 11-14
        else:                                                      # 夏冬季
            if   18 <= h < 22:                                    p = PRICE_SHARP   # 尖峰 18-22
            elif (16 <= h < 18) or (22 <= h < 23):                p = PRICE_PEAK    # 高峰 16-18, 22-23
            elif (7 <= h < 11) or (14 <= h < 16) or (h == 23):    p = PRICE_FLAT    # 平段
            else:                                                 p = PRICE_VALLEY  # 低谷
        prices.append(p)
    return prices


def scale_pv(pv_raw, pv_rated):
    s = pv_rated / PV_RATED_ORIGINAL
    return [v * s for v in pv_raw]


# ============================================================
# 3. 单场景 MILP 求解 (单一电池)
# ============================================================

def solve_scenario(pv, load, prices, pv_rated, consumption_target, bat_price_factor=1.0):
    total_pv = sum(pv)
    max_pv = max(pv)
    total_load = sum(load)

    if total_pv == 0:
        return None
    if total_load == 0:
        return None

    # 容量上界估计 (基于可消纳的剩余光伏)
    total_surplus = sum(max(pv[t] - load[t], 0) for t in range(T))
    add_ub = max(1000, min(total_surplus * 0.5, max_pv * 8760 * 0.25))
    etotal_ub = E_BAT_MAX + add_ub

    model = gp.Model("bat_sizing_merged")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 300)   # 配合60/70%用MIPGap=0.005(高PV端偶发慢解), 提到300s避免服务器上假"不可行"缺口
    # 60/70%(低消纳率, 目标函数平缓)收紧到0.005消除残余抖动; 80%+本就平滑, 保持0.02省时
    mipgap = 0.005 if consumption_target < 0.8 else 0.02
    model.setParam("MIPGap", mipgap)
    model.setParam("Threads", 1)   # 单线程确定性求解, 与并行版(_init_worker Threads=1)对齐, 保证串行==并行可复现

    idx = range(T)

    # ---- 电池总容量决策变量 ----
    E_TOTAL = model.addVar(lb=E_BAT_MAX, ub=etotal_ub, name="E_TOTAL")

    # ---- 逐小时运行变量 ----
    P_grid = model.addVars(idx, lb=0, name="P_grid")
    P_PV_load = model.addVars(idx, lb=0, name="P_PV_load")
    P_PV_curt = model.addVars(idx, lb=0, name="P_PV_curt")

    # 电池 (绝对能量 E_bat, 不除以容量 -> 全线性)
    P_bat_ch = model.addVars(idx, lb=0, name="P_bat_ch")
    P_bat_dis = model.addVars(idx, lb=0, name="P_bat_dis")
    E_bat = model.addVars(idx, lb=0, name="E_bat")

    # 充放电互斥二元变量 (必须保留! 否则LP会"同时充放"作弊——假装消纳更多PV、绕过功率/容量限制,
    # 得到物理不可能的更低成本/更小电池; 高消纳率和高PV下尤其严重, 已实证。)
    u_ch = model.addVars(idx, vtype=GRB.BINARY, name="u_ch")
    u_dis = model.addVars(idx, vtype=GRB.BINARY, name="u_dis")

    # ---- 目标函数 ----
    crf_pv = crf(DISCOUNT_RATE, PV_LIFE)
    crf_bat = crf(DISCOUNT_RATE, BAT_LIFE)

    eff_bat_capex = BAT_CAPEX * bat_price_factor
    cost_pv = (crf_pv + PV_OM_RATE) * PV_CAPEX * pv_rated
    cost_bat = (crf_bat + BAT_OM_RATE) * eff_bat_capex * E_TOTAL
    cost_grid = gp.quicksum(prices[t] * P_grid[t] for t in idx)

    model.setObjective(cost_pv + cost_bat + cost_grid, GRB.MINIMIZE)

    # ---- 约束条件 ----
    for t in idx:
        # C1: PV分配 (充电 + 直供负荷 + 弃光 = PV)
        model.addConstr(
            P_bat_ch[t] + P_PV_load[t] + P_PV_curt[t] == pv[t],
            name=f"pv_alloc_{t}"
        )

        # C2: 功率上限 (E_TOTAL / BAT_POWER_RATIO)
        model.addConstr(P_bat_ch[t] <= E_TOTAL / BAT_POWER_RATIO, name=f"ch_lim_{t}")
        model.addConstr(P_bat_dis[t] <= E_TOTAL / BAT_POWER_RATIO, name=f"dis_lim_{t}")

        # C3: 充放电互斥 (用逐小时紧M: P_ch≤pv[t], P_dis≤load[t]; 可行集与原大M相同但LP松弛更强, 消除抖动且不卡死)
        model.addConstr(P_bat_ch[t] <= pv[t] * u_ch[t], name=f"ch_{t}")
        model.addConstr(P_bat_dis[t] <= load[t] * u_dis[t], name=f"dis_{t}")
        model.addConstr(u_ch[t] + u_dis[t] <= 1, name=f"excl_{t}")

        # C4: 能量动态 (绝对能量, 线性)
        if t == 0:
            model.addConstr(
                E_bat[t] == SOC_BAT_INIT * E_TOTAL
                + ETA_CHARGE * P_bat_ch[t] - P_bat_dis[t] / ETA_DISCHARGE,
                name=f"e_{t}"
            )
        else:
            model.addConstr(
                E_bat[t] == E_bat[t - 1]
                + ETA_CHARGE * P_bat_ch[t] - P_bat_dis[t] / ETA_DISCHARGE,
                name=f"e_{t}"
            )

        # C5: 能量上下限 (SOC_MIN/MAX * E_TOTAL, 线性)
        model.addConstr(E_bat[t] >= SOC_BAT_MIN * E_TOTAL, name=f"e_lo_{t}")
        model.addConstr(E_bat[t] <= SOC_BAT_MAX * E_TOTAL, name=f"e_hi_{t}")

        # C6: 负荷平衡
        model.addConstr(
            load[t] == P_PV_load[t] + P_bat_dis[t] + P_grid[t],
            name=f"balance_{t}"
        )

    # C7: 年末能量 = 初始
    model.addConstr(E_bat[T - 1] == SOC_BAT_INIT * E_TOTAL, name="bat_final")

    # C8: 消纳率约束
    curtail_sum = gp.quicksum(P_PV_curt[t] for t in idx)
    cons_tol = CONSUMPTION_TOL   # 所有消纳目标统一用 0.05%
    max_curtail = (1 - consumption_target + cons_tol) * total_pv
    min_curtail = max(0, (1 - consumption_target - cons_tol) * total_pv)
    model.addConstr(curtail_sum <= max_curtail, name="curt_hi")
    if consumption_target < 1.0:
        model.addConstr(curtail_sum >= min_curtail, name="curt_lo")

    # ---- 求解 ----
    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None
    if model.SolCount == 0:
        return None

    # ---- 提取结果 ----
    e_total = E_TOTAL.X
    p_total = e_total / BAT_POWER_RATIO

    total_grid = sum(P_grid[t].X for t in idx)
    total_grid_cost = sum(prices[t] * P_grid[t].X for t in idx)
    total_pv_load = sum(P_PV_load[t].X for t in idx)
    total_bat_dis = sum(P_bat_dis[t].X for t in idx)
    total_curtail = sum(P_PV_curt[t].X for t in idx)

    energy_pen = (total_pv_load + total_bat_dis) / total_load * 100
    actual_cons = (total_pv - total_curtail) / total_pv * 100
    lcoe = model.ObjVal / total_load

    return {
        "bat_price_factor": bat_price_factor,
        "PV_MW": pv_rated / 1000,
        "cons_target_pct": consumption_target * 100,
        "total_MWh": round(e_total / 1000, 2),
        "storage_duration_h": round(e_total / (total_load / T), 2),
        "total_MW": round(p_total / 1000, 2),
        "grid_kWh": round(total_grid, 0),
        "grid_cost_USD": round(total_grid_cost, 0),
        "curtail_kWh": round(total_curtail, 0),
        "actual_cons_pct": round(actual_cons, 2),
        "energy_pen_pct": round(energy_pen, 2),
        "total_cost_USD": round(model.ObjVal, 0),
        "LCOE": round(lcoe, 4),
    }


# ============================================================
# 4. 结果保存
# ============================================================

def save_to_excel(results, filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_bat_sizing_merged_greenfield_results.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    headers = [
        "bat_price_factor",
        "PV_MW", "cons_target_%",
        "total_MWh", "storage_duration_h", "total_MW",
        "grid_kWh", "grid_cost_USD", "curtail_kWh",
        "actual_cons_%", "energy_pen_%",
        "total_cost_USD", "LCOE_USD_per_kWh",
    ]
    ws.append(headers)

    for r in results:
        ws.append([
            r["bat_price_factor"],
            r["PV_MW"], r["cons_target_pct"],
            r["total_MWh"], r["storage_duration_h"], r["total_MW"],
            r["grid_kWh"], r["grid_cost_USD"], r["curtail_kWh"],
            r["actual_cons_pct"], r["energy_pen_pct"],
            r["total_cost_USD"], r["LCOE"],
        ])

    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 2

    wb.save(filename)
    wb.close()
    print(f"结果已保存到 {filename}")


# ============================================================
# 5. 可视化
# ============================================================

def plot_results(results):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 常规三图仅画 baseline(价格倍数=1.0), 避免价格维度导致曲线过多
    base = [r for r in results if r["bat_price_factor"] == 1.0]

    # ---- 图1: 渗透率 vs LCOE (baseline) ----
    fig1, ax1 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = [(r["energy_pen_pct"], r["LCOE"])
               for r in base if r["cons_target_pct"] == ct_pct]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, linewidth=1.5, color=colors[i], marker="o",
                     markersize=2, label=f"{ct_pct:.0f}%")
    ax1.set_xlabel("Energy Penetration (%)")
    ax1.set_ylabel("LCOE (USD/kWh)")
    ax1.set_title("Battery Sizing (Single, baseline): LCOE vs Energy Penetration")
    ax1.legend(title="Consumption", loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "plot_bat_merged_lcoe_vs_penetration.png"), dpi=150)
    plt.close(fig1)

    # ---- 图2: PV容量 vs LCOE (baseline) ----
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = [(r["PV_MW"], r["LCOE"])
               for r in base if r["cons_target_pct"] == ct_pct]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, linewidth=1.5, color=colors[i], label=f"{ct_pct:.0f}%")
    ax2.set_xlabel("PV Capacity (MW)")
    ax2.set_ylabel("LCOE (USD/kWh)")
    ax2.set_title("Battery Sizing (Single, baseline): LCOE vs PV Capacity")
    ax2.legend(title="Consumption", loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "plot_bat_merged_lcoe_vs_pv.png"), dpi=150)
    plt.close(fig2)

    # ---- 图3: PV容量 vs 最优电池容量 (baseline) ----
    fig3, ax3 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = [(r["PV_MW"], r["total_MWh"])
               for r in base if r["cons_target_pct"] == ct_pct]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax3.plot(xs, ys, linewidth=1.5, color=colors[i], label=f"{ct_pct:.0f}%")
    ax3.set_xlabel("PV Capacity (MW)")
    ax3.set_ylabel("Total Battery Capacity (MWh)")
    ax3.set_title("Optimal Battery Capacity vs PV Capacity (baseline)")
    ax3.legend(title="Consumption", loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "plot_bat_merged_capacity.png"), dpi=150)
    plt.close(fig3)

    # ---- 图4: 价格灵敏度 (5子图=消纳率, 每子图4条价格倍数曲线) ----
    factor_colors = {1.0: "tab:blue", 0.9: "tab:orange", 0.7: "tab:green", 0.5: "tab:red"}
    fig4, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.flatten()
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ax = flat[i]
        ct_pct = ct * 100
        for factor in BAT_PRICE_FACTORS:
            pts = [(r["PV_MW"], r["total_MWh"])
                   for r in results
                   if r["cons_target_pct"] == ct_pct and r["bat_price_factor"] == factor]
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, linewidth=1.4, color=factor_colors[factor],
                        label=f"x{factor}")
        ax.set_title(f"Consumption {ct_pct:.0f}%")
        ax.set_xlabel("PV Capacity (MW)")
        ax.set_ylabel("Total Battery (MWh)")
        ax.legend(title="Price factor", fontsize=8)
        ax.grid(True, alpha=0.3)
    flat[5].axis("off")
    fig4.suptitle("Battery Capacity Sensitivity to Battery Price Factor", fontsize=13)
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "plot_bat_merged_capacity_by_pricefactor.png"), dpi=150)
    plt.close(fig4)

    print("图表已保存: plot_bat_merged_lcoe_vs_penetration.png, "
          "plot_bat_merged_lcoe_vs_pv.png, plot_bat_merged_capacity.png, "
          "plot_bat_merged_capacity_by_pricefactor.png")


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":
    print("正在读取数据...")
    pv_raw, load_data = load_raw_data()
    prices = build_prices()

    pv_values = build_pv_values()
    n_pv = len(pv_values)
    n_ct = len(CONSUMPTION_TARGETS)
    n_pf = len(BAT_PRICE_FACTORS)
    n_total = n_pv * n_ct * n_pf

    all_results = []
    n_done = 0
    n_ok = 0
    t_start = time.time()

    print(f"扫描(单电池): {n_pv} PV容量 × {n_ct} 消纳率 × {n_pf} 价格倍数 = {n_total} 场景")
    print(f"PV范围: {PV_SCAN_MIN/1000:.1f}~{PV_SCAN_MAX/1000:.1f} MW, 步长 {PV_SCAN_STEP/1000:.1f} MW")
    print(f"电池价格倍数(相对BAT_CAPEX): {BAT_PRICE_FACTORS}")
    print()

    for factor in BAT_PRICE_FACTORS:
        for pv_kw in pv_values:
            pv = scale_pv(pv_raw, pv_kw)
            for ct in CONSUMPTION_TARGETS:
                n_done += 1

                if n_done % 30 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(f"  进度 {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                          f"可行 {n_ok}  已用 {elapsed:.0f}s  剩余 ~{eta:.0f}s")

                r = solve_scenario(pv, load_data, prices, pv_kw, ct, bat_price_factor=factor)
                if r is not None:
                    all_results.append(r)
                    n_ok += 1

    elapsed = time.time() - t_start
    print(f"\n扫描完成: {n_ok}/{n_total} 可行, 耗时 {elapsed:.1f}s")

    all_results.sort(key=lambda r: (r["bat_price_factor"], r["PV_MW"], r["cons_target_pct"]))

    if all_results:
        save_to_excel(all_results)
        plot_results(all_results)
    else:
        print("无可行解, 请检查参数设置")
