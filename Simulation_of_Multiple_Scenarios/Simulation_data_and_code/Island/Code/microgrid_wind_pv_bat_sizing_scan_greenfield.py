"""
Wind-PV-battery microgrid: PV + wind capacity scan and battery sizing (single merged battery)
Scan total renewable capacity (Wind:PV = 1000:835), utilization targets (60%-100%), and battery price factors.
Single battery: E_TOTAL is the capacity decision variable; power = E_TOTAL / BAT_POWER_RATIO.
Track SOC using absolute energy E_bat(t) with linear constraints and tight big-M charge/discharge exclusion.
Battery price sensitivity: evaluate each BAT_PRICE_FACTORS value independently.
Note: this is a grid-connected self-consumption model with a perfectly reliable grid slack supply, no diesel generator or LPSP constraint.
Objective: minimize total annualized system cost. Solver: Gurobi.
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
# 1. System parameters
# ============================================================

# PV
PV_RATED_ORIGINAL = 835        # kW
PV_CAPEX = 530                 # $/kW
PV_LIFE = 25
PV_OM_RATE = 0.015

# Wind turbines
WIND_RATED_ORIGINAL = 1000     # kW
WIND_CAPEX = 844              # $/kW
WIND_LIFE = 30
WIND_OM_RATE = 0.03

# Wind and PV capacity scan (Wind:PV = 1000:835)
RATIO_WIND = 1000 / 1835
RATIO_PV = 835 / 1835
RENEW_SCAN_MIN = 10          # kW
RENEW_SCAN_MAX = 16000         # kW
RENEW_SCAN_STEP = 100          # kW

# Single battery storage system
E_BAT_MAX = 0                  # kWh: greenfield planning, with no existing battery capacity lower bound.
BAT_POWER_RATIO = 2.25         # Energy-to-power ratio = 2.25:1; power = E_TOTAL / 2.25.
BAT_CAPEX = 250                # $/kWh
BAT_LIFE = 8
BAT_OM_RATE = 0.01
BAT_PRICE_FACTORS = [1.0]              # Greenfield: baseline cost factor only.

ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95
SOC_BAT_MIN = 0.1
SOC_BAT_MAX = 0.9
SOC_BAT_INIT = 0.5

# Electricity price: RMB/kWh divided by 7 for USD; two periods, without seasonal variation.
PRICE_PEAK   = 0.568 / 7    # Peak period: 08:00-22:00
PRICE_VALLEY = 0.288 / 7    # Off-peak period: 22:00-08:00 the following day

# Economic parameters
DISCOUNT_RATE = 0.05

# Renewable utilization targets
CONSUMPTION_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]
CONSUMPTION_TOL = 0.0005       # 0.05% tolerance for all utilization targets

T = 8760


# ============================================================
# 2. Helper functions
# ============================================================

def crf(rate, years):
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def load_raw_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annual_8760.xlsx")
    wb = openpyxl.load_workbook(filepath, read_only=True)
    pv_raw = [row[0] for row in wb["PV"].iter_rows(min_row=1, max_row=T, max_col=1, values_only=True)]
    wind_raw = [row[0] for row in wb["Wind"].iter_rows(min_row=1, max_row=T, max_col=1, values_only=True)]
    load = [row[0] for row in wb["Load"].iter_rows(min_row=1, max_row=T, max_col=1, values_only=True)]
    wb.close()
    return pv_raw, wind_raw, load


def build_prices():
    # Two-period tariff, without seasonal variation: peak 08:00-22:00; off-peak 22:00-08:00.
    prices = []
    for i in range(T):
        h = i % 24
        if 8 <= h < 22:                p = PRICE_PEAK     # Peak (14 h)
        else:                          p = PRICE_VALLEY   # Off-peak (10 h)
        prices.append(p)
    return prices


def scale_pv(pv_raw, pv_rated):
    s = pv_rated / PV_RATED_ORIGINAL
    return [v * s for v in pv_raw]


def scale_wind(wind_raw, wind_rated):
    s = wind_rated / WIND_RATED_ORIGINAL
    return [v * s for v in wind_raw]


# ============================================================
# 3. Solve one scenario as a MILP (single battery).
# ============================================================

def solve_scenario(pv, wind, load, prices, pv_rated, wind_rated, consumption_target,
                   bat_price_factor=1.0):
    ren = [pv[t] + wind[t] for t in range(T)]
    total_ren = sum(ren)
    max_ren = max(ren)
    total_load = sum(load)

    if total_ren == 0:
        return None
    if total_load == 0:
        return None

    # Estimate the capacity upper bound.
    total_surplus = sum(max(ren[t] - load[t], 0) for t in range(T))
    add_ub = max(1000, min(total_surplus * 0.5, max_ren * 8760 * 0.25))
    etotal_ub = E_BAT_MAX + add_ub

    model = gp.Model("wind_pv_bat_sizing")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 300)
    mipgap = 0.005 if consumption_target < 0.8 else 0.02
    model.setParam("MIPGap", mipgap)
    model.setParam("Threads", 1)   # Use one solver thread, matching parallel _init_worker Threads=1 for consistent serial/parallel runs.

    idx = range(T)

    # Total battery capacity decision variable
    E_TOTAL = model.addVar(lb=E_BAT_MAX, ub=etotal_ub, name="E_TOTAL")

    # Hourly operating variables
    P_grid = model.addVars(idx, lb=0, name="P_grid")
    P_ren_load = model.addVars(idx, lb=0, name="P_ren_load")
    P_ren_curt = model.addVars(idx, lb=0, name="P_ren_curt")

    P_bat_ch = model.addVars(idx, lb=0, name="P_bat_ch")
    P_bat_dis = model.addVars(idx, lb=0, name="P_bat_dis")
    E_bat = model.addVars(idx, lb=0, name="E_bat")

    u_ch = model.addVars(idx, vtype=GRB.BINARY, name="u_ch")
    u_dis = model.addVars(idx, vtype=GRB.BINARY, name="u_dis")

    # Objective function
    crf_pv = crf(DISCOUNT_RATE, PV_LIFE)
    crf_wind = crf(DISCOUNT_RATE, WIND_LIFE)
    crf_bat = crf(DISCOUNT_RATE, BAT_LIFE)

    eff_bat_capex = BAT_CAPEX * bat_price_factor
    cost_pv = (crf_pv + PV_OM_RATE) * PV_CAPEX * pv_rated
    cost_wind = (crf_wind + WIND_OM_RATE) * WIND_CAPEX * wind_rated
    cost_bat = (crf_bat + BAT_OM_RATE) * eff_bat_capex * E_TOTAL
    cost_grid = gp.quicksum(prices[t] * P_grid[t] for t in idx)

    model.setObjective(cost_pv + cost_wind + cost_bat + cost_grid, GRB.MINIMIZE)

    # Constraints
    for t in idx:
        # C1: Renewable allocation: charging + direct load supply + curtailment = PV + wind.
        model.addConstr(
            P_bat_ch[t] + P_ren_load[t] + P_ren_curt[t] == ren[t],
            name=f"ren_alloc_{t}"
        )

        # C2: Power limit (E_TOTAL / BAT_POWER_RATIO)
        model.addConstr(P_bat_ch[t] <= E_TOTAL / BAT_POWER_RATIO, name=f"ch_lim_{t}")
        model.addConstr(P_bat_dis[t] <= E_TOTAL / BAT_POWER_RATIO, name=f"dis_lim_{t}")

        # C3: Charge/discharge exclusion using hourly tight big-M values: P_ch <= ren[t], P_dis <= load[t].
        model.addConstr(P_bat_ch[t] <= ren[t] * u_ch[t], name=f"ch_{t}")
        model.addConstr(P_bat_dis[t] <= load[t] * u_dis[t], name=f"dis_{t}")
        model.addConstr(u_ch[t] + u_dis[t] <= 1, name=f"excl_{t}")

        # C4: Energy dynamics using absolute energy (linear)
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

        # C5: Energy limits
        model.addConstr(E_bat[t] >= SOC_BAT_MIN * E_TOTAL, name=f"e_lo_{t}")
        model.addConstr(E_bat[t] <= SOC_BAT_MAX * E_TOTAL, name=f"e_hi_{t}")

        # C6: Load balance
        model.addConstr(
            load[t] == P_ren_load[t] + P_bat_dis[t] + P_grid[t],
            name=f"balance_{t}"
        )

    # C7: End-of-year energy equals initial energy.
    model.addConstr(E_bat[T - 1] == SOC_BAT_INIT * E_TOTAL, name="bat_final")

    # C8: Renewable utilization constraints
    curtail_sum = gp.quicksum(P_ren_curt[t] for t in idx)
    cons_tol = CONSUMPTION_TOL   # Use a 0.05% tolerance for every utilization target.
    max_curtail = (1 - consumption_target + cons_tol) * total_ren
    min_curtail = max(0, (1 - consumption_target - cons_tol) * total_ren)
    model.addConstr(curtail_sum <= max_curtail, name="curt_hi")
    if consumption_target < 1.0:
        model.addConstr(curtail_sum >= min_curtail, name="curt_lo")

    # Solve the model.
    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None
    if model.SolCount == 0:
        return None

    # Extract results.
    e_total = E_TOTAL.X
    p_total = e_total / BAT_POWER_RATIO

    total_grid = sum(P_grid[t].X for t in idx)
    total_grid_cost = sum(prices[t] * P_grid[t].X for t in idx)
    total_ren_load = sum(P_ren_load[t].X for t in idx)
    total_bat_dis = sum(P_bat_dis[t].X for t in idx)
    total_curtail = sum(P_ren_curt[t].X for t in idx)

    energy_pen = (total_ren_load + total_bat_dis) / total_load * 100
    actual_cons = (total_ren - total_curtail) / total_ren * 100
    lcoe = model.ObjVal / total_load

    return {
        "bat_price_factor": bat_price_factor,
        "Total_Renewable_MW": (pv_rated + wind_rated) / 1000,
        "PV_MW": pv_rated / 1000,
        "Wind_MW": wind_rated / 1000,
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
# 4. Save results.
# ============================================================

def save_to_excel(results, filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sensitivity_wind_pv_bat_sizing_greenfield_results.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    headers = [
        "bat_price_factor",
        "Total_Renewable_MW", "PV_MW", "Wind_MW", "cons_target_%",
        "total_MWh", "storage_duration_h", "total_MW",
        "grid_kWh", "grid_cost_USD", "curtail_kWh",
        "actual_cons_%", "energy_pen_%",
        "total_cost_USD", "LCOE_USD_per_kWh",
    ]
    ws.append(headers)

    for r in results:
        ws.append([
            r["bat_price_factor"],
            r["Total_Renewable_MW"], r["PV_MW"], r["Wind_MW"], r["cons_target_pct"],
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
    print(f"Results saved to {filename}")


# ============================================================
# 5. Visualization
# ============================================================

def plot_results(results):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Figure 1: Penetration vs LCOE (baseline)
    base = [r for r in results if r["bat_price_factor"] == 1.0]
    fig1, ax1 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = [(r["energy_pen_pct"], r["LCOE"]) for r in base if r["cons_target_pct"] == ct_pct]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, linewidth=1.5, color=colors[i], marker="o",
                     markersize=2, label=f"{ct_pct:.0f}%")
    ax1.set_xlabel("Energy Penetration (%)")
    ax1.set_ylabel("LCOE (USD/kWh)")
    ax1.set_title("Wind-PV-Battery (baseline): LCOE vs Energy Penetration")
    ax1.legend(title="Consumption", loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "plot_wind_pv_bat_lcoe_vs_penetration.png"), dpi=150)
    plt.close(fig1)

    # Figure 2: Total renewable capacity vs LCOE (baseline)
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = [(r["Total_Renewable_MW"], r["LCOE"]) for r in base if r["cons_target_pct"] == ct_pct]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, linewidth=1.5, color=colors[i], label=f"{ct_pct:.0f}%")
    ax2.set_xlabel("Total Renewable Capacity (MW)")
    ax2.set_ylabel("LCOE (USD/kWh)")
    ax2.set_title("Wind-PV-Battery (baseline): LCOE vs Renewable Capacity")
    ax2.legend(title="Consumption", loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "plot_wind_pv_bat_lcoe_vs_capacity.png"), dpi=150)
    plt.close(fig2)

    # Figure 3: Price sensitivity; five utilization-target panels, each with four price-factor curves.
    factor_colors = {1.0: "tab:blue", 0.9: "tab:orange", 0.7: "tab:green", 0.5: "tab:red"}
    fig4, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.flatten()
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ax = flat[i]
        ct_pct = ct * 100
        for factor in BAT_PRICE_FACTORS:
            pts = [(r["Total_Renewable_MW"], r["total_MWh"])
                   for r in results
                   if r["cons_target_pct"] == ct_pct and r["bat_price_factor"] == factor]
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, linewidth=1.4, color=factor_colors[factor], label=f"x{factor}")
        ax.set_title(f"Consumption {ct_pct:.0f}%")
        ax.set_xlabel("Total Renewable (MW)")
        ax.set_ylabel("Battery (MWh)")
        ax.legend(title="Price factor", fontsize=8)
        ax.grid(True, alpha=0.3)
    flat[5].axis("off")
    fig4.suptitle("Battery Capacity Sensitivity to Battery Price Factor (Wind-PV)", fontsize=13)
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "plot_wind_pv_bat_capacity_by_pricefactor.png"), dpi=150)
    plt.close(fig4)

    print("Figures saved: plot_wind_pv_bat_lcoe_vs_penetration.png, "
          "plot_wind_pv_bat_lcoe_vs_capacity.png, "
          "plot_wind_pv_bat_capacity_by_pricefactor.png")


# ============================================================
# 6. Main program
# ============================================================

if __name__ == "__main__":
    print("Reading data...")
    pv_raw, wind_raw, load_data = load_raw_data()
    prices = build_prices()

    renew_values = list(range(RENEW_SCAN_MIN, RENEW_SCAN_MAX + RENEW_SCAN_STEP, RENEW_SCAN_STEP))
    n_ren = len(renew_values)
    n_ct = len(CONSUMPTION_TARGETS)
    n_pf = len(BAT_PRICE_FACTORS)
    n_total = n_ren * n_ct * n_pf

    all_results = []
    n_done = 0
    n_ok = 0
    t_start = time.time()

    print(f"Scan (single battery): {n_ren} renewable capacities x {n_ct} utilization targets x {n_pf} price factors = {n_total} scenarios")
    print(f"Total renewable capacity: {RENEW_SCAN_MIN/1000:.3f}~{RENEW_SCAN_MAX/1000:.1f} MW, "
          f"step {RENEW_SCAN_STEP/1000:.1f} MW, Wind:PV = 1000:835")
    print(f"Battery price factors: {BAT_PRICE_FACTORS}")
    print()

    for factor in BAT_PRICE_FACTORS:
        for total_kw in renew_values:
            wind_kw = total_kw * RATIO_WIND
            pv_kw = total_kw * RATIO_PV
            pv = scale_pv(pv_raw, pv_kw)
            wind = scale_wind(wind_raw, wind_kw)
            for ct in CONSUMPTION_TARGETS:
                n_done += 1

                if n_done % 200 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(f"  Progress {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                          f"feasible {n_ok}  elapsed {elapsed:.0f}s  remaining ~{eta:.0f}s")

                r = solve_scenario(pv, wind, load_data, prices, pv_kw, wind_kw, ct,
                                   bat_price_factor=factor)
                if r is not None:
                    all_results.append(r)
                    n_ok += 1

    elapsed = time.time() - t_start
    print(f"\nScan complete: {n_ok}/{n_total} feasible, elapsed {elapsed:.1f}s")

    all_results.sort(key=lambda r: (r["bat_price_factor"], r["Total_Renewable_MW"], r["cons_target_pct"]))

    if all_results:
        save_to_excel(all_results)
        plot_results(all_results)
    else:
        print("No feasible solution; check the parameters")
