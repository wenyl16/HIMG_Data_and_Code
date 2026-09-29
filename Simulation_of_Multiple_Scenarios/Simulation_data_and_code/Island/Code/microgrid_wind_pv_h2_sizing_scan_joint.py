"""
Wind-PV-battery-hydrogen microgrid: joint storage cost sensitivity (Island)
============================================
Differences from microgrid_wind_pv_h2_sizing_scan.py:
  - Original: fixed battery cost; independent price sensitivities for Elec/H2/FC equipment.
  - Joint: multiply battery CAPEX and all hydrogen CAPEX (ELEC/H2/FC) by storage_factor.
             Wind (WIND_CAPEX) and PV costs remain unchanged.
Scan total renewable capacity (Wind:PV = 1000:835), utilization targets (60%-100%), and storage_factor [1.0/0.9/0.7/0.5].
Solver: Gurobi.
"""

import os
import openpyxl
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. System parameters, matching the original Island wind_pv_h2 model.
# ============================================================

PV_RATED_ORIGINAL = 835
PV_CAPEX = 530
PV_LIFE = 25
PV_OM_RATE = 0.015

WIND_RATED_ORIGINAL = 1000
WIND_CAPEX = 844
WIND_LIFE = 30
WIND_OM_RATE = 0.03

RATIO_WIND = 1000 / 1835
RATIO_PV = 835 / 1835
RENEW_SCAN_MIN = 1835
RENEW_SCAN_MAX = 30000
RENEW_SCAN_STEP = 500

P_BAT_MAX = 2000
E_BAT_MAX = 4500
ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95
SOC_BAT_MIN = 0.1
SOC_BAT_MAX = 0.9
SOC_BAT_INIT = 0.5
BAT_CAPEX = 250
BAT_LIFE = 8
BAT_OM_RATE = 0.01

ETA_ELEC = 0.0205
ELEC_CAPEX = 1000
ELEC_LIFE = 10
ELEC_OM_RATE = 0.01

SOC_H2_MIN = 0.05
SOC_H2_MAX = 0.98
SOC_H2_INIT = 0.5
H2_CAPEX = 400
H2_LIFE = 20
H2_OM_RATE = 0.01

ETA_FC = 18.3315
FC_CAPEX = 1300
FC_LIFE = 10
FC_OM_RATE = 0.01

# Joint cost scaling for battery and all hydrogen CAPEX; wind cost remains unchanged.
STORAGE_PRICE_FACTORS = [1.0, 0.9, 0.7, 0.5]

PRICE_PEAK = 0.568 / 7
PRICE_VALLEY = 0.288 / 7

DISCOUNT_RATE = 0.05

CONSUMPTION_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]
CONSUMPTION_TOL = 0.0005

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
    prices = []
    for i in range(T):
        h = i % 24
        p = PRICE_PEAK if 8 <= h < 22 else PRICE_VALLEY
        prices.append(p)
    return prices


def scale_pv(pv_raw, pv_rated):
    s = pv_rated / PV_RATED_ORIGINAL
    return [v * s for v in pv_raw]


def scale_wind(wind_raw, wind_rated):
    s = wind_rated / WIND_RATED_ORIGINAL
    return [v * s for v in wind_raw]


# ============================================================
# 3. Solve one wind-PV-H2 scenario; apply storage_factor to costs, leaving wind unchanged.
# ============================================================

def solve_scenario(pv, wind, load, prices, pv_rated, wind_rated, consumption_target, storage_factor=1.0):
    """
    storage_factor scales BAT/ELEC/H2/FC CAPEX together; wind and PV costs are unchanged.
    """
    ren = [pv[t] + wind[t] for t in range(T)]
    total_ren = sum(ren)
    max_load = max(load)
    total_load = sum(load)

    if total_ren == 0:
        return None
    if total_load == 0:
        return None

    surplus_list = [max(ren[t] - load[t], 0) for t in range(T)]
    max_surplus = max(surplus_list)
    total_surplus = sum(surplus_list)

    elec_cap_ub = max(max_surplus, 1.0)
    fc_cap_ub = max_load
    h2_tank_ub = max(1000, total_surplus * ETA_ELEC * 0.5)

    model = gp.Model("wind_pv_h2_sizing_joint")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 600)
    mipgap = 0.005 if consumption_target < 1.0 else 0.02   # Tighter bounds for the Island scenario
    model.setParam("MIPGap", mipgap)
    model.setParam("Threads", 1)

    idx = range(T)

    P_ELEC_CAP = model.addVar(lb=0, ub=elec_cap_ub, name="P_ELEC_CAP")
    P_FC_CAP = model.addVar(lb=0, ub=fc_cap_ub, name="P_FC_CAP")
    H2_TANK = model.addVar(lb=0, ub=h2_tank_ub, name="H2_TANK")

    h2_on = model.addVar(vtype=GRB.BINARY, name="h2_on")
    model.addConstr(P_ELEC_CAP <= elec_cap_ub * h2_on, name="elec_link")
    model.addConstr(P_FC_CAP >= max_load * h2_on, name="fc_min_link")
    model.addConstr(P_FC_CAP <= fc_cap_ub * h2_on, name="fc_max_link")
    model.addConstr(H2_TANK <= h2_tank_ub * h2_on, name="tank_link")

    P_grid = model.addVars(idx, lb=0, name="P_grid")
    P_ren_load = model.addVars(idx, lb=0, name="P_ren_load")
    P_ren_batt = model.addVars(idx, lb=0, name="P_ren_batt")
    P_ren_H2 = model.addVars(idx, lb=0, name="P_ren_H2")
    P_ren_curt = model.addVars(idx, lb=0, name="P_ren_curt")

    P_bat_ch = model.addVars(idx, lb=0, ub=P_BAT_MAX, name="P_bat_ch")
    P_bat_dis = model.addVars(idx, lb=0, ub=P_BAT_MAX, name="P_bat_dis")
    SOC_bat = model.addVars(idx, lb=SOC_BAT_MIN, ub=SOC_BAT_MAX, name="SOC_bat")

    P_elec = model.addVars(idx, lb=0, name="P_elec")
    P_FC = model.addVars(idx, lb=0, name="P_FC")
    m_h2 = model.addVars(idx, lb=0, name="m_h2")

    u_ch = model.addVars(idx, vtype=GRB.BINARY, name="u_ch")
    u_dis = model.addVars(idx, vtype=GRB.BINARY, name="u_dis")

    crf_pv = crf(DISCOUNT_RATE, PV_LIFE)
    crf_wind = crf(DISCOUNT_RATE, WIND_LIFE)
    crf_bat = crf(DISCOUNT_RATE, BAT_LIFE)
    crf_elec = crf(DISCOUNT_RATE, ELEC_LIFE)
    crf_h2 = crf(DISCOUNT_RATE, H2_LIFE)
    crf_fc = crf(DISCOUNT_RATE, FC_LIFE)

    # Scale battery and hydrogen CAPEX by storage_factor; retain wind and PV costs.
    cost_pv = (crf_pv + PV_OM_RATE) * PV_CAPEX * pv_rated
    cost_wind = (crf_wind + WIND_OM_RATE) * WIND_CAPEX * wind_rated
    cost_bat = (crf_bat + BAT_OM_RATE) * (BAT_CAPEX * storage_factor) * E_BAT_MAX
    cost_elec = (crf_elec + ELEC_OM_RATE) * (ELEC_CAPEX * storage_factor) * P_ELEC_CAP
    cost_h2 = (crf_h2 + H2_OM_RATE) * (H2_CAPEX * storage_factor) * H2_TANK
    cost_fc = (crf_fc + FC_OM_RATE) * (FC_CAPEX * storage_factor) * P_FC_CAP
    cost_grid = gp.quicksum(prices[t] * P_grid[t] for t in idx)

    model.setObjective(cost_pv + cost_wind + cost_bat + cost_elec + cost_h2 + cost_fc + cost_grid, GRB.MINIMIZE)

    for t in idx:
        model.addConstr(P_ren_load[t] + P_ren_batt[t] + P_ren_H2[t] + P_ren_curt[t] == ren[t], name=f"ren_alloc_{t}")
        model.addConstr(P_elec[t] == P_ren_H2[t], name=f"elec_ren_{t}")
        model.addConstr(P_bat_ch[t] == P_ren_batt[t], name=f"bat_ren_{t}")
        model.addConstr(P_elec[t] <= P_ELEC_CAP, name=f"elec_cap_{t}")
        model.addConstr(P_FC[t] <= P_FC_CAP, name=f"fc_cap_{t}")
        model.addConstr(P_bat_ch[t] <= ren[t] * u_ch[t], name=f"ch_flag_{t}")
        model.addConstr(P_bat_dis[t] <= load[t] * u_dis[t], name=f"dis_flag_{t}")
        model.addConstr(u_ch[t] + u_dis[t] <= 1, name=f"excl_{t}")
        if t == 0:
            model.addConstr(SOC_bat[t] == SOC_BAT_INIT + (ETA_CHARGE * P_bat_ch[t] - P_bat_dis[t] / ETA_DISCHARGE) / E_BAT_MAX, name=f"soc_bat_{t}")
        else:
            model.addConstr(SOC_bat[t] == SOC_bat[t - 1] + (ETA_CHARGE * P_bat_ch[t] - P_bat_dis[t] / ETA_DISCHARGE) / E_BAT_MAX, name=f"soc_bat_{t}")
        if t == 0:
            model.addConstr(m_h2[t] == SOC_H2_INIT * H2_TANK + ETA_ELEC * P_elec[t] - P_FC[t] / ETA_FC, name=f"h2_{t}")
        else:
            model.addConstr(m_h2[t] == m_h2[t - 1] + ETA_ELEC * P_elec[t] - P_FC[t] / ETA_FC, name=f"h2_{t}")
        model.addConstr(m_h2[t] >= SOC_H2_MIN * H2_TANK, name=f"h2_lo_{t}")
        model.addConstr(m_h2[t] <= SOC_H2_MAX * H2_TANK, name=f"h2_hi_{t}")
        model.addConstr(load[t] == P_ren_load[t] + P_bat_dis[t] + P_FC[t] + P_grid[t], name=f"balance_{t}")

    model.addConstr(SOC_bat[T - 1] == SOC_BAT_INIT, name="bat_final")
    model.addConstr(m_h2[T - 1] == SOC_H2_INIT * H2_TANK, name="h2_final")

    curtail_sum = gp.quicksum(P_ren_curt[t] for t in idx)
    cons_tol = CONSUMPTION_TOL
    max_curtail = (1 - consumption_target + cons_tol) * total_ren
    min_curtail = max(0, (1 - consumption_target - cons_tol) * total_ren)
    model.addConstr(curtail_sum <= max_curtail, name="curt_hi")
    if consumption_target < 1.0:
        model.addConstr(curtail_sum >= min_curtail, name="curt_lo")

    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None
    if model.SolCount == 0:
        return None

    elec_cap = P_ELEC_CAP.X
    fc_cap = P_FC_CAP.X
    h2_cap = H2_TANK.X

    total_grid = sum(P_grid[t].X for t in idx)
    total_grid_cost = sum(prices[t] * P_grid[t].X for t in idx)
    total_ren_load = sum(P_ren_load[t].X for t in idx)
    total_bat_dis = sum(P_bat_dis[t].X for t in idx)
    total_fc_gen = sum(P_FC[t].X for t in idx)
    total_curtail = sum(P_ren_curt[t].X for t in idx)

    energy_pen = (total_ren_load + total_bat_dis + total_fc_gen) / total_load * 100
    actual_cons = (total_ren - total_curtail) / total_ren * 100
    lcoe = model.ObjVal / total_load

    # Annualized costs: baseline wind/PV costs, scaled battery/H2 costs.
    ann_pv = (crf_pv + PV_OM_RATE) * PV_CAPEX * pv_rated
    ann_wind = (crf_wind + WIND_OM_RATE) * WIND_CAPEX * wind_rated
    ann_bat = (crf_bat + BAT_OM_RATE) * (BAT_CAPEX * storage_factor) * E_BAT_MAX
    ann_elec = (crf_elec + ELEC_OM_RATE) * (ELEC_CAPEX * storage_factor) * elec_cap
    ann_h2 = (crf_h2 + H2_OM_RATE) * (H2_CAPEX * storage_factor) * h2_cap
    ann_fc = (crf_fc + FC_OM_RATE) * (FC_CAPEX * storage_factor) * fc_cap

    return {
        "storage_factor": storage_factor,
        "Total_Renewable_MW": (pv_rated + wind_rated) / 1000,
        "PV_MW": pv_rated / 1000,
        "Wind_MW": wind_rated / 1000,
        "cons_target_pct": consumption_target * 100,
        "elec_cap_kW": round(elec_cap, 1),
        "fc_cap_kW": round(fc_cap, 1),
        "h2_tank_kg": round(h2_cap, 1),
        "h2_tank_MWh": round(round(h2_cap, 1) * 33.33 / 1000, 4),
        "grid_kWh": round(total_grid, 0),
        "grid_cost_USD": round(total_grid_cost, 0),
        "curtail_kWh": round(total_curtail, 0),
        "actual_cons_pct": round(actual_cons, 2),
        "energy_pen_pct": round(energy_pen, 2),
        "ann_pv": round(ann_pv, 0),
        "ann_wind": round(ann_wind, 0),
        "ann_bat": round(ann_bat, 0),
        "ann_elec": round(ann_elec, 0),
        "ann_h2": round(ann_h2, 0),
        "ann_fc": round(ann_fc, 0),
        "total_cost_USD": round(model.ObjVal, 0),
        "LCOE": round(lcoe, 4),
    }


# ============================================================
# 4. Save results.
# ============================================================

def save_to_excel(results, filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sensitivity_wind_pv_h2_sizing_joint_results.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    headers = [
        "storage_factor",
        "Total_Renewable_MW", "PV_MW", "Wind_MW", "cons_target_%",
        "elec_cap_kW", "fc_cap_kW", "h2_tank_kg", "h2_tank_MWh",
        "grid_kWh", "grid_cost_USD", "curtail_kWh",
        "actual_cons_%", "energy_pen_%",
        "ann_pv", "ann_wind", "ann_bat", "ann_elec", "ann_h2", "ann_fc",
        "total_cost_USD", "LCOE_USD_per_kWh",
    ]
    ws.append(headers)

    for r in results:
        # Exclude configurations without hydrogen storage (h2_on=0; all three capacities are zero).
        if r["elec_cap_kW"] == 0 and r["fc_cap_kW"] == 0 and r["h2_tank_kg"] == 0:
            continue
        ws.append([
            r["storage_factor"],
            r["Total_Renewable_MW"], r["PV_MW"], r["Wind_MW"], r["cons_target_pct"],
            r["elec_cap_kW"], r["fc_cap_kW"], r["h2_tank_kg"], r["h2_tank_MWh"],
            r["grid_kWh"], r["grid_cost_USD"], r["curtail_kWh"],
            r["actual_cons_pct"], r["energy_pen_pct"],
            r["ann_pv"], r["ann_wind"], r["ann_bat"], r["ann_elec"], r["ann_h2"], r["ann_fc"],
            r["total_cost_USD"], r["LCOE"],
        ])

    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 2

    wb.save(filename)
    wb.close()
    print(f"Results saved to {filename}")


# ============================================================
# 5. Plot by storage_factor, using Total_Renewable_MW as the capacity axis.
# ============================================================

def plot_results(results):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    factor_colors = {1.0: "tab:blue", 0.9: "tab:orange", 0.7: "tab:green", 0.5: "tab:red"}

    base = [r for r in results if r["storage_factor"] == 1.0]

    fig1, ax1 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = sorted([(r["energy_pen_pct"], r["LCOE"]) for r in base if r["cons_target_pct"] == ct_pct])
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, linewidth=1.5, color=colors[i], marker="o", markersize=2, label=f"{ct_pct:.0f}%")
    ax1.set_xlabel("Energy Penetration (%)")
    ax1.set_ylabel("LCOE (USD/kWh)")
    ax1.set_title("Wind-PV-H2 Joint (baseline): LCOE vs Energy Penetration")
    ax1.legend(title="Consumption", loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "plot_joint_lcoe_vs_penetration.png"), dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 7))
    for i, ct in enumerate(CONSUMPTION_TARGETS):
        ct_pct = ct * 100
        pts = sorted([(r["Total_Renewable_MW"], r["LCOE"]) for r in base if r["cons_target_pct"] == ct_pct])
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, linewidth=1.5, color=colors[i], label=f"{ct_pct:.0f}%")
    ax2.set_xlabel("Total Renewable Capacity (MW)")
    ax2.set_ylabel("LCOE (USD/kWh)")
    ax2.set_title("Wind-PV-H2 Joint (baseline): LCOE vs Renewable Capacity")
    ax2.legend(title="Consumption", loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "plot_joint_lcoe_vs_capacity.png"), dpi=150)
    plt.close(fig2)

    for cap_name, cap_key, ylab in [("Electrolyzer", "elec_cap_kW", "kW"),
                                    ("H2 Tank", "h2_tank_kg", "kg"),
                                    ("Fuel Cell", "fc_cap_kW", "kW")]:
        fig, sub = plt.subplots(2, 3, figsize=(16, 9))
        flat = sub.flatten()
        for i, ct in enumerate(CONSUMPTION_TARGETS):
            ax = flat[i]
            ct_pct = ct * 100
            for factor in STORAGE_PRICE_FACTORS:
                pts = sorted([(r["Total_Renewable_MW"], r[cap_key]) for r in results
                              if r["cons_target_pct"] == ct_pct and r["storage_factor"] == factor])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linewidth=1.3, color=factor_colors[factor], label=f"x{factor}")
            ax.set_title(f"Consumption {ct_pct:.0f}%")
            ax.set_xlabel("Total Renewable Capacity (MW)")
            ax.set_ylabel(ylab)
            ax.legend(title="storage_factor", fontsize=8)
            ax.grid(True, alpha=0.3)
        flat[5].axis("off")
        fig.suptitle(f"Joint Storage Cost Sensitivity — Optimal {cap_name} ({ylab})", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"plot_joint_{cap_key.replace('_kW','').replace('_kg','')}_sensitivity.png"), dpi=150)
        plt.close(fig)

    fig_lcoe, ax_l = plt.subplots(figsize=(10, 6))
    factors_sorted = sorted(STORAGE_PRICE_FACTORS)
    means, lo, hi = [], [], []
    for f in factors_sorted:
        vals = [r["LCOE"] for r in results if r["storage_factor"] == f]
        if vals:
            means.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
        else:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    ax_l.plot(factors_sorted, means, "o-", color="tab:blue", label="mean")
    ax_l.fill_between(factors_sorted, lo, hi, color="tab:blue", alpha=0.15, label="min-max")
    ax_l.set_title("LCOE Sensitivity to Joint Storage Cost Factor (over all renew & consumption)")
    ax_l.set_xlabel("storage_factor (battery + whole H2 system; wind unchanged)")
    ax_l.set_ylabel("LCOE (USD/kWh)")
    ax_l.legend(fontsize=9)
    ax_l.grid(True, alpha=0.3)
    fig_lcoe.tight_layout()
    fig_lcoe.savefig(os.path.join(out_dir, "plot_joint_lcoe_sensitivity.png"), dpi=150)
    plt.close(fig_lcoe)

    print("Figures saved: plot_joint_lcoe_vs_penetration.png, plot_joint_lcoe_vs_capacity.png, "
          "plot_joint_elec_cap_sensitivity.png, plot_joint_h2_tank_sensitivity.png, "
          "plot_joint_fc_cap_sensitivity.png, plot_joint_lcoe_sensitivity.png")


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
    n_sf = len(STORAGE_PRICE_FACTORS)
    n_total = n_ren * n_ct * n_sf

    all_results = []
    n_done = 0
    n_ok = 0
    t_start = time.time()

    print(f"Scan (joint storage cost scaling; wind unchanged): {n_ren} wind-PV capacities x {n_ct} utilization targets x {n_sf} storage_factor = {n_total} scenarios")
    print(f"Total renewable capacity: {RENEW_SCAN_MIN/1000:.3f}~{RENEW_SCAN_MAX/1000:.0f} MW, step {RENEW_SCAN_STEP/1000:.1f} MW")
    print(f"storage_factor (battery and all hydrogen equipment; wind unchanged): {STORAGE_PRICE_FACTORS}")
    print()

    for storage_factor in STORAGE_PRICE_FACTORS:
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

                r = solve_scenario(pv, wind, load_data, prices, pv_kw, wind_kw, ct, storage_factor=storage_factor)
                if r is not None:
                    all_results.append(r)
                    n_ok += 1

    elapsed = time.time() - t_start
    print(f"\nScan complete: {n_ok}/{n_total} feasible, elapsed {elapsed:.1f}s")

    if all_results:
        all_results.sort(key=lambda r: (r["storage_factor"], r["PV_MW"], r["cons_target_pct"]))
        save_to_excel(all_results)
        plot_results(all_results)
    else:
        print("No feasible solution; check the parameters")
