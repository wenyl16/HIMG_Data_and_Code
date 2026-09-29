"""
USA (Mesa Del Sol) PV-battery sizing: multiprocessing version
Model, logic, and output match microgrid_bat_sizing_scan.py; only the main loop is parallelized.
Usage: python microgrid_bat_sizing_scan_parallel.py (N_WORKERS=32 ...)
"""
import os
for _k in list(os.environ):
    if "GRB" in _k or "GUROBI" in _k:
        os.environ.pop(_k, None)
import signal
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gurobipy as gp
import microgrid_bat_sizing_scan as base

_N_RAW = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))
N_WORKERS = 30 if (os.name == "nt" and _N_RAW > 30) else _N_RAW
CHECKPOINT_EVERY = 50

_W = {}


def _init_worker():
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    pv_raw, load, prices = base.load_raw_data()
    _W["pv_raw"] = pv_raw
    _W["load"] = load
    _W["prices"] = prices


def _worker(task):
    factor, pv_kw, ct = task
    pv = base.scale_pv(_W["pv_raw"], pv_kw)
    return base.solve_scenario(pv, _W["load"], _W["prices"], pv_kw, ct, bat_price_factor=factor)


def _checkpoint(results, out_dir):
    try:
        base.save_to_excel(results,
                           filename=os.path.join(out_dir, "sensitivity_bat_sizing_checkpoint.xlsx"))
    except Exception as e:
        print(f"  [warn] Could not write checkpoint: {e}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    _, load_main, _ = base.load_raw_data()
    avg_load_mw = sum(load_main) / len(load_main) / 1000

    pv_values = list(range(base.PV_SCAN_MIN, base.PV_SCAN_MAX + base.PV_SCAN_STEP,
                           base.PV_SCAN_STEP))
    tasks = [(factor, pv_kw, ct)
             for factor in base.BAT_PRICE_FACTORS
             for pv_kw in pv_values
             for ct in base.CONSUMPTION_TARGETS]
    n_total = len(tasks)
    n_workers = min(N_WORKERS, n_total)

    print(f"Parallel scan (USA PV + battery): {n_total} scenarios, using {n_workers} processes (one thread per process)")
    print(f"PV: {base.PV_SCAN_MIN}~{base.PV_SCAN_MAX} kW, step {base.PV_SCAN_STEP} kW, {len(pv_values)} capacity levels")
    print()

    results = []
    n_done = 0
    n_ok = 0
    pool = Pool(n_workers, initializer=_init_worker)
    for r in pool.imap_unordered(_worker, tasks):
        n_done += 1
        if r is not None:
            results.append(r)
            n_ok += 1
        if n_done % 20 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  Progress {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                  f"feasible {n_ok}  elapsed {elapsed:.0f}s  remaining ~{eta:.0f}s")
        if r is not None and n_ok > 0 and n_ok % CHECKPOINT_EVERY == 0:
            _checkpoint(results, out_dir)

    results.sort(key=lambda r: (r["bat_price_factor"], r["PV_MW"], r["cons_target_pct"]))
    elapsed = time.time() - t_start
    print(f"\nScan complete: {n_ok}/{n_total} feasible, elapsed {elapsed:.1f}s")

    pool.close(); pool.join()
    for _p in list(getattr(pool, "_pool", [])):
        try:
            os.kill(_p.pid, signal.SIGKILL)
        except Exception:
            pass

    if results:
        try:
            base.save_to_excel(results)
            base.plot_results(results, avg_load_mw=avg_load_mw)
        except PermissionError as e:
            alt = os.path.join(out_dir, "sensitivity_bat_sizing_results_parallel.xlsx")
            print(f"[warn] Main result file is in use ({e}); writing to {alt}")
            base.save_to_excel(results, filename=alt)
            base.plot_results(results, avg_load_mw=avg_load_mw)
        cp = os.path.join(out_dir, "sensitivity_bat_sizing_checkpoint.xlsx")
        if os.path.exists(cp):
            os.remove(cp)
            print("Checkpoint file removed")
    else:
        print("No feasible solution; check the parameters")

    os._exit(0)
