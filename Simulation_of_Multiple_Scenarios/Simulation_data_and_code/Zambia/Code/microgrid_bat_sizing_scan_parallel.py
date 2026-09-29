"""
Zambia mining PV-battery sizing: multiprocessing version
Model, logic, and output match microgrid_bat_sizing_scan.py; only the main loop is parallelized.
Same structure as Norway parallel; no wind, tasks are (factor, pv_kw, ct), and the loader returns three values.

Usage:
    python microgrid_bat_sizing_scan_parallel.py
    N_WORKERS=32 python microgrid_bat_sizing_scan_parallel.py
"""
import os
# Clear environment settings for the v10 commercial license to use the gurobipy 13 academic license setup.
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
if os.name == "nt" and _N_RAW > 30:
    N_WORKERS = 30
else:
    N_WORKERS = _N_RAW
CHECKPOINT_EVERY = 50

_W = {}


def _init_worker():
    """Initialize each worker once: limit solver threads and load the data."""
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
        base.save_to_excel(
            results,
            filename=os.path.join(out_dir, "sensitivity_bat_sizing_checkpoint.xlsx"),
        )
    except Exception as e:
        print(f"  [warn] Could not write checkpoint: {e}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    # Load data once in the parent to compute avg_load_mw for duration plots; each worker loads its own data.
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

    print(f"Parallel scan (Zambia PV + battery): {n_total} scenarios, using {n_workers} processes (one thread per process)")
    print(f"PV: {base.PV_SCAN_MIN}~{base.PV_SCAN_MAX} kW, step {base.PV_SCAN_STEP} kW, "
          f"total {len(pv_values)} capacity levels")
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

        if n_done % 50 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  Progress {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                  f"feasible {n_ok}  elapsed {elapsed:.0f}s  remaining ~{eta:.0f}s")

        if r is not None and n_ok > 0 and n_ok % CHECKPOINT_EVERY == 0:
            _checkpoint(results, out_dir)

    # Refine each (factor, ct) group with WALL_FINE_STEP where energy_pen>=WALL_PEN_ENTER.
    fine_tasks = base.boundary_fine_tasks(results)
    n_fine_ok = 0
    if fine_tasks:
        print(f"\nBoundary refinement: {len(fine_tasks)} additional tasks (step {base.WALL_FINE_STEP} kW, "
              f"pen>={base.WALL_PEN_ENTER*100:.0f}%)", flush=True)
        nf_done = 0
        for r in pool.imap_unordered(_worker, fine_tasks):
            nf_done += 1
            if r is not None:
                results.append(r)
                n_fine_ok += 1
            if nf_done % 50 == 0 or nf_done == len(fine_tasks):
                print(f"  Refinement progress {nf_done}/{len(fine_tasks)}  feasible {n_fine_ok}", flush=True)
        print(f"Boundary refinement complete: feasible {n_fine_ok}/{len(fine_tasks)}", flush=True)

    results.sort(key=lambda r: (r["bat_price_factor"], r["PV_MW"], r["cons_target_pct"]))

    elapsed = time.time() - t_start
    print(f"\nScan complete: coarse feasible {n_ok}/{n_total} + refinement feasible {n_fine_ok}, elapsed {elapsed:.1f}s")

    # Close workers and release subprocess handles before saving to avoid open file handles.
    pool.close()
    pool.join()
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
