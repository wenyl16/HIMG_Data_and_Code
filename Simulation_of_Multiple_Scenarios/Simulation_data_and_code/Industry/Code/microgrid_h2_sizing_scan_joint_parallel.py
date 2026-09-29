"""
PV-battery-hydrogen model with joint storage cost scaling, parallel version.
================================================
The model, logic, and outputs match microgrid_h2_sizing_scan_joint.py; only the main loop is parallelized.
Scale BAT_CAPEX and all hydrogen CAPEX terms (ELEC/H2/FC) together by storage_factor [1.0/0.9/0.7/0.5].

Usage:
    cd path\\to\\Industry\\Code
    E:\\Anaconda\\Files\\python.exe microgrid_h2_sizing_scan_joint_parallel.py
    N_WORKERS=32 E:\\Anaconda\\Files\\python.exe microgrid_h2_sizing_scan_joint_parallel.py
"""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gurobipy as gp
import microgrid_h2_sizing_scan_joint as base

_N_RAW = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))
N_WORKERS = 30 if (os.name == "nt" and _N_RAW > 30) else _N_RAW
CHECKPOINT_EVERY = 1000

_W = {}


def _init_worker():
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    pv_raw, load = base.load_raw_data()
    _W["pv_raw"] = pv_raw
    _W["load"] = load
    _W["prices"] = base.build_prices()


def _worker(task):
    storage_factor, pv_kw, ct = task
    pv = base.scale_pv(_W["pv_raw"], pv_kw)
    r = base.solve_scenario(pv, _W["load"], _W["prices"], pv_kw, ct, storage_factor=storage_factor)
    if r is not None:
        r["storage_factor"] = storage_factor
    return r


def _checkpoint(results, out_dir):
    try:
        base.save_to_excel(
            results,
            filename=os.path.join(out_dir, "sensitivity_h2_sizing_joint_checkpoint.xlsx"),
        )
    except Exception as e:
        print(f"  [warn] Failed to write checkpoint: {e}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    pv_values = list(range(base.PV_SCAN_MIN, base.PV_SCAN_MAX + base.PV_SCAN_STEP, base.PV_SCAN_STEP))
    tasks = [(sf, pv_kw, ct)
             for sf in base.STORAGE_PRICE_FACTORS
             for pv_kw in pv_values
             for ct in base.CONSUMPTION_TARGETS]
    n_total = len(tasks)
    n_workers = min(N_WORKERS, n_total)

    print(f"Parallel scan (joint storage cost scaling): {n_total} scenarios, {n_workers} processes (1 thread each)")
    print(f"PV range: {base.PV_SCAN_MIN/1000:.1f}~{base.PV_SCAN_MAX/1000:.0f} MW, "
          f"step {base.PV_SCAN_STEP/1000:.1f} MW")
    print(f"storage_factor (battery and all hydrogen components): {base.STORAGE_PRICE_FACTORS}")
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

        if n_done % 500 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  Progress {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                  f"Feasible {n_ok}  Elapsed {elapsed:.0f}s  Remaining ~{eta:.0f}s")

        if r is not None and n_ok > 0 and n_ok % CHECKPOINT_EVERY == 0:
            _checkpoint(results, out_dir)

    results.sort(key=lambda r: (r["storage_factor"], r["PV_MW"], r["cons_target_pct"]))

    elapsed = time.time() - t_start
    print(f"\nScan complete: {n_ok}/{n_total} feasible, elapsed {elapsed:.1f}s")

    if results:
        base.save_to_excel(results)
        base.plot_results(results)
        cp = os.path.join(out_dir, "sensitivity_h2_sizing_joint_checkpoint.xlsx")
        if os.path.exists(cp):
            os.remove(cp)
            print("Checkpoint file removed")
    else:
        print("No feasible solution; check the parameter settings")

    for _p in list(getattr(pool, "_pool", [])):
        try:
            _p.terminate()
        except Exception:
            pass
    os._exit(0)
