"""
Battery sizing with multiprocessing for multicore servers.
The model, logic, and outputs match microgrid_bat_sizing_scan_merged.py; the main loop uses multiple processes.
  - Each scenario has an independent Gurobi model, allowing parallel execution.
  - Each worker uses one thread (gp.setParam Threads=1); use roughly one worker per physical core.
  - Load pv_raw/load/prices once when each worker starts to avoid repeated I/O and large-array transfers.
  - Write periodic checkpoints to preserve results if a long run fails.
  - Finish by saving/plotting, then terminating workers and calling os._exit to avoid Pool cleanup hangs in some Gurobi environments.

Usage:
    python microgrid_bat_sizing_scan_merged_parallel.py
    # Set worker count with N_WORKERS (default: os.cpu_count()); N_WORKERS=32 is recommended for Xeon 6338N (32 physical cores).
    N_WORKERS=32 python microgrid_bat_sizing_scan_merged_parallel.py
"""
import os
import signal
import sys
import time
from multiprocessing import Pool

# Allow imports from this directory regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gurobipy as gp
import microgrid_bat_sizing_scan_merged_greenfield as base

_N_RAW = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))
if os.name == "nt" and _N_RAW > 30:
    # On Windows, Pool uses WaitForMultipleObjects, which supports at most 63 handles.
    # Each worker uses about two handles, so limit workers to about 30 to avoid "need at most 63 handles".
    # Linux wait() does not have this limit.
    N_WORKERS = 30
else:
    N_WORKERS = _N_RAW
CHECKPOINT_EVERY = 1000   # Write a checkpoint after this many additional feasible results.

# Worker-global data, populated by each subprocess in _init_worker.
_W = {}


def _init_worker():
    """Initialize each worker once: use one thread and load the data."""
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)        # One thread per process avoids oversubscribing CPU cores.
    pv_raw, load = base.load_raw_data()
    _W["pv_raw"] = pv_raw
    _W["load"] = load
    _W["prices"] = base.build_prices()


def _worker(task):
    factor, pv_kw, ct = task
    pv = base.scale_pv(_W["pv_raw"], pv_kw)
    return base.solve_scenario(pv, _W["load"], _W["prices"], pv_kw, ct, bat_price_factor=factor)


def _checkpoint(results, out_dir):
    try:
        base.save_to_excel(
            results,
            filename=os.path.join(out_dir, "sensitivity_bat_sizing_merged_greenfield_checkpoint.xlsx"),
        )
    except Exception as e:
        print(f"  [warn] Failed to write checkpoint: {e}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    pv_values = list(range(base.PV_SCAN_MIN, base.PV_SCAN_MAX + base.PV_SCAN_STEP, base.PV_SCAN_STEP))
    tasks = [(factor, pv_kw, ct)
             for factor in base.BAT_PRICE_FACTORS
             for pv_kw in pv_values
             for ct in base.CONSUMPTION_TARGETS]
    n_total = len(tasks)
    n_workers = min(N_WORKERS, n_total)

    print(f"Parallel scan (single combined battery): {n_total} scenarios, {n_workers} processes (1 thread each)")
    print(f"PV range: {base.PV_SCAN_MIN/1000:.1f}~{base.PV_SCAN_MAX/1000:.1f} MW, step {base.PV_SCAN_STEP/1000:.1f} MW")
    print(f"Battery price factors (relative to BAT_CAPEX): {base.BAT_PRICE_FACTORS}")
    print()

    results = []
    n_done = 0
    n_ok = 0
    pool = Pool(n_workers, initializer=_init_worker)   # Avoid a context manager: terminate+join can hang during cleanup.
    for r in pool.imap_unordered(_worker, tasks):
        n_done += 1
        if r is not None:
            results.append(r)
            n_ok += 1

        if n_done % 200 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  Progress {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                  f"Feasible {n_ok}  Elapsed {elapsed:.0f}s  Remaining ~{eta:.0f}s")

        if r is not None and n_ok > 0 and n_ok % CHECKPOINT_EVERY == 0:
            _checkpoint(results, out_dir)

    # 1) Save and plot the collected results before worker cleanup to ensure they reach disk.
    results.sort(key=lambda r: (r["bat_price_factor"], r["PV_MW"], r["cons_target_pct"]))

    elapsed = time.time() - t_start
    print(f"\nScan complete: {n_ok}/{n_total} feasible, elapsed {elapsed:.1f}s")

    try:
        if results:
            base.save_to_excel(results)
            base.plot_results(results)
            cp = os.path.join(out_dir, "sensitivity_bat_sizing_merged_greenfield_checkpoint.xlsx")
            if os.path.exists(cp):
                os.remove(cp)
                print("Checkpoint file removed")
        else:
            print("No feasible solution; check the parameter settings")
    except Exception as e:
        print(f"[error] Failed to save/plot results: {e}")

    # 2) Force worker termination and exit immediately to bypass Pool.terminate()/join() hangs with Gurobi workers.
    for _p in list(getattr(pool, "_pool", [])):
        try:
            _p.terminate()        # Cross-platform: TerminateProcess on Windows, SIGTERM on Unix.
        except Exception:
            pass
    os._exit(0)
