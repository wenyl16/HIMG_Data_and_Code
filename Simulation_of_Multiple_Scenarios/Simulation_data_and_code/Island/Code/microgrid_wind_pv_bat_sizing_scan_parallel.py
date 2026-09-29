"""
风光储电池容量优化 —— 多进程并行版 (部署到多核服务器)
模型/逻辑/输出与 microgrid_wind_pv_bat_sizing_scan.py 完全一致, 仅主循环并行化。
用法:
    python microgrid_wind_pv_bat_sizing_scan_parallel.py
    N_WORKERS=32 python microgrid_wind_pv_bat_sizing_scan_parallel.py
"""
import os
import signal
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gurobipy as gp
import microgrid_wind_pv_bat_sizing_scan as base

_N_RAW = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))
if os.name == "nt" and _N_RAW > 30:
    N_WORKERS = 30
else:
    N_WORKERS = _N_RAW
CHECKPOINT_EVERY = 1000

_W = {}


def _init_worker():
    """每个工作进程启动时执行一次: 限定单线程 + 载入数据。"""
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    pv_raw, wind_raw, load = base.load_raw_data()
    _W["pv_raw"] = pv_raw
    _W["wind_raw"] = wind_raw
    _W["load"] = load
    _W["prices"] = base.build_prices()


def _worker(task):
    factor, total_kw, ct = task
    wind_kw = total_kw * base.RATIO_WIND
    pv_kw = total_kw * base.RATIO_PV
    pv = base.scale_pv(_W["pv_raw"], pv_kw)
    wind = base.scale_wind(_W["wind_raw"], wind_kw)
    return base.solve_scenario(pv, wind, _W["load"], _W["prices"], pv_kw, wind_kw, ct,
                               bat_price_factor=factor)


def _checkpoint(results, out_dir):
    try:
        base.save_to_excel(
            results,
            filename=os.path.join(out_dir, "sensitivity_wind_pv_bat_sizing_checkpoint.xlsx"),
        )
    except Exception as e:
        print(f"  [warn] 检查点写入失败: {e}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()

    renew_values = list(range(base.RENEW_SCAN_MIN, base.RENEW_SCAN_MAX + base.RENEW_SCAN_STEP,
                              base.RENEW_SCAN_STEP))
    tasks = [(factor, total_kw, ct)
             for factor in base.BAT_PRICE_FACTORS
             for total_kw in renew_values
             for ct in base.CONSUMPTION_TARGETS]
    n_total = len(tasks)
    n_workers = min(N_WORKERS, n_total)

    print(f"并行扫描(风光电池): {n_total} 场景, 使用 {n_workers} 进程(每进程1线程)")
    print(f"风光总容量: {base.RENEW_SCAN_MIN/1000:.3f}~{base.RENEW_SCAN_MAX/1000:.1f} MW, "
          f"步长 {base.RENEW_SCAN_STEP/1000:.1f} MW, Wind:PV = 1000:835")
    print(f"电池价格倍数: {base.BAT_PRICE_FACTORS}")
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

        if n_done % 200 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"  进度 {n_done}/{n_total} ({n_done/n_total*100:.0f}%)  "
                  f"可行 {n_ok}  已用 {elapsed:.0f}s  剩余 ~{eta:.0f}s")

        if r is not None and n_ok > 0 and n_ok % CHECKPOINT_EVERY == 0:
            _checkpoint(results, out_dir)

    results.sort(key=lambda r: (r["bat_price_factor"], r["Total_Renewable_MW"], r["cons_target_pct"]))

    elapsed = time.time() - t_start
    print(f"\n扫描完成: {n_ok}/{n_total} 可行, 耗时 {elapsed:.1f}s")

    if results:
        base.save_to_excel(results)
        base.plot_results(results)
        cp = os.path.join(out_dir, "sensitivity_wind_pv_bat_sizing_checkpoint.xlsx")
        if os.path.exists(cp):
            os.remove(cp)
            print("已清理检查点文件")
    else:
        print("无可行解, 请检查参数设置")

    for _p in list(getattr(pool, "_pool", [])):
        try:
            os.kill(_p.pid, signal.SIGKILL)
        except Exception:
            pass
    os._exit(0)
