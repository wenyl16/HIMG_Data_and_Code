"""
电池储能容量优化 —— 多进程并行版 (部署到多核服务器)
模型/逻辑/输出与 microgrid_bat_sizing_scan_merged.py 完全一致, 仅把主循环改写为多进程并行。
  - 每个场景一个独立 Gurobi 模型, 场景间无依赖 -> 易并行
  - 每个工作进程单线程 (gp.setParam Threads=1), 进程数≈物理核数, 充分利用多核
  - 工作进程启动时一次性载入数据 (pv_raw/load/prices), 避免重复IO与大数组传递
  - 周期性写检查点, 防止长跑中途崩溃丢失全部结果
  - 收尾: 先保存/画图, 再 terminate worker + os._exit, 绕开 Pool 清理在某些环境(如服务器上 Gurobi worker)的卡死

用法:
    python microgrid_bat_sizing_scan_merged_parallel.py
    # 用环境变量 N_WORKERS 指定进程数(默认 os.cpu_count()); Xeon 6338N(32物理核)建议 N_WORKERS=32
    N_WORKERS=32 python microgrid_bat_sizing_scan_merged_parallel.py
"""
import os
import signal
import sys
import time
from multiprocessing import Pool

# 保证从任意工作目录运行都能 import 同目录下的基线模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gurobipy as gp
import microgrid_bat_sizing_scan_merged_greenfield as base

_N_RAW = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))
if os.name == "nt" and _N_RAW > 30:
    # Windows: 多进程 Pool 内部用 WaitForMultipleObjects 等待 worker, 上限 63 句柄;
    # 每个 worker 约占 2 句柄, 故 worker 数须 ≤ ~30, 否则报 "need at most 63 handles"。
    # (Linux 服务器的 wait() 无此限制, 不受影响)
    N_WORKERS = 30
else:
    N_WORKERS = _N_RAW
CHECKPOINT_EVERY = 1000   # 每累计这么多条可行结果写一次检查点

# 工作进程全局数据 (每个子进程在 _init_worker 中填充)
_W = {}


def _init_worker():
    """每个工作进程启动时执行一次: 限定单线程 + 载入数据。"""
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)        # 每进程单线程, 配合多进程避免核过度订阅
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
        print(f"  [warn] 检查点写入失败: {e}")


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

    print(f"并行扫描(电池, 单电池合并版): {n_total} 场景, 使用 {n_workers} 进程(每进程1线程)")
    print(f"PV范围: {base.PV_SCAN_MIN/1000:.1f}~{base.PV_SCAN_MAX/1000:.1f} MW, 步长 {base.PV_SCAN_STEP/1000:.1f} MW")
    print(f"电池价格倍数(相对BAT_CAPEX): {base.BAT_PRICE_FACTORS}")
    print()

    results = []
    n_done = 0
    n_ok = 0
    pool = Pool(n_workers, initializer=_init_worker)   # 不用 with: 避免退出时 terminate+join 卡死
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

    # 1) 先保存/画图 (结果已全部收集, 不依赖 worker 清理) —— 保证结果一定落盘
    results.sort(key=lambda r: (r["bat_price_factor"], r["PV_MW"], r["cons_target_pct"]))

    elapsed = time.time() - t_start
    print(f"\n扫描完成: {n_ok}/{n_total} 可行, 耗时 {elapsed:.1f}s")

    try:
        if results:
            base.save_to_excel(results)
            base.plot_results(results)
            cp = os.path.join(out_dir, "sensitivity_bat_sizing_merged_greenfield_checkpoint.xlsx")
            if os.path.exists(cp):
                os.remove(cp)
                print("已清理检查点文件")
        else:
            print("无可行解, 请检查参数设置")
    except Exception as e:
        print(f"[error] 保存/画图失败: {e}")

    # 2) 强杀 worker + 立即退出主进程, 绕开 Pool.terminate()/join() 在 Gurobi worker 上的卡死
    for _p in list(getattr(pool, "_pool", [])):
        try:
            _p.terminate()        # 跨平台: Windows 走 TerminateProcess, Unix 走 SIGTERM
        except Exception:
            pass
    os._exit(0)
