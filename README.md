# 电氢微电网研究：数据与代码说明

本目录汇集电池储能（BESS）与电氢耦合微电网的工程运行数据、多场景容量优化代码，以及灵活性拐点和经济交叉点的分析与绘图脚本。

研究流程分为两条：

- **工程运行分析**：展示典型晴天、阴天及连续 168 小时内的功率平衡、电池荷电状态（SOC）和储氢状态，并提供纯电池离网仿真。
- **多场景分析**：对乡村、商业、工业、海岛、挪威、美国和赞比亚场景进行容量扫描，计算成本和离网持续时长，再分析储能需求的灵活性拐点及两种储能方案的经济交叉点。

本说明以当前代码的实际读写逻辑为准。目录中已包含仿真结果和绘图数据，可以从绘图层开始；完整容量优化需要另外配置 Gurobi，并处理第 5 节所述的输入路径。

## 1. 目录结构

```text
Data_and_Code/
├── README.md
├── Engineering_and_Inverse_Analysis/
│   ├── simulation_data_and_code/
│   │   ├── offgrid_ES.m                  # 纯电池离网仿真
│   │   └── data_7day.xlsx                # 7 天光伏与负荷输入
│   ├── power_balance_*.py               # 工程运行绘图
│   ├── *.xlsx                           # 典型日与 168 小时运行数据
│   └── figures/                         # 工程图输出
└── Simulation_of_Multiple_Scenarios/
    ├── Simulation_data_and_code/
    │   └── <Scenario>/                  # Rural、Commercial、Industry、Island、Norway、USA、Zambia
    │       ├── Code/                    # 容量优化、并行扫描、离网时长计算
    │       ├── Data/                    # 场景时间序列
    │       └── Result/                  # 已归档的计算结果
    └── Plot_data_and_code/
        ├── Flexibility Turning Point/
        │   ├── *.py                     # 拐点与方法敏感性分析
        │   ├── data/                    # domestic、global 绘图输入
        │   └── results/                 # 图、统计表和 clean 分图
        ├── Economic Breakeven Point/
        │   ├── *.py                     # 经济交叉点与成本敏感性分析
        │   ├── data/                    # sensitivity、global 绘图输入
        │   └── results/
        ├── compare_knee_breakeven_order.py
        ├── knee_method_breakeven_comparison.py
        └── results/knee_breakeven_order/ # 两类阈值的比较结果
```

`__pycache__/` 是 Python 缓存。各 `results/clean/` 主要保存供后续排版使用的无文字图或独立面板。

## 2. 环境与快速开始

### 2.1 环境

| 用途 | 所需环境 |
|---|---|
| Python 数据处理与绘图 | Python 3；`numpy`、`pandas`、`matplotlib`、`scipy`、`openpyxl` |
| 容量优化与参数扫描 | 上述环境，以及 `gurobipy` 和可用的 Gurobi 许可证 |
| 离网时长计算 | 上述 Python 数据处理与绘图环境；可选的 `tqdm` 用于进度显示，无需调用 Gurobi |
| 工程离网仿真 | MATLAB，使用 `xlsread`、`table`、`writetable` 等函数 |

项目未提供依赖锁定文件。可用 Python 3.12 新建独立环境，避免与已有环境的 NumPy/pandas 二进制版本冲突。以下 PowerShell 命令均从 **`Data_and_Code` 目录**执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy openpyxl tqdm
# 仅在需要重新运行容量优化时安装：
.\.venv\Scripts\python.exe -m pip install gurobipy
```

下文用 `python` 简写解释器；使用上述虚拟环境时，将其替换为 `.\.venv\Scripts\python.exe`，或先激活环境。安装 `gurobipy` 后仍需单独配置许可证。

### 2.2 使用随附数据生成主要图

先生成国内灵活性拐点，再生成依赖该结果的国内经济综合图：

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/plot_domestic_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_domestic_cases.py"
```

其余主要图可分别运行：

```powershell
python "Engineering_and_Inverse_Analysis/power_balance_stacked_bar.py"
python "Engineering_and_Inverse_Analysis/power_balance_168h_BESS_only.py"
python "Engineering_and_Inverse_Analysis/power_balance_168h_H2_BESS.py"

python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/plot_global_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/knee_cap_sensitivity_analysis.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_global_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_cost_factor_sensitivity.py"
```

路径含空格时保留引号。绘图脚本按自身位置定位数据，重跑会覆盖同名输出。多场景绘图使用 Matplotlib 的 `Agg` 后端；工程绘图未显式固定后端，无图形界面的环境可设置 `MPLBACKEND=Agg`。

`knee_method_comparison.py` 存在字段兼容问题，详见第 9 节，因此未列入上述直接运行流程。

## 3. 数据流与指标

```text
场景 Data/ 中的时序数据
    → Code/ 中的电池或电氢容量扫描
    → 容量、年化成本与 LCOE 结果
    → 离网持续时长计算（经济分析需要）
    → 整理至场景 Result/
    → 复制为 Plot_data_and_code 下的 battery.xlsx / hydrogen.xlsx
    → 灵活性拐点、经济交叉点及敏感性分析
    → results/ 中的图、统计表与作图数据
```

绘图输入是单独保存的工作簿副本。更新场景 `Result/` 后，绘图目录中的数据不会自动同步；对应关系见第 8 节。

| 字段或术语 | 含义 |
|---|---|
| `cons_target_%` | 可再生能源消纳率目标，常用 60、70、80、90、100% |
| `actual_cons_%` | 优化结果的实际消纳率 |
| `energy_pen_%` | 可再生能源能量渗透率；与消纳率目标是不同指标 |
| `total_MWh`、`total_MW` | 电池总能量容量与功率容量 |
| `storage_duration_h` | 电池容量相对于场景平均负荷的等效供电时长 |
| `min_offgrid_hours` | 多个停电起点中最短的离网持续时长 |
| `LCOE_USD_per_kWh` | 年化总成本除以全年总负荷，包含购电费，单位 USD/kWh |
| `bat_price_factor` | 电池方案的成本系数 |
| `storage_factor` | 电氢方案中共同作用于相关储能设备的成本系数 |

`storage_duration_h` 是容量指标，`min_offgrid_hours` 则来自逐时运行模拟，两者不能互换。各场景的负荷归一化、设备价格、效率和容量范围由对应脚本定义。

## 4. 工程运行与离网仿真

路径：`Engineering_and_Inverse_Analysis/`。

| 脚本 | 输入 | 功能与主要输出 |
|---|---|---|
| `power_balance_stacked_bar.py` | `typical_sunny_day.xlsx`、`typical_cloudy_day.xlsx` | 对比晴天与阴天功率堆叠、电池 SOC、储氢状态和加氢量；输出 `figures/figure_stacked_bar_hourly.{svg,png}` |
| `power_balance_168h_BESS_only.py` | `enough_BESS.xlsx`、`original_BESS.xlsx` | 分别绘制充足电池与原始电池配置的功率和 SOC；输出 `figures/figure_power_balance_{enough,original}_BESS.{svg,png}` |
| `power_balance_168h_H2_BESS.py` | `seven_day_offgrid_H2_BESS.xlsx` | 绘制功率、电池与氢 SOC、每日充放电能量占比；输出 `figures/figure1_power_balance_H2_BESS.{svg,png}` |
| `simulation_data_and_code/offgrid_ES.m` | 同目录 `data_7day.xlsx` | 模拟光伏与电池离网运行，输出断电时间及 `simulation_results.xlsx`；Excel 写入失败时尝试 CSV |

### 数据格式

- 典型日工作簿读取首张表，使用前 48 个半小时点。主要列为 `pv`、`电化学储能放电`、`电化学储能充电`、`燃料电池放电`、`制氢功率`、`充电桩负荷`、`直流和站用负荷`、`ess_soc`、`h_soc`、`h_load`；缺少 `购电功率` 或 `光伏上网` 时按零处理。
- 典型日脚本将 `ess_soc` 除以 6000、`h_soc` 除以 450，再转为百分数。替换工程配置时需要同步调整这两个基准值。虽然脚本及输出名含 `hourly`，实际仍按半小时点作图，没有聚合为小时数据。
- 168 小时工作簿读取 `SimulationResults`，要求 `Time_h`、`PV_kW`、`Load_kW`、`Battery_kW`、`Battery_SOC_pct`；电氢图另外使用 `Electrolyzer_kW`、`FuelCell_kW`、`H2_SOC_pct`。`Battery_kW` 正值表示充电，绘图时取负值表示对外供电。
- 当前 `original_BESS.xlsx` 有 164 个数据点，脚本补至 336 点：功率补零、SOC 保持最后值。补齐段用于统一图的时间范围，不代表系统继续供电。
- `data_7day.xlsx` 无表头，前两列依次为光伏功率和负荷功率，单位 kW，步长 0.5 h，共 336 点。

### MATLAB 入口

在 MATLAB 中将当前目录切换到 `Engineering_and_Inverse_Analysis/simulation_data_and_code/`，执行：

```matlab
offgrid_ES
```

当前默认电池容量为 11600 kWh、功率为 5800 kW，初始 SOC 为 0.713，SOC 上限为 0.9，停止阈值为 0.1，充放电效率均为 0.95。参数位于脚本开头。输出的 `simulation_results.xlsx` 使用 `SimulationResults` 工作表。

该 MATLAB 脚本只包含纯电池系统。随附的三份 168 小时工程工作簿不会由一次运行全部生成；若要更新 Python 图，需要将对应仿真输出另行整理为其指定文件名。

## 5. 多场景容量优化与离网持续时长

路径：`Simulation_of_Multiple_Scenarios/Simulation_data_and_code/`。

### 5.1 场景与时序输入

| 场景目录 | 场景 | 数据文件与读取格式 |
|---|---|---|
| `Rural` | 乡村，光伏 | `Data/Annual_8760.xlsx`：无表头的 `PV`、`Load` 表，第 2 列为数值 |
| `Commercial` | 商业，光伏 | 同上 |
| `Industry` | 工业，光伏 | 同上；绘图目录名为 `industrial` |
| `Island` | 海岛，风光 | `Data/Annual_8760.xlsx`：无表头的 `PV`、`Wind`、`Load` 表，第 1 列为数值 |
| `Norway` | 挪威，风光 | `Data/Norway.xlsx`：`train` 表，使用 `pv_production`、`wind_production`、`consumption` 等字段，由脚本截取与处理 |
| `USA` | 美国，光伏 | `Data/Annual_8760_clean.xlsx`：`pv_kw`、`load_kw` 表，含时间及数值表头 |
| `Zambia` | 赞比亚，光伏 | `Data/Annual_8760.xlsx`：无表头的 `PV`、`Load` 表，第 2 列为数值 |

模型使用小时级时序。多数输入直接包含 8760 点；挪威原始表长于 8760 点。各场景的格式和归一化逻辑不同，替换数据时需核对对应 `load_raw_data()` 或离网计算的读表函数。

### 5.2 脚本类型

| 文件模式 | 职责 |
|---|---|
| `microgrid_*bat_sizing_scan*.py` | 扫描可再生能源装机、消纳目标和电池成本系数，用 Gurobi 求解电池容量及运行，输出成本和能量指标 |
| `*_merged.py` | 将原有与新增电池合并为一个等效电池，同时保留原有总容量下界 |
| `*_greenfield.py` | 从零容量规划电池；国内场景还调整了装机扫描下限，并仅使用成本系数 1.0 |
| `microgrid_*h2_sizing_scan_joint.py` | 在场景给定的电池配置上优化电解槽、储氢罐和燃料电池容量；`joint` 表示电池、电解槽、储氢罐和燃料电池的投资成本同乘一个系数 |
| `*_parallel.py` | 导入同目录基础模型，以多进程执行参数组合扫描，保存检查点；通过环境变量 `N_WORKERS` 控制进程数 |
| `calc_bat_offgrid_duration.py` | 读取电池容量结果，逐个配置计算离网持续时长并写回工作簿 |
| `calc_h2_offgrid_duration.py` | 对电氢配置进行相应的离网持续时长计算并写回工作簿 |

海岛和挪威的容量扫描文件名含 `wind_pv`。各场景默认容量、成本、扫描范围和求解设置不同，不能只凭文件后缀判断模型：例如挪威、美国、赞比亚的普通电池脚本原有容量下界为零，而海岛普通电池脚本保留原有电池。`merged` 模型仍按电池总容量计入成本，不能将其解释为只计新增投资。

优化目标为年化设备投资、运维与购电费用之和。模型接受求解时限内找到的可行解，因此结果表不等于所有配置均已证明全局最优。电氢结果还会过滤氢设备容量全零的配置，部分全球场景另有渗透率回退点过滤；结果表并非所有扫描点的原始记录。

离网计算使用每天 00:00、06:00、12:00、18:00 共 1460 个停电起点，初始电池和储氢 SOC 为 90%，循环使用年度时序，单次最多模拟 8760 h。电氢调度在富余时先充电池、再制氢，不足时先放电池、再使用燃料电池。结果记录最短、平均、10% 分位、最长时长、存活比例及最差起点等。

离网脚本写回前会在备份不存在时尝试创建 `_backup.xlsx`，然后重写目标工作簿。运行时需关闭 Excel 中占用的目标文件。

### 5.3 重新运行前的路径准备

归档采用 `Data/`、`Code/`、`Result/` 分目录结构，但容量扫描和离网计算脚本目前默认在 **`Code/` 内寻找输入，并把结果写入 `Code/`**。仅改变终端当前目录不能解决这一问题。

可以将路径变量改为场景目录下的 `Data/` 和 `Result/`，或按现有代码将需要的输入复制到 `Code/`。以下以乡村场景为例，沿用当前默认路径：

```powershell
$scenarioDir = "Simulation_of_Multiple_Scenarios/Simulation_data_and_code/Rural"
Copy-Item -LiteralPath "$scenarioDir/Data/Annual_8760.xlsx" -Destination "$scenarioDir/Code/Annual_8760.xlsx"
$env:N_WORKERS = "4"

# 含原有电池容量下界的电池扫描
python "$scenarioDir/Code/microgrid_bat_sizing_scan_merged_parallel.py"
python "$scenarioDir/Code/calc_bat_offgrid_duration.py"

# 电氢扫描
python "$scenarioDir/Code/microgrid_h2_sizing_scan_joint_parallel.py"
python "$scenarioDir/Code/calc_h2_offgrid_duration.py"

# 从零规划的电池扫描，用于国内灵活性分析
python "$scenarioDir/Code/microgrid_bat_sizing_scan_merged_greenfield_parallel.py"
```

这些扫描可能产生大量年度优化任务。运行前在基础模型顶部核对容量范围、消纳目标、成本系数、求解时间限制及并行资源设置。容量扫描和离网脚本没有统一的命令行参数接口，主要通过源码常量配置。

若只重算归档结果的离网时长，可先把对应 `Result/sensitivity_*.xlsx` 复制到 `Code/`。乡村电池离网脚本默认处理 `sensitivity_bat_sizing_merged_results.xlsx`；处理 greenfield 或其他结果时，应修改 `INPUT_SENSITIVITY`。国内灵活性分析直接使用容量时长，不要求先补充离网时长列。

计算完成后，将需要归档的结果整理到场景 `Result/`，再按第 8 节更新绘图副本。

## 6. 灵活性拐点分析

路径：`Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/`。

该模块分析不同消纳目标下，电池等效供电时长随可再生能源渗透率变化的曲线。国内图使用从零规划的电池结果，比较最大曲率法、Kneedle 和 L-method 三种拐点识别方法；主图标示最大曲率法结果。

### 输入与方法

- 国内输入：`data/domestic/{rural,commercial,industrial,island}/battery.xlsx`。
- 全球输入：`data/global/{Norway,USA,Zambia}/battery.xlsx`。同目录 `hydrogen.xlsx` 和 `battery_backup.xlsx` 不由该模块的全球绘图脚本读取。
- 关键字段为 `cons_target_%`、`energy_pen_%`、`storage_duration_h`。国内读表逻辑还支持部分替代列名或从容量与配置的平均负荷换算时长。
- 曲线采用 PCHIP 保形插值，国内主图默认分析上限为 200 h、增长起点为时长首次达到 1 h。Kneedle 在脚本内实现，无需另装 `kneed` 包。全球与国内脚本的归一化和边界排除参数不同，应随数据一起保留算法配置。
- 国内主图及上限敏感性分析筛选电池成本系数 1.0；全球绘图没有相同的系数筛选步骤，更新其输入时应先确定使用的数据子集。

### 脚本与主要产物

| 脚本 | 功能 | `results/` 下主要产物 |
|---|---|---|
| `plot_domestic_cases.py` | 国内四场景曲线、三方法拐点及拐点迁移 | `Figure1_storage_requirement_turning_points.svg`、`Figure1_caption.txt`、`domestic_flexibility_knees_three_methods.xlsx`、`domestic_flexibility_knee_method_differences.csv`，以及 `clean/domestic/` 分图 |
| `plot_global_cases.py` | 全球三场景曲线与最大曲率拐点 | `global_comparison.svg`、`global_flexibility_knees_max_curvature.csv`，以及 `clean/global/` 分图 |
| `knee_cap_sensitivity_analysis.py` | 在 150、200、250 h 上限下重算三方法结果，汇总后调用偏差绘图 | `cap_sensitivity_all_results.csv`、`cap_sensitivity_statistics.csv`、各上限 SVG 等 |
| `knee_cap_deviation_plot.py` | 读取上一步汇总表，以 200 h 为基准计算拐点漂移与方法稳健性 | `cap_sensitivity_deviation_from_200h.csv`、`cap_sensitivity_method_robustness.csv`、`cap_sensitivity_three_methods*.svg` |
| `knee_method_comparison.py` | 比较三方法相对于其均值的偏差 | `knee_method_deviations.svg` 和作图 CSV；当前字段兼容限制见第 9 节 |

国内拐点 Excel 包含 `method_comparison` 宽表和 `long_format` 长表。宽表中的 `scenario`、`consumption_target_%`、`maximum_curvature_penetration_%` 是国内经济综合图的直接输入。

上限敏感性默认一次处理全部三档，也可单独生成某档：

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/knee_cap_sensitivity_analysis.py" --y-cap 200
```

`--data-only` 与 `--y-cap` 配合使用时，仅导出该档临时数值结果并跳过单档绘图。需要完整的跨上限统计及偏差图时，运行不带参数的入口；默认流程会生成各档图并自动调用偏差绘图脚本。

## 7. 经济交叉点与阈值比较

### 7.1 经济分析

路径：`Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/`。

国内输入在 `data/sensitivity/{rural,commercial,industrial,island}/`，全球输入在 `data/global/{Norway,USA,Zambia}/`，每个场景均读取 `battery.xlsx` 与 `hydrogen.xlsx`。

代码读取首张工作表，要求 `cons_target_%`、`energy_pen_%`、`min_offgrid_hours`、`LCOE_USD_per_kWh`。此外，**第一列必须是成本系数**：当前电池表为 `bat_price_factor`，电氢表为 `storage_factor`；脚本按列位置筛选，不能随意调整第一列。

分析过程如下：

1. 按相同消纳目标与成本系数配对电池和电氢结果。
2. 将电氢方案 LCOE 按渗透率线性插值到电池方案的渗透率点；超出电氢数据范围时使用端点值。
3. 计算 `ΔLCOE = LCOE_BESS − LCOE_H2`。正值表示电氢方案成本较低，负值表示电池方案成本较低。
4. 沿曲线寻找成本差为零或发生符号变化的位置，插值得到经济交叉点。

该模块使用已计算的 LCOE，不重新优化设备容量。方案匹配依据渗透率；离网时长使用电池方案的坐标，因此不应把这里的比较解释为同时严格匹配渗透率和离网时长。

| 脚本 | 设置与功能 | 主要输出 |
|---|---|---|
| `plot_domestic_cases.py` | 国内系数 1.0；结合最大曲率拐点，计算经济交叉点与物理拐点的渗透率差 | `results/Figure2_storage_pathway_economics.svg`、同名前缀的 `_source_data.csv` 和 `_caption.txt`；`clean/domestic/` 分图 |
| `plot_global_cases.py` | 挪威、美国、赞比亚，系数 1.0 | `results/global_comparison.svg`、`clean/global/` 分图 |
| `plot_cost_factor_sensitivity.py` | 国内四场景，系数 1.0、0.9、0.7、0.5 | `results/sensitivity/<scenario>.svg`、`clean/sensitivity/<scenario>/factor_*.svg` |

国内综合图当前离网时长上限为 500 h，全球及成本敏感性图为 200 h。国内综合图依赖第 6 节的拐点 Excel，并输出 `Δp = p_economic − p_physical`，单位为百分点。没有交叉点时保留缺失值及相应状态。

### 7.2 三种拐点方法与经济交叉点的比较

`Plot_data_and_code/` 根目录的两个脚本进一步比较国内各场景、各消纳目标下的阈值顺序与偏差：

| 脚本 | 用途 |
|---|---|
| `compare_knee_breakeven_order.py` | 对各拐点方法与经济交叉点进行配对，输出明细、分组汇总及文字统计 |
| `knee_method_breakeven_comparison.py` | 绘制各拐点相对于经济交叉点的偏差，导出作图数据与图注 |

结果集中写入 `Plot_data_and_code/results/knee_breakeven_order/`。这两个脚本支持当前英文拐点字段，也保留中文字段兼容逻辑。

先生成国内灵活性 CSV，再执行以下命令；经济交叉点直接从经济模块的工作簿重算，无需先生成经济图：

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/compare_knee_breakeven_order.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/knee_method_breakeven_comparison.py"
```

- 统计脚本输出 `comparison_details.csv`、`summary_by_method.csv`、`summary_by_scenario_method.csv`、`summary.txt`。可用 `--tolerance-pp` 调整重合容差（默认 0.0001 个百分点），用 `--output-dir` 更改输出目录。
- 比较图脚本可独立于统计脚本运行，输出 `knee_method_breakeven_deviations.{svg,pdf,png}`、同名前缀的 `_plot_data.csv`、`_caption.txt` 和 `clean/` SVG。
- 统计差值为 `p_economic − p_flexibility`，图的横轴为 `p_flexibility − p_economic`，两者符号相反。
- 无经济交叉点的情况单列，不计入先后比例的分母。比较图使用 100% 作这类记录的定位基准，真实交叉点仍为空；灰色跨度表示三方法范围。

## 8. 仿真结果与绘图输入的对应关系

以下路径中，仿真结果均相对 `Simulation_data_and_code/<Scenario>/Result/`；绘图路径相对 `Plot_data_and_code/`。表中普通对应关系经文件 SHA-256 核对，源文件与绘图副本内容相同。

| 场景 | 结果文件 | 绘图用途 |
|---|---|---|
| Rural、Commercial、Industry | `sensitivity_bat_sizing_merged_greenfield_results.xlsx` | `Flexibility Turning Point/data/domestic/<scenario>/battery.xlsx` |
| Island | `sensitivity_wind_pv_bat_sizing_greenfield_results.xlsx` | 国内海岛灵活性 `battery.xlsx` |
| Rural、Commercial、Industry | `sensitivity_bat_sizing_merged_results.xlsx` | `Economic Breakeven Point/data/sensitivity/<scenario>/battery.xlsx` |
| Island | `sensitivity_wind_pv_bat_sizing_results.xlsx` | 国内海岛经济分析 `battery.xlsx` |
| 国内四场景 | `sensitivity_*h2_sizing_joint_results.xlsx` | 对应经济分析 `hydrogen.xlsx`；Island 文件名含 `wind_pv` |
| Norway | `sensitivity_wind_pv_bat_sizing_merged_results.xlsx`、`sensitivity_wind_pv_h2_sizing_joint_results.xlsx` | 两个绘图模块的 `data/global/Norway/{battery,hydrogen}.xlsx` |
| USA | `sensitivity_bat_sizing_merged_results.xlsx`、`sensitivity_h2_sizing_joint_results.xlsx` | 两个绘图模块的 `data/global/USA/{battery,hydrogen}.xlsx` |
| Zambia | `sensitivity_bat_sizing_merged_results.xlsx` | 两个绘图模块的 `data/global/Zambia/battery_backup.xlsx` |
| Zambia | `sensitivity_h2_sizing_joint_results.xlsx` | 两个绘图模块的 `data/global/Zambia/hydrogen.xlsx` |

场景名映射为 `Rural → rural`、`Commercial → commercial`、`Industry → industrial`、`Island → island`。全球路径保留国家目录名的大小写。

**赞比亚的例外：** 当前绘图实际读取的 `battery.xlsx` 与归档 `Result/` 中的电池工作簿不完全相同；一致的是 `battery_backup.xlsx`。更新赞比亚结果时，需要先确认希望复现的输入版本。

## 9. 当前代码与归档的复现注意事项

1. **仿真路径尚未统一。** 归档中的 `Data/`、`Result/` 与代码默认的 `Code/` 路径不一致，按第 5.3 节准备输入后再运行。
2. **一个方法比较脚本尚未适配英文表头。** `knee_method_comparison.py` 要求 `场景`、`消纳率目标(%)`、`曲率最大_能量渗透率(%)` 等中文列及中文场景值，而当前 `plot_domestic_cases.py` 和随附 CSV 使用 `scenario`、`consumption_target_%`、`maximum_curvature_penetration_%` 等英文列。直接衔接会报 `Missing required columns`；使用该脚本前需统一全部字段和场景值映射。
3. **部分结果是历史产物。** 两个绘图模块中的 `results/domestic_comparison.svg`、灵活性模块中的 `global_flexibility_knees_kneedle.csv`，以及经济模块中的 `economic_crossover_vs_flexibility_knee_x_comparison.csv`，均不由当前对应绘图入口重新生成。当前主图以 `Figure1_*`、`Figure2_*` 和脚本实际保存的文件名为准。
4. **工作簿副本与结果不会自动联动。** 重新求解后需主动整理结果、补充经济分析需要的离网字段，并同步绘图输入。
5. **运行参数主要保存在源码中。** 图形范围、容量扫描范围、初始 SOC、设备效率、成本系数及拐点算法设置均会影响结果。比较不同版本时，应同时记录输入工作簿和源码配置。

本 README 的整理依据为源码静态阅读、Python 语法检查和工作簿结构/对应关系核对；未重新执行完整容量优化、MATLAB 仿真或全部绘图，不能据此视为端到端复现已通过。
