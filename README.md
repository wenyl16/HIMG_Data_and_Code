# Electricity-Hydrogen Microgrid Research: Data and Code

This directory contains engineering operation data for battery energy storage systems (BESS) and electricity-hydrogen microgrids, capacity optimization models for multiple scenarios, and scripts for analyzing and plotting flexibility knees and economic breakeven points.

The work follows two workflows:

- **Engineering operation analysis:** visualize power balance, battery state of charge (SOC), and hydrogen storage levels for typical sunny and cloudy days and a 168-hour period; simulate off-grid operation with battery storage alone.
- **Multiple-scenario analysis:** scan system capacities for rural, commercial, industrial, island, Norway, USA, and Zambia scenarios; calculate costs and off-grid durations; identify flexibility knees in storage requirements and economic breakeven points between battery and hydrogen storage pathways.

This guide follows the current code's input and output behavior. Simulation results and plotting data are included, so the plotting workflows can be run directly. Repeating the capacity optimization requires Gurobi and the input-path preparation described in Section 5.

## 1. Directory structure

```text
HIMG_Data_and_Code/
├── README.md
├── Engineering_and_Inverse_Analysis/
│   ├── simulation_data_and_code/
│   │   ├── offgrid_ES.m                  # Battery-only off-grid simulation
│   │   └── data_7day.xlsx                # Seven-day PV and load inputs
│   ├── power_balance_*.py               # Engineering operation plots
│   ├── *.xlsx                           # Typical-day and 168-hour operation data
│   └── figures/                         # Engineering figure outputs
└── Simulation_of_Multiple_Scenarios/
    ├── Simulation_data_and_code/
    │   └── <Scenario>/                  # Rural, Commercial, Industry, Island, Norway, USA, Zambia
    │       ├── Code/                    # Capacity optimization, parallel scans, off-grid duration
    │       ├── Data/                    # Scenario time series
    │       └── Result/                  # Archived simulation results
    └── Plot_data_and_code/
        ├── Flexibility Turning Point/
        │   ├── *.py                     # Knee detection and method sensitivity analysis
        │   ├── data/                    # Domestic and global plotting inputs
        │   └── results/                 # Figures, statistics, and clean panels
        ├── Economic Breakeven Point/
        │   ├── *.py                     # Breakeven and cost sensitivity analysis
        │   ├── data/                    # Sensitivity and global plotting inputs
        │   └── results/
        ├── compare_knee_breakeven_order.py
        ├── knee_method_breakeven_comparison.py
        └── results/knee_breakeven_order/ # Comparisons between the two thresholds
```

`__pycache__/` contains Python cache files. The `results/clean/` directories mainly contain figures without text and individual panels for subsequent layout work.

## 2. Environment and quick start

### 2.1 Environment

| Task | Requirements |
|---|---|
| Python data processing and plotting | Python 3; `numpy`, `pandas`, `matplotlib`, `scipy`, and `openpyxl` |
| Capacity optimization and parameter scans | The Python environment above, plus `gurobipy` and a valid Gurobi license |
| Off-grid duration calculation | The Python data processing and plotting environment above; optional `tqdm` for progress bars; Gurobi is not called |
| Engineering off-grid simulation | MATLAB, using functions including `xlsread`, `table`, and `writetable` |

The project does not include a dependency lock file. A separate Python 3.12 environment can help avoid NumPy/pandas binary-version conflicts with existing environments. Run the following PowerShell commands from **`HIMG_Data_and_Code`**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy openpyxl tqdm
# Install only if you need to repeat the capacity optimization:
.\.venv\Scripts\python.exe -m pip install gurobipy
```

The commands below use `python` as shorthand. When using this virtual environment, replace it with `.\.venv\Scripts\python.exe` or activate the environment first. Installing `gurobipy` does not configure the required license.

### 2.2 Generate the main figures from the supplied data

Generate the domestic flexibility knees first, then the domestic economic figure that uses those results. The method comparison also reads the domestic flexibility output:

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/plot_domestic_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_domestic_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/knee_method_comparison.py"
```

The other main figures can be generated separately:

```powershell
python "Engineering_and_Inverse_Analysis/power_balance_stacked_bar.py"
python "Engineering_and_Inverse_Analysis/power_balance_168h_BESS_only.py"
python "Engineering_and_Inverse_Analysis/power_balance_168h_H2_BESS.py"

python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/plot_global_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/knee_cap_sensitivity_analysis.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_global_cases.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/plot_cost_factor_sensitivity.py"
```

Keep quotation marks around paths containing spaces. Plotting scripts locate their inputs relative to their own files; rerunning a script overwrites outputs with the same names. The multiple-scenario plotting scripts use Matplotlib's `Agg` backend. The engineering plotting scripts do not explicitly select a backend; set `MPLBACKEND=Agg` when running without a graphical display.

## 3. Data flow and metrics

```text
Time series in each scenario's Data/
    → Battery or battery-hydrogen capacity scans in Code/
    → Capacity, annualized cost, and LCOE results
    → Off-grid duration calculations (required for economic analysis)
    → Archive results in the scenario's Result/
    → Copy to battery.xlsx / hydrogen.xlsx under Plot_data_and_code/
    → Flexibility knees, economic breakeven points, and sensitivity analysis
    → Figures, statistics, and plotting data in results/
```

Plotting inputs are separate workbook copies. Updating a scenario's `Result/` directory does not update them automatically; Section 8 lists the corresponding files.

| Field or term | Meaning |
|---|---|
| `cons_target_%` | Renewable energy utilization target, commonly 60, 70, 80, 90, or 100% |
| `actual_cons_%` | Actual renewable energy utilization in the optimization result |
| `energy_pen_%` | Renewable energy penetration; a separate metric from the utilization target |
| `total_MWh`, `total_MW` | Total battery energy capacity and power capacity |
| `storage_duration_h` | Battery energy capacity divided by the scenario's average load |
| `min_offgrid_hours` | Shortest off-grid operating duration across the tested outage start times |
| `LCOE_USD_per_kWh` | Annualized total cost divided by total annual load, including grid electricity purchases, in USD/kWh |
| `bat_price_factor` | Cost factor for the battery pathway |
| `storage_factor` | Common cost factor applied to the relevant storage equipment in the battery-hydrogen pathway |

`storage_duration_h` is a capacity-based measure; `min_offgrid_hours` comes from an hourly operating simulation. They are not interchangeable. Each scenario's scripts define its load normalization, equipment prices, efficiencies, and capacity ranges.

## 4. Engineering operation and off-grid simulation

Directory: `Engineering_and_Inverse_Analysis/`.

| Script | Inputs | Purpose and main outputs |
|---|---|---|
| `power_balance_stacked_bar.py` | `typical_sunny_day.xlsx`, `typical_cloudy_day.xlsx` | Compare sunny/cloudy-day power balance, battery SOC, hydrogen storage, and hydrogen demand; outputs `figures/figure_stacked_bar_hourly.{svg,png}` |
| `power_balance_168h_BESS_only.py` | `enough_BESS.xlsx`, `original_BESS.xlsx` | Plot power and SOC for sufficient and original battery capacities; outputs `figures/figure_power_balance_{enough,original}_BESS.{svg,png}` |
| `power_balance_168h_H2_BESS.py` | `seven_day_offgrid_H2_BESS.xlsx` | Plot power, battery/hydrogen SOC, and daily shares of charging/discharging energy; outputs `figures/figure1_power_balance_H2_BESS.{svg,png}` |
| `simulation_data_and_code/offgrid_ES.m` | `data_7day.xlsx` in the same directory | Simulate off-grid PV and battery operation, report the outage time, and write `simulation_results.xlsx`; attempt CSV output if the Excel write fails |

### Data formats

- Typical-day workbooks use the first sheet and the first 48 half-hour observations. The main columns are `pv`, `battery_discharge_kw`, `battery_charge_kw`, `fuel_cell_kw`, `electrolyzer_kw`, `ev_charging_load_kw`, `dc_station_load_kw`, `ess_soc`, `h_soc`, and `h_load`. Missing `grid_purchase_kw` or `pv_export_kw` columns are treated as zero.
- The typical-day script divides `ess_soc` by 6000 and `h_soc` by 450 before converting to percentages. Update these reference values if the engineering configuration changes. Although the script and output names include `hourly`, the figures use half-hour observations without hourly aggregation.
- The 168-hour workbooks use the `SimulationResults` sheet and require `Time_h`, `PV_kW`, `Load_kW`, `Battery_kW`, and `Battery_SOC_pct`. The battery-hydrogen figure also uses `Electrolyzer_kW`, `FuelCell_kW`, and `H2_SOC_pct`. Positive `Battery_kW` means charging; the plots reverse its sign to show power supplied to the system.
- The supplied `original_BESS.xlsx` contains 164 observations. The script extends it to 336 points by filling power with zero and holding SOC at its final value. This extension aligns the plot time ranges; it does not represent continued power supply.
- `data_7day.xlsx` has no header. Its first two columns are PV power and load power in kW, with 336 observations at 0.5-hour intervals.

### MATLAB entry point

In MATLAB, change the current directory to `Engineering_and_Inverse_Analysis/simulation_data_and_code/` and run:

```matlab
offgrid_ES
```

The defaults are 11600 kWh of battery energy capacity, 5800 kW of battery power, initial SOC 0.713, maximum SOC 0.9, stopping threshold 0.1, and charge/discharge efficiencies of 0.95. These parameters are at the top of the script. The output `simulation_results.xlsx` uses the `SimulationResults` sheet.

This MATLAB script models the battery-only system. A single run does not generate all three supplied 168-hour engineering workbooks. To update the Python figures, prepare the corresponding simulation outputs under the filenames expected by each plotting script.

## 5. Capacity optimization and off-grid duration across scenarios

Directory: `Simulation_of_Multiple_Scenarios/Simulation_data_and_code/`.

### 5.1 Scenarios and time-series inputs

| Scenario directory | Scenario | Data file and reading format |
|---|---|---|
| `Rural` | Rural, PV | `Data/Annual_8760.xlsx`: headerless `PV` and `Load` sheets; values in column 2 |
| `Commercial` | Commercial, PV | Same format as Rural |
| `Industry` | Industrial, PV | Same format as Rural; plotting directory is named `industrial` |
| `Island` | Island, wind and PV | `Data/Annual_8760.xlsx`: headerless `PV`, `Wind`, and `Load` sheets; values in column 1 |
| `Norway` | Norway, wind and PV | `Data/Norway.xlsx`: `train` sheet, including `pv_production`, `wind_production`, and `consumption`; the script selects and processes the required rows |
| `USA` | USA, PV | `Data/Annual_8760_clean.xlsx`: `pv_kw` and `load_kw` sheets, with headers for time and values |
| `Zambia` | Zambia, PV | `Data/Annual_8760.xlsx`: headerless `PV` and `Load` sheets; values in column 2 |

The models use hourly time series. Most inputs contain 8760 observations; the original Norway table is longer. Reading formats and normalization differ by scenario. When replacing data, check the relevant `load_raw_data()` function or the off-grid script's workbook reader.

### 5.2 Script types

| Filename pattern | Purpose |
|---|---|
| `microgrid_*bat_sizing_scan*.py` | Scan renewable capacity, utilization targets, and battery cost factors; use Gurobi to optimize battery capacity and operation; export cost and energy metrics |
| `*_merged.py` | Combine existing and additional batteries into one equivalent battery while retaining the existing capacity as a lower bound |
| `*_greenfield.py` | Size battery storage from zero; domestic scenarios also lower the starting renewable capacity and use only cost factor 1.0 |
| `microgrid_*h2_sizing_scan_joint.py` | Optimize electrolyzer, hydrogen tank, and fuel-cell capacities with the scenario's fixed battery configuration; `joint` applies the same multiplier to all four storage CAPEX terms |
| `*_parallel.py` | Import the base model from the same directory, scan parameter combinations with multiple processes, and save checkpoints; set the worker count through `N_WORKERS` |
| `calc_bat_offgrid_duration.py` | Read battery sizing results, calculate off-grid duration for each configuration, and write the results back to the workbook |
| `calc_h2_offgrid_duration.py` | Calculate off-grid duration for battery-hydrogen configurations and write the results back to the workbook |

The Island and Norway capacity-scan filenames include `wind_pv`. Default capacities, costs, scan ranges, and solver settings vary across scenarios, so filename suffixes alone do not identify all model assumptions. For example, the ordinary battery scripts for Norway, USA, and Zambia have zero lower bounds for existing capacity, while the ordinary Island battery script retains the existing battery. The `merged` models charge for total battery capacity, including the existing capacity.

The objective combines annualized equipment investment, operation and maintenance, and grid electricity purchases. The models accept feasible solutions found within the time limit, so the result tables do not establish global optimality for every configuration. Battery-hydrogen outputs also exclude configurations with all hydrogen equipment capacities equal to zero. Some global scenarios apply a further filter where energy penetration decreases as capacity increases. The output tables therefore do not contain every raw scan result.

Off-grid calculations test 1460 outage start times: 00:00, 06:00, 12:00, and 18:00 on each day. Initial battery and hydrogen SOC are 90%. The annual series wraps around, with a maximum of 8760 simulated hours per outage. With surplus power, the battery-hydrogen dispatch charges the battery before producing hydrogen; with a deficit, it discharges the battery before using the fuel cell. Outputs include minimum, mean, 10th-percentile, and maximum duration, survival rate, and the worst outage start time.

Before writing results back, the off-grid scripts attempt to create an `_backup.xlsx` file if it does not already exist, then rewrite the target workbook. Close the target file in Excel before running these scripts.

### 5.3 Prepare the paths before rerunning

The archive separates `Data/`, `Code/`, and `Result/`, but the capacity-scan and off-grid scripts currently **look for inputs in `Code/` and write results to `Code/`**. Changing only the terminal's working directory does not change these paths.

Either update the path variables to use the scenario's `Data/` and `Result/` directories or copy the required inputs into `Code/`. The following Rural example uses the current defaults:

```powershell
$scenarioDir = "Simulation_of_Multiple_Scenarios/Simulation_data_and_code/Rural"
Copy-Item -LiteralPath "$scenarioDir/Data/Annual_8760.xlsx" -Destination "$scenarioDir/Code/Annual_8760.xlsx"
$env:N_WORKERS = "4"

# Battery scan retaining the existing-capacity lower bound
python "$scenarioDir/Code/microgrid_bat_sizing_scan_merged_parallel.py"
python "$scenarioDir/Code/calc_bat_offgrid_duration.py"

# Battery-hydrogen scan
python "$scenarioDir/Code/microgrid_h2_sizing_scan_joint_parallel.py"
python "$scenarioDir/Code/calc_h2_offgrid_duration.py"

# Greenfield battery scan for domestic flexibility analysis
python "$scenarioDir/Code/microgrid_bat_sizing_scan_merged_greenfield_parallel.py"
```

These scans can involve many annual optimization problems. Before running, review the capacity ranges, utilization targets, cost factors, time limits, and parallel resource settings at the top of the base model. The capacity-scan and off-grid scripts are configured mainly through source constants and do not share a command-line parameter interface.

To recalculate off-grid durations from archived results, first copy the relevant `Result/sensitivity_*.xlsx` workbook into `Code/`. The Rural battery off-grid script reads `sensitivity_bat_sizing_merged_results.xlsx` by default. Change `INPUT_SENSITIVITY` to process greenfield or other result files. Domestic flexibility analysis uses capacity-based storage duration directly and does not require the off-grid duration columns.

After calculation, archive the required results under the scenario's `Result/` directory and update the plotting copies listed in Section 8.

## 6. Flexibility-knee analysis

Directory: `Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/`.

This module analyzes how battery storage duration changes with renewable energy penetration at different utilization targets. Domestic figures use greenfield battery results and compare three knee-detection methods: maximum curvature, Kneedle, and the L-method. The main figure marks the maximum-curvature knees.

### Inputs and methods

- Domestic inputs: `data/domestic/{rural,commercial,industrial,island}/battery.xlsx`.
- Global inputs: `data/global/{Norway,USA,Zambia}/battery.xlsx`. This module's global plotting script does not read the neighboring `hydrogen.xlsx` or `battery_backup.xlsx` files.
- The key fields are `cons_target_%`, `energy_pen_%`, and `storage_duration_h`. Domestic readers also accept some alternative English field names or calculate duration from capacity and the configured average load.
- Curves use shape-preserving PCHIP interpolation. The domestic main figure uses a 200-hour analysis cap and defines the start of storage growth as the first point reaching 1 hour. Kneedle is implemented within the scripts; the `kneed` package is not required. Domestic and global scripts use different normalization and boundary-exclusion settings, which should be retained alongside the data.
- The domestic main figure and duration-cap sensitivity analysis select battery cost factor 1.0. The global plotting script does not apply the same factor filter; choose the intended input subset when updating its workbooks.

### Scripts and main outputs

| Script | Purpose | Main outputs under `results/` |
|---|---|---|
| `plot_domestic_cases.py` | Four domestic scenario curves, knees from three methods, and shifts in knee locations | `Figure1_storage_requirement_turning_points.svg`, `Figure1_caption.txt`, `domestic_flexibility_knees_three_methods.xlsx`, `domestic_flexibility_knee_method_differences.csv`, and panels in `clean/domestic/` |
| `plot_global_cases.py` | Three global scenario curves and maximum-curvature knees | `global_comparison.svg`, `global_flexibility_knees_max_curvature.csv`, and panels in `clean/global/` |
| `knee_cap_sensitivity_analysis.py` | Recalculate all three methods with 150-, 200-, and 250-hour caps; summarize results and call the deviation plotting script | `cap_sensitivity_all_results.csv`, `cap_sensitivity_statistics.csv`, and SVGs for individual caps |
| `knee_cap_deviation_plot.py` | Read the cap summary, calculate knee shifts relative to 200 hours, and compare method robustness | `cap_sensitivity_deviation_from_200h.csv`, `cap_sensitivity_method_robustness.csv`, and `cap_sensitivity_three_methods*.svg` |
| `knee_method_comparison.py` | Compare each method's deviation from the three-method mean using the shared English fields | `knee_method_deviations.svg`, `knee_method_deviations_plot_data.csv`, and `clean/knee_method_deviations.svg` |

The domestic knee workbook contains a wide `method_comparison` sheet and a `long_format` sheet. The wide sheet's `scenario`, `consumption_target_%`, and `maximum_curvature_penetration_%` fields are direct inputs to the domestic economic figure.

The duration-cap sensitivity script processes all three caps by default. To generate one cap separately, run:

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/Flexibility Turning Point/knee_cap_sensitivity_analysis.py" --y-cap 200
```

With `--y-cap`, the `--data-only` option exports temporary numerical results for that cap and skips its individual plot. Run the entry point without arguments for the complete comparison across caps: the default workflow creates the individual figures and automatically calls the deviation plotting script.

## 7. Economic breakeven and threshold comparisons

### 7.1 Economic analysis

Directory: `Simulation_of_Multiple_Scenarios/Plot_data_and_code/Economic Breakeven Point/`.

Domestic inputs are in `data/sensitivity/{rural,commercial,industrial,island}/`; global inputs are in `data/global/{Norway,USA,Zambia}/`. Each scenario reads both `battery.xlsx` and `hydrogen.xlsx`.

The code reads the first worksheet and requires `cons_target_%`, `energy_pen_%`, `min_offgrid_hours`, and `LCOE_USD_per_kWh`. In addition, **the first column must contain the cost factor**: `bat_price_factor` for the current battery tables and `storage_factor` for the battery-hydrogen tables. Filtering uses the column position, so preserve this first-column layout.

The analysis proceeds as follows:

1. Pair battery and battery-hydrogen results at the same utilization target and cost factor.
2. Linearly interpolate battery-hydrogen LCOE onto the battery pathway's penetration points. Outside the battery-hydrogen data range, use the endpoint values.
3. Calculate `ΔLCOE = LCOE_BESS − LCOE_H2`. Positive values favor the battery-hydrogen pathway; negative values favor the battery pathway.
4. Find points where the cost difference is zero or changes sign, and interpolate the economic breakeven point.

This module uses previously calculated LCOE values and does not optimize capacities again. It matches the pathways by penetration and uses the battery pathway's off-grid duration as the duration coordinate. The comparison therefore does not impose an exact match in both penetration and off-grid duration.

| Script | Settings and purpose | Main outputs |
|---|---|---|
| `plot_domestic_cases.py` | Four domestic scenarios at factor 1.0; combine economic breakeven points with maximum-curvature knees and calculate their penetration differences | `results/Figure2_storage_pathway_economics.svg`, `_source_data.csv` and `_caption.txt` with the same prefix, and panels in `clean/domestic/` |
| `plot_global_cases.py` | Norway, USA, and Zambia at factor 1.0 | `results/global_comparison.svg` and panels in `clean/global/` |
| `plot_cost_factor_sensitivity.py` | Four domestic scenarios at factors 1.0, 0.9, 0.7, and 0.5 | `results/sensitivity/<scenario>.svg` and `clean/sensitivity/<scenario>/factor_*.svg` |

The domestic combined figure uses an off-grid duration cap of 500 hours; the global and cost sensitivity figures use 200 hours. The domestic figure requires the knee workbook from Section 6 and reports `Δp = p_economic − p_physical` in percentage points. Cases without a breakeven point retain a missing value and a corresponding status.

### 7.2 Compare the three knee methods with economic breakeven

Two scripts in the `Plot_data_and_code/` root compare threshold order and differences for each domestic scenario and utilization target:

| Script | Purpose |
|---|---|
| `compare_knee_breakeven_order.py` | Pair each knee method with economic breakeven points and export detailed comparisons, grouped summaries, and text statistics |
| `knee_method_breakeven_comparison.py` | Plot each knee's deviation from economic breakeven and export plotting data and a caption |

Both write to `Plot_data_and_code/results/knee_breakeven_order/` and use the shared English knee fields and scenario values.

Generate the domestic flexibility CSV first, then run the commands below. Economic breakeven points are recalculated directly from the economic input workbooks, so the economic figure does not need to be generated first:

```powershell
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/compare_knee_breakeven_order.py"
python "Simulation_of_Multiple_Scenarios/Plot_data_and_code/knee_method_breakeven_comparison.py"
```

- The statistics script exports `comparison_details.csv`, `summary_by_method.csv`, `summary_by_scenario_method.csv`, and `summary.txt`. Use `--tolerance-pp` to change the coincidence tolerance (default: 0.0001 percentage points) and `--output-dir` to change the output directory.
- The comparison figure runs independently of the statistics script. It exports `knee_method_breakeven_deviations.{svg,pdf,png}`, `_plot_data.csv` and `_caption.txt` with the same prefix, and SVGs in `clean/`.
- The statistics use `p_economic − p_flexibility`, whereas the figure's horizontal axis uses `p_flexibility − p_economic`; their signs are opposite.
- Cases without an economic breakeven point are reported separately and excluded from the denominator when calculating ordering proportions. The comparison figure positions these records relative to a 100% reference; their actual breakeven values remain missing. Gray spans show the range across the three methods.

## 8. Mapping simulation results to plotting inputs

Simulation result paths below are relative to `Simulation_data_and_code/<Scenario>/Result/`; plotting paths are relative to `Plot_data_and_code/`. The table identifies the archived workbook associated with each plotting copy.

| Scenario | Result file | Plotting input or purpose |
|---|---|---|
| Rural, Commercial, Industry | `sensitivity_bat_sizing_merged_greenfield_results.xlsx` | `Flexibility Turning Point/data/domestic/<scenario>/battery.xlsx` |
| Island | `sensitivity_wind_pv_bat_sizing_greenfield_results.xlsx` | Domestic island flexibility `battery.xlsx` |
| Rural, Commercial, Industry | `sensitivity_bat_sizing_merged_results.xlsx` | `Economic Breakeven Point/data/sensitivity/<scenario>/battery.xlsx` |
| Island | `sensitivity_wind_pv_bat_sizing_results.xlsx` | Domestic island economic analysis `battery.xlsx` |
| Four domestic scenarios | `sensitivity_*h2_sizing_joint_results.xlsx` | Corresponding economic analysis `hydrogen.xlsx`; the Island filename includes `wind_pv` |
| Norway | `sensitivity_wind_pv_bat_sizing_merged_results.xlsx`, `sensitivity_wind_pv_h2_sizing_joint_results.xlsx` | `data/global/Norway/{battery,hydrogen}.xlsx` in both plotting modules |
| USA | `sensitivity_bat_sizing_merged_results.xlsx`, `sensitivity_h2_sizing_joint_results.xlsx` | `data/global/USA/{battery,hydrogen}.xlsx` in both plotting modules |
| Zambia | `sensitivity_bat_sizing_merged_results.xlsx` | `data/global/Zambia/battery_backup.xlsx` in both plotting modules |
| Zambia | `sensitivity_h2_sizing_joint_results.xlsx` | `data/global/Zambia/hydrogen.xlsx` in both plotting modules |

Scenario names map as follows: `Rural → rural`, `Commercial → commercial`, `Industry → industrial`, and `Island → island`. Global paths retain the capitalization of the country directories.

**Zambia exception:** the `battery.xlsx` files currently used for plotting differ from the battery workbook archived in `Result/`. The matching copies are named `battery_backup.xlsx`. When updating Zambia results, first identify which input version the analysis should reproduce.

## 9. Reproduction notes

1. **Simulation input and archive paths differ.** The archive uses `Data/` and `Result/`, while the scripts default to `Code/`. Prepare the inputs as described in Section 5.3.
2. **Field names are standardized in English.** Scripts, input tables, plotting data, and historical statistics use English fields. Preserve the documented field names, scenario values, and required column order when replacing data. The method comparison reads the English output of `plot_domestic_cases.py` directly.
3. **Some outputs are historical artifacts.** Neither module's `results/domestic_comparison.svg`, the flexibility module's `global_flexibility_knees_kneedle.csv`, nor the economic module's `economic_crossover_vs_flexibility_knee_x_comparison.csv` is regenerated by its current main plotting entry point. Current main figures use the `Figure1_*`, `Figure2_*`, and other filenames saved by the scripts.
4. **Archived results and plotting copies do not update together.** After optimization, organize the results, add the off-grid fields needed for economic analysis, and update the plotting inputs.
5. **Most settings are source constants.** Plot limits, capacity scan ranges, initial SOC, equipment efficiencies, cost factors, and knee-detection settings affect the results. Record both the input workbooks and source configuration when comparing versions.

Validation of the English version confirmed that all 69 Python files parse successfully. All numeric values and worksheet structures in the 61 Excel workbooks, and all numeric fields in the 15 CSV files, match the pre-translation data. Text checks include decoded workbook metadata, SVG labels, and the existing PDF; the three existing PNG figures were also inspected.

All 13 plotting scripts completed in an isolated test copy: 12 direct entry points and the deviation plot called by the cap-sensitivity workflow. The tests used Python 3.12.14, NumPy 2.5.3, pandas 3.0.6, Matplotlib 3.11.2, SciPy 1.18.1, and openpyxl 3.1.5. Full Gurobi capacity optimization and the MATLAB simulation were not rerun.
