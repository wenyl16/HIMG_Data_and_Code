"""Sensitivity of three knee methods to the storage-duration analysis cap.

With no command-line argument, the script independently evaluates 150, 200,
and 250 h caps using the shared PCHIP, Kneedle, maximum-curvature, and L-method
workflow.
"""
import sys, io, os, argparse, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'domestic')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
CLEAN_ROOT = os.path.join(RESULT_DIR, 'clean', 'cap_sensitivity')
TEMP_DIR = os.path.join(RESULT_DIR, '.tmp_limits')
os.makedirs(CLEAN_ROOT, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

parser = argparse.ArgumentParser(
    description='Test knee-point sensitivity to the storage-duration cap.')
parser.add_argument(
    '--y-cap', type=float, default=None,
    help='Run one analysis cap; omit to run 150, 200 and 250 h.')
parser.add_argument(
    '--data-only', action='store_true',
    help='Child-process mode: export temporary numeric results only.')
ARGS = parser.parse_args()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

# ═══ Nature-style rcParams ═══
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Microsoft YaHei', 'SimHei', 'Arial',
                           'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype':       'none',
    'pdf.fonttype':       42,
    'font.size':          7,
    'axes.spines.right':  False,
    'axes.spines.top':    False,
    'axes.linewidth':     0.8,
    'legend.frameon':     False,
    'axes.unicode_minus': False,
})

# ═══ Configuration ═══
SCENARIOS = [
    ('Rural', {
        'path': os.path.join(DATA_DIR, 'rural', 'battery.xlsx'),
        'divisor': 0.1526255708, 'x_cap': None,
    }),
    ('Commercial', {
        'path': os.path.join(DATA_DIR, 'commercial', 'battery.xlsx'),
        'divisor': 1.33869863, 'x_cap': None,
    }),
    ('Industrial', {
        'path': os.path.join(DATA_DIR, 'industrial', 'battery.xlsx'),
        'divisor': 0.7547945205, 'x_cap': None,
    }),
    ('Island', {
        'path': os.path.join(DATA_DIR, 'island', 'battery.xlsx'),
        'divisor': 0.7324200913, 'x_cap': 100.0,
    }),
]
SCENARIO_SLUGS = {
    'Rural': 'rural',
    'Commercial': 'commercial',
    'Industrial': 'industrial',
    'Island': 'island',
}

COLOR_MAP = {
    60:  '#2166AC',
    70:  '#4393C3',
    80:  '#7FCDBB',
    90:  '#41AB5D',
    100: '#006D2C',
}

PANEL_LABELS = ['a', 'b', 'c', 'd']
Y_CAPS = (150, 200, 250)
Y_CAP = int(ARGS.y_cap) if ARGS.y_cap is not None else None
KNEE_METHODS = [
    'maximum_curvature',
    'kneedle',
    'L_method',
]
KNEE_METHOD_STYLES = {
    'kneedle': {'marker': '^', 'markersize': 4.5},
    'maximum_curvature': {'marker': 'o', 'markersize': 4.2},
    'L_method': {'marker': 's', 'markersize': 4.0},
}
KNEE_RESULTS_CSV = (
    os.path.join(TEMP_DIR, f'cap_{Y_CAP}.csv')
    if Y_CAP is not None else None
)

# The sensitivity cases use the original PCHIP interpolation and discard the
# leading curve section whose storage duration is below 1 hour.
MIN_STORAGE_DURATION_H = 1.0
MIN_METHOD_POINTS = 7
# A compact derivative window keeps the detected peak aligned with the
# visible bend of the original PCHIP curve.
COMMON_SAVGOL_WINDOW = 101
COMMON_SAVGOL_POLYORDER = 3
REFERENCE_BAT_PRICE_FACTOR = 1.0


# ═══ Data helpers ═══
def normalize_percent(series):
    s = series.astype(str).str.replace('%', '', regex=False).str.strip()
    s = pd.to_numeric(s, errors='coerce')
    if s.max() <= 1.5:
        s = s * 100
    return s


def get_smooth_xy(x, y, num=300):
    """Interpolate every displayed curve with shape-preserving PCHIP."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        y_sum = np.bincount(inverse, weights=y)
        y_count = np.bincount(inverse)
        x, y = unique_x, y_sum / y_count
    xs = np.linspace(x[0], x[-1], num)
    ys = PchipInterpolator(x, y, extrapolate=False)(xs)
    return xs, ys


def insert_y_cap_endpoint(x, y, y_cap):
    """Insert the first exact raw-data crossing of ``y_cap``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    above = np.flatnonzero(y > y_cap)
    if not len(above) or above[0] == 0:
        return x, y
    idx = int(above[0])
    x1, y1, x2, y2 = x[idx - 1], y[idx - 1], x[idx], y[idx]
    xc = x1 if y2 == y1 else x1 + (y_cap - y1) * (x2 - x1) / (y2 - y1)
    return np.concatenate([x[:idx], [xc]]), np.concatenate([y[:idx], [y_cap]])


def clip_curve_by_y_cap(x, y, y_cap):
    if len(x) == 0:
        return x, y
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = y <= y_cap
    if keep.all():
        return x, y
    if not keep[0]:
        return np.array([]), np.array([])
    idx = np.argmax(~keep)
    x1, y1, x2, y2 = x[idx-1], y[idx-1], x[idx], y[idx]
    xc = x1 if y2 == y1 else x1 + (y_cap - y1) * (x2 - x1) / (y2 - y1)
    return np.concatenate([x[:idx], [xc]]), np.concatenate([y[:idx], [y_cap]])


def clip_at_x(x, y, x_max):
    if len(x) == 0:
        return x, y
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x[-1] <= x_max:
        return x, y
    keep = x <= x_max
    if not keep[0]:
        return np.array([]), np.array([])
    if keep.all():
        return x, y
    idx = np.argmax(~keep)
    yi = y[idx-1] + (y[idx]-y[idx-1]) * (x_max-x[idx-1]) / (x[idx]-x[idx-1])
    return np.concatenate([x[:idx], [x_max]]), np.concatenate([y[:idx], [yi]])


def find_growth_start(x_curve, y_curve):
    """Return the first curve point with storage duration >= 1 hour."""
    x = np.asarray(x_curve, float)
    y = np.asarray(y_curve, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return None
    eligible = np.flatnonzero(y >= MIN_STORAGE_DURATION_H)
    return float(x[eligible[0]]) if len(eligible) else None


def prepare_method_interval(x, y, growth_start_x):
    """Filter the leading low-duration section and normalize to tested axes."""
    if growth_start_x is None:
        return None
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= growth_start_x)
    x, y = x[valid], y[valid]
    if len(x) < MIN_METHOD_POINTS or x[-1] <= x[0]:
        return None
    x_span = x[-1] - x[0]
    if x_span <= 0 or Y_CAP <= 0:
        return None
    xn = (x - x[0]) / x_span
    yn = y / Y_CAP
    return x, y, xn, yn


def apply_common_savgol(xn, yn, deriv=0):
    """Apply the same Savitzky-Golay model to every knee-point method."""
    from scipy.signal import savgol_filter

    window = min(
        COMMON_SAVGOL_WINDOW,
        len(yn) if len(yn) % 2 else len(yn) - 1,
    )
    min_window = COMMON_SAVGOL_POLYORDER + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(min_window, window)
    delta = float(np.mean(np.diff(xn)))
    return savgol_filter(
        yn,
        window_length=window,
        polyorder=COMMON_SAVGOL_POLYORDER,
        deriv=deriv,
        delta=delta,
        mode='interp',
    )


def find_kneedle(x, y, growth_start_x):
    """Kneedle on the common Savitzky-Golay curve."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    x, y, xn, yn = prepared
    yn_smooth = apply_common_savgol(xn, yn)
    distance = xn - yn_smooth
    # Exclude endpoints: they cannot be meaningful interior knees.
    idx = int(np.argmax(distance[1:-1]) + 1)
    return float(x[idx]), float(y[idx])


def curvature_search_bounds(x, y, growth_start_x):
    """Return the common post-1-hour interval used by maximum curvature."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    x = prepared[0]
    return float(x[0]), float(x[-1])


def find_max_curvature(x, y, growth_start_x):
    """Find maximum curvature using the common Savitzky-Golay derivatives."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    x, y, xn, yn = prepared
    # Derivative orders 0, 1 and 2 share exactly the same local polynomial,
    # window length and normalized PCHIP input for all three methods.
    first = apply_common_savgol(xn, yn, deriv=1)
    second = apply_common_savgol(xn, yn, deriv=2)

    # Discard only the numerical derivative endpoints of the already-truncated
    # interval; no result from kneedle or L_method enters this calculation.
    edge = 2
    if len(xn) <= 2 * edge + 1:
        return None, None
    curvature = np.abs(second) / np.power(1.0 + first ** 2, 1.5)
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    if not np.isfinite(curvature).any():
        return None, None
    idx = int(np.nanargmax(curvature))
    return float(x[idx]), float(y[idx])


def find_first_prominent_curvature_peak(
        x, y, growth_start_x, return_info=False, savgol_window=None):
    """Return the leftmost prominent curvature peak in the growth region."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    empty_info = {
        'prominent_curvature_peak_count': 0,
        'eligible_prominent_curvature_peak_count': 0,
        'curvature_peak_min_duration_h': np.nan,
        'used_maximum_curvature_fallback': False,
    }
    if prepared is None:
        return (None, None, empty_info) if return_info else (None, None)

    x, y, xn, yn = prepared
    if savgol_window is None:
        savgol_window = COMMON_SAVGOL_WINDOW
    first = apply_common_savgol(
        xn, yn, deriv=1, window_limit=savgol_window)
    second = apply_common_savgol(
        xn, yn, deriv=2, window_limit=savgol_window)
    curvature = np.abs(second) / np.power(1.0 + first ** 2, 1.5)
    # Savitzky-Golay derivatives are unreliable within half a window of a
    # truncated endpoint. Excluding that full support prevents cap-edge peaks
    # from suppressing earlier genuine curvature peaks.
    edge = max(
        savgol_window // 2 + 1,
        int(round(0.02 * len(x))),
    )
    if len(x) <= 2 * edge + 1:
        return (None, None, empty_info) if return_info else (None, None)
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    finite = np.isfinite(curvature)
    if not finite.any():
        return (None, None, empty_info) if return_info else (None, None)

    max_curvature = float(np.nanmax(curvature))
    peak_input = np.where(finite, curvature, -np.inf)
    min_distance = max(
        1, int(round(FIRST_PEAK_MIN_DISTANCE_FRACTION * len(x))))
    min_peak_height = np.nextafter(
        FIRST_PEAK_MIN_HEIGHT_RATIO * max_curvature, np.inf)
    peaks, _ = find_peaks(
        peak_input,
        height=min_peak_height,
        distance=min_distance,
    )
    min_peak_y = max(
        FIRST_PEAK_MIN_STORAGE_DURATION_H,
        float(y[0] + 0.03 * (y[-1] - y[0])),
    )
    # Judge growth on the original PCHIP curve over a local interval instead
    # of using the derivative at one Savitzky-Golay sample. A long SG window
    # can make that point derivative negative close to a steep terminal rise
    # even when the underlying curve is monotonically increasing.
    growth_half_width = max(2, savgol_window // 4)

    def locally_rising(indices):
        flags = []
        for peak_idx in np.asarray(indices, dtype=int):
            left = max(0, peak_idx - growth_half_width)
            right = min(len(y) - 1, peak_idx + growth_half_width)
            steps = np.diff(y[left:right + 1])
            flags.append(
                y[right] > y[left]
                and len(steps) > 0
                and np.mean(steps > 0.0) >= 0.75
            )
        return np.asarray(flags, dtype=bool)

    valid_peaks = peaks[
        (y[peaks] >= min_peak_y) & locally_rising(peaks)]
    used_fallback = False
    if len(valid_peaks):
        idx = int(valid_peaks[0])
    else:
        candidates = np.flatnonzero(finite & (y >= min_peak_y))
        eligible = candidates[locally_rising(candidates)]
        if not len(eligible):
            info = dict(empty_info)
            info['prominent_curvature_peak_count'] = int(len(peaks))
            info['curvature_peak_min_duration_h'] = min_peak_y
            return (None, None, info) if return_info else (None, None)
        idx = int(eligible[np.argmax(curvature[eligible])])
        used_fallback = True

    info = {
        'prominent_curvature_peak_count': int(len(peaks)),
        'eligible_prominent_curvature_peak_count': int(len(valid_peaks)),
        'curvature_peak_min_duration_h': min_peak_y,
        'used_maximum_curvature_fallback': used_fallback,
    }
    point = (float(x[idx]), float(y[idx]))
    return (*point, info) if return_info else point


def _line_sse(x, y):
    """Residual sum of squares for a least-squares straight line."""
    design = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(residual @ residual)


def find_l_method(x, y, growth_start_x):
    """Two-line L-method on the common Savitzky-Golay curve."""
    prepared = prepare_method_interval(x, y, growth_start_x)
    if prepared is None:
        return None, None
    x, y, xn, yn = prepared
    yn = apply_common_savgol(xn, yn)
    min_segment = max(3, int(round(0.03 * len(xn))))
    candidates = range(min_segment - 1, len(xn) - min_segment)
    best_idx, best_score = None, np.inf
    for idx in candidates:
        left_n = idx + 1
        right_n = len(xn) - idx
        left_sse = _line_sse(xn[:idx + 1], yn[:idx + 1])
        right_sse = _line_sse(xn[idx:], yn[idx:])
        # Weighted RMSE is the standard L-method objective.
        score = (
            left_n * np.sqrt(left_sse / left_n)
            + right_n * np.sqrt(right_sse / right_n)
        ) / (left_n + right_n)
        if score < best_score:
            best_idx, best_score = idx, score
    if best_idx is None:
        return None, None
    return float(x[best_idx]), float(y[best_idx])


def run_all_caps():
    """Run three independent cap cases and aggregate their sensitivity."""
    script_path = os.path.abspath(__file__)
    case_frames = []
    case_files = []
    for y_cap in Y_CAPS:
        print(f'\n{"=" * 18} Running storage-duration cap {y_cap} h {"=" * 18}')
        subprocess.run(
            [sys.executable, '-X', 'utf8', script_path,
             '--y-cap', str(y_cap)],
            cwd=SCRIPT_DIR,
            check=True,
        )
        case_file = os.path.join(TEMP_DIR, f'cap_{y_cap}.csv')
        case_frames.append(pd.read_csv(case_file, encoding='utf-8-sig'))
        case_files.append(case_file)

    all_results = pd.concat(case_frames, ignore_index=True)

    sensitivity_rows = []
    for _, row in all_results.iterrows():
        for method in KNEE_METHODS:
            sensitivity_rows.append({
                'duration_cap_h': row['duration_cap_h'],
                'scenario': row['scenario'],
                'consumption_target_%': row['consumption_target_%'],
                'method': method,
                'turning_point_penetration_%': row[f'{method}_penetration_%'],
                'turning_point_duration_h': row[f'{method}_duration_h'],
            })
    sensitivity_long = pd.DataFrame(sensitivity_rows)
    point_pivot = sensitivity_long.pivot(
        index=['scenario', 'consumption_target_%', 'method'],
        columns='duration_cap_h',
        values='turning_point_penetration_%',
    ).reset_index()
    point_pivot.columns.name = None
    point_pivot = point_pivot.rename(
        columns={cap: f'cap_{cap}h_penetration_%' for cap in Y_CAPS})
    cap_columns = [f'cap_{cap}h_penetration_%' for cap in Y_CAPS]
    point_pivot['max_shift_pp'] = (
        point_pivot[cap_columns].max(axis=1)
        - point_pivot[cap_columns].min(axis=1)
    )
    cap_1, cap_2, cap_3 = Y_CAPS
    point_pivot[f'cap_{cap_1}_to_{cap_2}h_change_pp'] = (
        point_pivot[f'cap_{cap_2}h_penetration_%']
        - point_pivot[f'cap_{cap_1}h_penetration_%']
    )
    point_pivot[f'cap_{cap_2}_to_{cap_3}h_change_pp'] = (
        point_pivot[f'cap_{cap_3}h_penetration_%']
        - point_pivot[f'cap_{cap_2}h_penetration_%']
    )
    detailed_output = os.path.join(
        RESULT_DIR, 'cap_sensitivity_all_results.csv')
    summary_table_output = os.path.join(
        RESULT_DIR, 'cap_sensitivity_statistics.csv')
    all_results.to_csv(
        detailed_output, index=False, encoding='utf-8-sig',
        float_format='%.4f')
    point_pivot.to_csv(
        summary_table_output, index=False, encoding='utf-8-sig',
        float_format='%.4f')

    method_styles = {
        'kneedle': {'marker': '^', 'linestyle': '-'},
        'maximum_curvature': {'marker': 'o', 'linestyle': '--'},
        'L_method': {'marker': 's', 'linestyle': ':'},
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for ax, scenario in zip(axes.flat, [item[0] for item in SCENARIOS]):
        scenario_data = sensitivity_long[
            sensitivity_long['scenario'].eq(scenario)]
        for target in sorted(scenario_data['consumption_target_%'].unique()):
            for method in KNEE_METHODS:
                line_data = scenario_data[
                    scenario_data['consumption_target_%'].eq(target)
                    & scenario_data['method'].eq(method)
                ].sort_values('duration_cap_h')
                style = method_styles[method]
                ax.plot(
                    line_data['duration_cap_h'],
                    line_data['turning_point_penetration_%'],
                    color=COLOR_MAP[int(target)],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=3.3,
                    markeredgecolor='black',
                    markeredgewidth=0.45,
                    linewidth=0.9,
                )
        ax.set_title(scenario, fontsize=8, fontweight='bold')
        ax.set_xticks(Y_CAPS)
        ax.set_xlim(140, 260)
        ax.set_ylim(40, 100)
        ax.tick_params(labelsize=6, length=2.5, width=0.6)
    axes[1, 0].set_xlabel('Y-axis upper limit (h)', fontsize=7)
    axes[1, 1].set_xlabel('Y-axis upper limit (h)', fontsize=7)
    axes[0, 0].set_ylabel('Detected penetration rate (%)', fontsize=7)
    axes[1, 0].set_ylabel('Detected penetration rate (%)', fontsize=7)
    target_handles = [
        Line2D([0], [0], color=COLOR_MAP[target], lw=1.4,
               label=f'Self-consumption {target}%')
        for target in sorted(COLOR_MAP)
    ]
    method_handles = [
        Line2D(
            [0], [0], color='black',
            linestyle=method_styles[method]['linestyle'],
            marker=method_styles[method]['marker'],
            markersize=3.3,
            label=method,
        )
        for method in KNEE_METHODS
    ]
    fig.legend(
        handles=target_handles, loc='lower center', ncol=5,
        fontsize=6, bbox_to_anchor=(0.5, 0.045),
        handletextpad=0.4, columnspacing=1.0,
    )
    fig.legend(
        handles=method_handles, loc='lower center', ncol=3,
        fontsize=6, bbox_to_anchor=(0.5, 0.005),
        handletextpad=0.5, columnspacing=1.4,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.16, top=0.94,
        hspace=0.30, wspace=0.25,
    )
    summary_output = os.path.join(
        RESULT_DIR, 'cap_sensitivity_three_methods.svg')
    fig.savefig(summary_output, bbox_inches='tight')
    plt.close(fig)

    for case_file in case_files:
        os.remove(case_file)
    if os.path.isdir(TEMP_DIR) and not os.listdir(TEMP_DIR):
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            # OneDrive may briefly lock an empty directory after child runs.
            pass

    print(f'\nSaved: {summary_output}')
    print(f'Saved: {detailed_output}')
    print(f'Saved: {summary_table_output}')
    deviation_script = os.path.join(
        SCRIPT_DIR, 'knee_cap_deviation_plot.py')
    subprocess.run(
        [sys.executable, '-X', 'utf8', deviation_script],
        cwd=SCRIPT_DIR,
        check=True,
    )
    print('\nAll turning-point shifts (largest shift first):')
    print(point_pivot.sort_values(
        'max_shift_pp', ascending=False).to_string(
        index=False, float_format=lambda value: f'{value:.3f}'))


if ARGS.y_cap is None:
    run_all_caps()
    sys.exit(0)


# ═══ Step 1: Process all scenario data ═══
all_panel_data = []
knee_long_rows = []
knee_wide_rows = []

for scenario, cfg in SCENARIOS:
    print(f'\nProcessing: {scenario}')
    divisor = cfg['divisor']
    x_cap   = cfg.get('x_cap')

    if not os.path.isfile(cfg['path']):
        print(f'  ⚠ Input file missing; skipped: {cfg["path"]}')
        continue
    df = pd.read_excel(cfg['path'], sheet_name=0)
    df.columns = df.columns.astype(str).str.strip()

    # Some supplemented workbooks contain several price-factor sweeps on
    # different penetration grids.  Mixing those grids creates artificial
    # saw-tooth curves even under PCHIP, so consistently use the 1.0 baseline.
    if 'bat_price_factor' in df.columns:
        price_factor = pd.to_numeric(df['bat_price_factor'], errors='coerce')
        baseline_mask = np.isclose(
            price_factor, REFERENCE_BAT_PRICE_FACTOR, rtol=0.0, atol=1e-9)
        if not baseline_mask.any():
            print(
                f'  ⚠ No bat_price_factor={REFERENCE_BAT_PRICE_FACTOR:g} '
                'baseline data; skipped')
            continue
        df = df.loc[baseline_mask].copy()

    # Support both the original files and the supplemented result schema.
    # Prefer an explicitly reported storage duration; otherwise derive it from
    # the original total battery energy and scenario-specific reference power.
    if 'storage_duration_h' in df.columns:
        duration_source = 'storage_duration_h'
        duration_divisor = 1.0
    elif 'total_batt_MWh' in df.columns:
        duration_source = 'total_batt_MWh'
        duration_divisor = divisor
    else:
        print(f'  ⚠ Storage-duration/capacity field missing; skipped: {cfg["path"]}')
        continue

    df = df[['cons_target_%', 'energy_pen_%', duration_source]].copy()
    df['cons_target_%']  = normalize_percent(df['cons_target_%'])
    df['energy_pen_%']   = normalize_percent(df['energy_pen_%'])
    df['storage_duration_h'] = (
        pd.to_numeric(df[duration_source], errors='coerce') / duration_divisor
    )
    df = df.dropna(
        subset=['cons_target_%', 'energy_pen_%', 'storage_duration_h'])

    plot_df = (
        df.groupby(['cons_target_%', 'energy_pen_%'], as_index=False)['storage_duration_h']
          .mean().sort_values(['cons_target_%', 'energy_pen_%'])
    )

    targets = sorted(t for t in plot_df['cons_target_%'].unique() if round(t) != 50)

    curve_data, method_knee_data, raw_data, growth_start_data = {}, {}, {}, {}
    for target in targets:
        sub = plot_df[plot_df['cons_target_%'] == target].sort_values('energy_pen_%')
        x_raw = sub['energy_pen_%'].to_numpy(float)
        y_raw = sub['storage_duration_h'].to_numpy(float)
        if len(x_raw) == 0:
            continue
        t_int = int(round(target))
        color = COLOR_MAP.get(t_int, '#767676')

        if scenario == 'Island' and t_int == 60:
            x_for_curve, y_for_curve = insert_y_cap_endpoint(
                x_raw, y_raw, Y_CAP)
            xs, ys = get_smooth_xy(x_for_curve, y_for_curve)
        else:
            xs, ys = get_smooth_xy(x_raw, y_raw)
        if x_cap is not None:
            xs, ys = clip_at_x(xs, ys, x_cap)
        x_line, y_line = clip_curve_by_y_cap(xs, ys, Y_CAP)
        if len(x_line) == 0:
            continue

        growth_start_x = find_growth_start(x_line, y_line)
        if growth_start_x is None:
            print(f'  ⚠ {t_int}% curve does not reach {MIN_STORAGE_DURATION_H:g} h; skipped')
            continue
        curve_data[t_int] = [x_line, y_line]
        raw_data[t_int]   = (x_raw, y_raw, color)
        growth_start_data[t_int] = growth_start_x

    sorted_t = sorted(curve_data.keys())
    for i in range(len(sorted_t) - 2, -1, -1):
        tc, tn = sorted_t[i], sorted_t[i+1]
        xc, yc = curve_data[tc]
        xn, yn = curve_data[tn]
        if len(yc) > 0 and len(yn) > 0 and yc[-1] > yn[-1]:
            curve_data[tc] = list(clip_curve_by_y_cap(xc, yc, yn[-1]))

    # Detect all three methods on the same final, displayed PCHIP curve.
    for t_int in sorted_t:
        x_line, y_line = curve_data[t_int]
        growth_start_x = growth_start_data[t_int]
        curvature_bounds = curvature_search_bounds(
            x_line, y_line, growth_start_x)
        knees = {
            'kneedle': find_kneedle(x_line, y_line, growth_start_x),
            'maximum_curvature': find_max_curvature(x_line, y_line, growth_start_x),
            'L_method': find_l_method(x_line, y_line, growth_start_x),
        }
        method_knee_data[t_int] = knees

        wide_row = {
            'duration_cap_h': Y_CAP,
            'scenario': scenario,
            'consumption_target_%': t_int,
            'growth_start_penetration_%': growth_start_x,
            'curvature_search_lower_bound_%': curvature_bounds[0],
            'curvature_search_upper_bound_%': curvature_bounds[1],
        }
        for method in KNEE_METHODS:
            method_x, method_y = knees[method]
            knee_long_rows.append({
                'scenario': scenario,
                'consumption_target_%': t_int,
                'growth_start_penetration_%': growth_start_x,
                'curvature_search_lower_bound_%': curvature_bounds[0],
                'curvature_search_upper_bound_%': curvature_bounds[1],
                'method': method,
                'turning_point_penetration_%': method_x,
                'turning_point_duration_h': method_y,
            })
            wide_row[f'{method}_penetration_%'] = method_x
            wide_row[f'{method}_duration_h'] = method_y
        knee_wide_rows.append(wide_row)

    all_panel_data.append({
        'scenario':    scenario,
        'x_cap':       x_cap,
        'curve_data':  curve_data,
        'method_knee_data': method_knee_data,
        'growth_start_data': growth_start_data,
        'raw_data':    raw_data,
        'sorted_t':    sorted_t,
    })


# Export the three-method results for this cap.
knee_results_long = pd.DataFrame(knee_long_rows)
knee_results_wide = pd.DataFrame(knee_wide_rows)
if not knee_results_long.empty:
    knee_results_wide.to_csv(
        KNEE_RESULTS_CSV,
        index=False,
        encoding='utf-8-sig',
        float_format='%.4f',
    )
    print('\nThree-method turning-point results (temporary aggregation input):')
    print(knee_results_wide.to_string(
        index=False, float_format=lambda value: f'{value:.3f}'))

if ARGS.data_only:
    print('\nNumeric export completed')
    sys.exit(0)


# ═══ Step 2: Draw figure (labeled or clean) ═══
def draw_figure(labeled=True):
    """Create the 2×2 figure. labeled=True adds all text annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes_flat = axes.flatten()

    for pidx, panel in enumerate(all_panel_data):
        ax = axes_flat[pidx]
        curve_data = panel['curve_data']
        method_knee_data = panel['method_knee_data']
        raw_data   = panel['raw_data']
        sorted_t   = panel['sorted_t']
        x_cap      = panel['x_cap']

        for t_int in sorted_t:
            x_line, y_line = curve_data[t_int]
            method_knees   = method_knee_data[t_int]
            x_raw, y_raw, color = raw_data[t_int]

            # Scatter
            mask = y_raw <= Y_CAP
            if x_cap is not None:
                mask &= x_raw <= x_cap
            ax.scatter(x_raw[mask], y_raw[mask],
                       s=3, color=color, alpha=0.25, zorder=3, linewidths=0)

            # Curve
            ax.plot(x_line, y_line, color=color, linewidth=1.1, zorder=4)

            if labeled:
                for method in KNEE_METHODS:
                    method_x, method_y = method_knees[method]
                    if method_x is None:
                        continue
                    style = KNEE_METHOD_STYLES[method]
                    ax.plot(
                        method_x, method_y,
                        linestyle='none',
                        marker=style['marker'],
                        markersize=style['markersize'],
                        markerfacecolor='white',
                        markeredgecolor=color,
                        markeredgewidth=0.9,
                        zorder=10,
                    )

        # ── Axis styling ──
        ax.set_xlim(0, 100)
        ax.set_ylim(0, Y_CAP)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(np.linspace(0, Y_CAP, 6))
        ax.tick_params(axis='both', which='major', length=2.5, width=0.6,
                       labelsize=6, direction='out', pad=2)

        if labeled:
            lines = ['Pen. (%): curvature / Kneedle / L_method']
            for t_int in sorted_t:
                values = [
                    method_knee_data[t_int][method][0]
                    for method in KNEE_METHODS
                ]
                if all(value is not None for value in values):
                    values_text = ' / '.join(f'{value:.1f}' for value in values)
                    lines.append(f'{t_int}%: {values_text}')
            if len(lines) > 1:
                ax.text(0.03, 0.97, '\n'.join(lines),
                        transform=ax.transAxes, fontsize=4.5,
                        fontfamily='monospace', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='#cccccc',
                                  alpha=0.85, linewidth=0.4))

            # Panel label (a, b, c, d)
            ax.text(-0.08, 1.03, PANEL_LABELS[pidx],
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color='black', ha='left', va='bottom')
            # Scenario title
            ax.set_title(panel['scenario'], fontsize=8, fontweight='bold',
                         pad=6, loc='center')
            # Axis labels (edge panels only)
            if pidx >= 2:
                ax.set_xlabel('Energy penetration rate (%)', fontsize=7)
            if pidx % 2 == 0:
                ax.set_ylabel('Storage duration (h)', fontsize=7)
        else:
            # Clean version: remove tick labels too
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Shared legend (only in labeled version)
    if labeled:
        target_handles = [
            Line2D([0], [0], color=COLOR_MAP[t], lw=1.4,
                   label=f'Self-consumption {t}%')
            for t in sorted(COLOR_MAP)
        ]
        method_handles = [
            Line2D(
                [0], [0],
                linestyle='none',
                marker=KNEE_METHOD_STYLES[method]['marker'],
                markersize=KNEE_METHOD_STYLES[method]['markersize'],
                markerfacecolor='white',
                markeredgecolor='black',
                markeredgewidth=0.9,
                label=method,
            )
            for method in KNEE_METHODS
        ]
        fig.legend(handles=target_handles, loc='lower center', ncol=5,
                   fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.035),
                   handletextpad=0.4, columnspacing=1.2)
        fig.legend(handles=method_handles, loc='lower center', ncol=1,
                   fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.005),
                   handletextpad=0.5, columnspacing=1.5)

    fig.subplots_adjust(left=0.09 if labeled else 0.03,
                        right=0.97, bottom=0.15 if labeled else 0.03,
                        top=0.93 if labeled else 0.97,
                        hspace=0.42 if labeled else 0.12,
                        wspace=0.32 if labeled else 0.10)
    return fig


def draw_clean_panel(panel):
    """Create one text-free case panel with all three method markers."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for target in panel['sorted_t']:
        x_line, y_line = panel['curve_data'][target]
        x_raw, y_raw, color = panel['raw_data'][target]
        mask = y_raw <= Y_CAP
        if panel['x_cap'] is not None:
            mask &= x_raw <= panel['x_cap']
        ax.scatter(
            x_raw[mask], y_raw[mask], s=3, color=color,
            alpha=0.25, zorder=3, linewidths=0,
        )
        ax.plot(x_line, y_line, color=color, linewidth=1.1, zorder=4)
        for method in KNEE_METHODS:
            method_x, method_y = panel['method_knee_data'][target][method]
            if method_x is None:
                continue
            style = KNEE_METHOD_STYLES[method]
            ax.plot(
                method_x, method_y, linestyle='none',
                marker=style['marker'], markersize=style['markersize'],
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=0.9, zorder=10,
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y_CAP)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.linspace(0, Y_CAP, 6))
    ax.tick_params(
        axis='both', which='major', length=2.5, width=0.6,
        direction='out', labelbottom=False, labelleft=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.99)
    return fig


# ═══ Step 3: Export both versions ═══
fig_labeled = draw_figure(labeled=True)
main_output = os.path.join(
        RESULT_DIR, f'cap_sensitivity_{Y_CAP}h_three_methods.svg')
fig_labeled.savefig(main_output, bbox_inches='tight')
plt.close(fig_labeled)
print(f'\nSaved: {main_output}')

clean_dir = os.path.join(CLEAN_ROOT, f'{Y_CAP}h_three_methods')
os.makedirs(clean_dir, exist_ok=True)
for panel in all_panel_data:
    fig_clean = draw_clean_panel(panel)
    panel_output = os.path.join(
        clean_dir, f"{SCENARIO_SLUGS[panel['scenario']]}.svg")
    fig_clean.savefig(panel_output, bbox_inches='tight')
    plt.close(fig_clean)
    print(f'Saved: {panel_output}')

print('\nCompleted')
