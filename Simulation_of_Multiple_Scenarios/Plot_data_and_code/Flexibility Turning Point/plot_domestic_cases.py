"""Domestic flexibility knees using the shared three-method workflow.

The displayed curve retains the original shape-preserving PCHIP interpolation.
Kneedle, maximum curvature, and the L-method are evaluated on the same
post-1-hour interval.  The publication figure displays maximum curvature.

Outputs:
  1. results/Figure1_storage_requirement_turning_points.svg
  2. results/Figure1_caption.txt
  3. results/clean/domestic/{industrial,commercial,rural,island,
     turning_point_migration}.svg
  4. results/domestic_flexibility_knees_three_methods.xlsx
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'domestic')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
CLEAN_DIR = os.path.join(RESULT_DIR, 'clean', 'domestic')
os.makedirs(CLEAN_DIR, exist_ok=True)

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
    ('Industrial', {
        'path': os.path.join(DATA_DIR, 'industrial', 'battery.xlsx'),
        'divisor': 0.7547945205, 'x_cap': None,
        'slug': 'industrial',
    }),
    ('Commercial', {
        'path': os.path.join(DATA_DIR, 'commercial', 'battery.xlsx'),
        'divisor': 1.33869863, 'x_cap': None,
        'slug': 'commercial',
    }),
    ('Rural', {
        'path': os.path.join(DATA_DIR, 'rural', 'battery.xlsx'),
        'divisor': 0.1526255708, 'x_cap': None,
        'slug': 'rural',
    }),
    ('Island', {
        'path': os.path.join(DATA_DIR, 'island', 'battery.xlsx'),
        'divisor': 0.7324200913, 'x_cap': 100.0,
        'slug': 'island',
    }),
]

COLOR_MAP = {
    60:  '#2166AC',
    70:  '#4393C3',
    80:  '#7FCDBB',
    90:  '#41AB5D',
    100: '#006D2C',
}

PANEL_LABELS = ['a', 'b', 'c', 'd', 'e']
Y_CAP = 200
CURVE_LINEWIDTH = 1.25
TURNING_POINT_MARKERSIZE = 4.4
TURNING_POINT_MARKEREDGEWIDTH = 0.75
SUMMARY_LINEWIDTH = 1.15
SUMMARY_MARKERSIZE = 4.0
SUMMARY_MARKEREDGEWIDTH = 0.8
CLEAN_STYLE_SCALE = 1.5
FIGURE_BASENAME = 'Figure1_storage_requirement_turning_points'
FIGURE_TITLE = (
    'Renewable penetration drives accelerated growth in storage requirements '
    'under tighter consumption constraints'
)
FIGURE_CAPTION = (
    'Fig. 1 | Renewable penetration drives accelerated growth in storage '
    'requirements under tighter consumption constraints. Load-normalised '
    'storage duration as a function of renewable penetration for industrial '
    '(b), commercial (c), rural (d) and island (e) microgrids under '
    'renewable-consumption targets of 60%, 70%, 80%, 90% and 100%. '
    'Load-normalised storage duration is defined as the optimised '
    'storage-energy capacity divided by annual mean load. Filled circles '
    'denote the flexibility turning points identified by the '
    'maximum-curvature criterion and provide representative markers of the '
    'continuous shift from gradual to accelerated storage-duration growth. '
    'Increasing the consumption target shifts the turning point towards '
    'lower renewable penetration in all four systems. This migration is '
    'summarised across all four systems in a: turning points converge within '
    'a narrow range of renewable penetration under the 60% target but '
    'disperse considerably under the 100% target, indicating that differences '
    'among system types become more pronounced as the consumption constraint '
    'tightens.'
)

SUMMARY_STYLE = {
    'Industrial': {'color': '#3C5488', 'marker': 'o'},
    'Commercial': {'color': '#00A087', 'marker': 's'},
    'Rural': {'color': '#7E6148', 'marker': '^'},
    'Island': {'color': '#E18727', 'marker': 'D'},
}

KNEE_METHODS = [
    'maximum_curvature',
    'kneedle',
    'L_method',
]
# Calculate all methods for traceability, but display maximum curvature here.
DISPLAY_KNEE_METHODS = ['maximum_curvature']
KNEE_METHOD_STYLES = {
    'kneedle': {'marker': '^', 'markersize': 4.5},
    'maximum_curvature': {'marker': 'o', 'markersize': 4.2},
    'L_method': {'marker': 's', 'markersize': 4.0},
}
# All three methods use exactly the same PCHIP curve and filter:
# discard the leading curve section whose storage duration is below 1 hour.
MIN_STORAGE_DURATION_H = 1.0
MIN_METHOD_POINTS = 7
# A compact derivative window keeps the detected peak aligned with the
# visible bend of the original PCHIP curve.
COMMON_SAVGOL_WINDOW = 101
COMMON_SAVGOL_POLYORDER = 3
REFERENCE_BAT_PRICE_FACTOR = 1.0
KNEE_RESULTS_FILE = os.path.join(
    RESULT_DIR, 'domestic_flexibility_knees_three_methods.xlsx')
KNEE_DIFFERENCE_FILE = os.path.join(
    RESULT_DIR, 'domestic_flexibility_knee_method_differences.csv')


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
    """Filter the leading low-duration section and normalize to plot axes."""
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
        '明显曲率峰数量': 0,
        '有效明显曲率峰数量': 0,
        '曲率峰储能时长下限(h)': np.nan,
        '是否回退到最大曲率': False,
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

    # Exclude the near-zero plateau and require a genuinely rising segment.
    # Judge growth on the original PCHIP curve over a local interval instead
    # of using the derivative at one Savitzky-Golay sample. A long SG window
    # can make that point derivative negative close to a steep terminal rise
    # even when the underlying curve is monotonically increasing.
    min_peak_y = max(
        FIRST_PEAK_MIN_STORAGE_DURATION_H,
        float(y[0] + 0.03 * (y[-1] - y[0])),
    )
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
        idx = int(valid_peaks[0])  # find_peaks returns ascending indices.
    else:
        candidates = np.flatnonzero(finite & (y >= min_peak_y))
        eligible = candidates[locally_rising(candidates)]
        if not len(eligible):
            info = dict(empty_info)
            info['明显曲率峰数量'] = int(len(peaks))
            info['曲率峰储能时长下限(h)'] = min_peak_y
            return (None, None, info) if return_info else (None, None)
        idx = int(eligible[np.argmax(curvature[eligible])])
        used_fallback = True

    info = {
        '明显曲率峰数量': int(len(peaks)),
        '有效明显曲率峰数量': int(len(valid_peaks)),
        '曲率峰储能时长下限(h)': min_peak_y,
        '是否回退到最大曲率': used_fallback,
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


# ═══ Step 1: Process all scenario data ═══
all_panel_data = []
knee_long_rows = []
knee_wide_rows = []

for scenario, cfg in SCENARIOS:
    print(f'\nProcessing: {scenario}')
    divisor = cfg['divisor']
    x_cap   = cfg.get('x_cap')

    if not os.path.isfile(cfg['path']):
        print(f'  WARNING: missing file; skipped: {cfg["path"]}')
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
                f'  WARNING: no bat_price_factor='
                f'{REFERENCE_BAT_PRICE_FACTOR:g} baseline; skipped')
            continue
        df = df.loc[baseline_mask].copy()

    # Support both the original files and the supplemented result schema.
    # Prefer an explicitly reported storage duration; otherwise derive it from
    # the original total battery energy and scenario-specific reference power.
    if 'storage_duration_h' in df.columns:
        duration_source = 'storage_duration_h'
        duration_divisor = 1.0
    elif '小时数' in df.columns:
        duration_source = '小时数'
        duration_divisor = 1.0
    elif 'total_batt_MWh' in df.columns:
        duration_source = 'total_batt_MWh'
        duration_divisor = divisor
    else:
        print(f'  WARNING: missing storage-duration/capacity field; '
              f'skipped: {cfg["path"]}')
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
            print(f'  WARNING: the {t_int}% curve does not reach '
                  f'{MIN_STORAGE_DURATION_H:g} h; skipped')
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
            'maximum_curvature': find_max_curvature(
                x_line, y_line, growth_start_x),
            'L_method': find_l_method(x_line, y_line, growth_start_x),
        }
        method_knee_data[t_int] = knees

        wide_row = {
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
        'slug':        cfg['slug'],
        'x_cap':       x_cap,
        'curve_data':  curve_data,
        'method_knee_data': method_knee_data,
        'growth_start_data': growth_start_data,
        'raw_data':    raw_data,
        'sorted_t':    sorted_t,
    })


# Export three-method knee-point statistics and pairwise gaps.
knee_results_long = pd.DataFrame(knee_long_rows)
knee_results_wide = pd.DataFrame(knee_wide_rows)
if not knee_results_long.empty:
    point_columns = [f'{method}_penetration_%' for method in KNEE_METHODS]
    differences = knee_results_wide[
        ['scenario', 'consumption_target_%', *point_columns]].copy()
    differences['maximum_curvature_vs_kneedle_abs_pp'] = (
        differences[point_columns[0]] - differences[point_columns[1]]).abs()
    differences['maximum_curvature_vs_L_method_abs_pp'] = (
        differences[point_columns[0]] - differences[point_columns[2]]).abs()
    differences['kneedle_vs_L_method_abs_pp'] = (
        differences[point_columns[1]] - differences[point_columns[2]]).abs()
    differences['three_method_max_span_pp'] = (
        differences[point_columns].max(axis=1)
        - differences[point_columns].min(axis=1))
    differences.to_csv(
        KNEE_DIFFERENCE_FILE, index=False, encoding='utf-8-sig',
        float_format='%.4f')
    with pd.ExcelWriter(KNEE_RESULTS_FILE) as writer:
        knee_results_wide.to_excel(
            writer, sheet_name='method_comparison', index=False,
            float_format='%.4f')
        knee_results_long.to_excel(
            writer, sheet_name='long_format', index=False,
            float_format='%.4f')
    print('\nThree-method results (main figure: maximum curvature only):')
    print(knee_results_wide.to_string(
        index=False, float_format=lambda value: f'{value:.3f}'))
    print(f'Saved: {KNEE_RESULTS_FILE}')
    print(f'Saved: {KNEE_DIFFERENCE_FILE}')


# ═══ Step 2: Draw the five-panel mechanism figure ═══
def _curve_value(x, y, x_value):
    return float(np.interp(x_value, np.asarray(x, float), np.asarray(y, float)))


def add_growth_annotations(ax, panel, preferred_target=80):
    """Label one representative curve without implying a discrete phase split."""
    available = panel['sorted_t']
    if not available:
        return
    target = min(available, key=lambda value: abs(value - preferred_target))
    x, y = panel['curve_data'][target]
    knee_x, _ = panel['method_knee_data'][target]['maximum_curvature']
    if knee_x is None or knee_x <= x[0] or knee_x >= x[-1]:
        return

    grey = '#595959'
    pre_start = x[0] + 0.50 * (knee_x - x[0])
    pre_end = x[0] + 0.82 * (knee_x - x[0])
    post_start = knee_x + 0.10 * (x[-1] - knee_x)
    post_end = knee_x + 0.48 * (x[-1] - knee_x)

    segments = [
        (pre_start, pre_end, 'gradual growth', 6),
        (post_start, post_end, 'accelerated growth', 10),
    ]
    for x0, x1, label, y_offset in segments:
        y0, y1 = _curve_value(x, y, x0), _curve_value(x, y, x1)
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color=grey, lw=0.65,
                            shrinkA=0, shrinkB=0, mutation_scale=7),
            zorder=12,
        )
        ax.text(
            (x0 + x1) / 2, min(Y_CAP - 6, (y0 + y1) / 2 + y_offset),
            label, color=grey, fontsize=5.5, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=0.6),
            zorder=13,
        )


def draw_curve_panel(ax, panel, pidx, labeled=True, style_scale=1.0):
    """Draw curves and filled maximum-curvature markers only."""
    for target in panel['sorted_t']:
        x_line, y_line = panel['curve_data'][target]
        color = COLOR_MAP.get(target, '#767676')
        ax.plot(
            x_line, y_line, color=color,
            linewidth=CURVE_LINEWIDTH * style_scale, zorder=4,
        )

        method_x, method_y = (
            panel['method_knee_data'][target]['maximum_curvature'])
        if method_x is not None:
            ax.plot(
                method_x, method_y, linestyle='none', marker='o',
                markersize=TURNING_POINT_MARKERSIZE * style_scale,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=(
                    TURNING_POINT_MARKEREDGEWIDTH * style_scale),
                zorder=10,
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y_CAP)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, Y_CAP + 1, 40))
    ax.tick_params(axis='both', which='major', length=2.5, width=0.6,
                   labelsize=6, direction='out', pad=2)

    if not labeled:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return

    ax.text(-0.09, 1.03, PANEL_LABELS[pidx], transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax.set_title(panel['scenario'], fontsize=8, fontweight='bold', pad=5)
    scenario_index = pidx - 1
    if scenario_index >= 2:
        ax.set_xlabel('Renewable penetration (%)', fontsize=7)
    if scenario_index % 2 == 0:
        ax.set_ylabel('Load-normalised storage duration (h)', fontsize=7)
    add_growth_annotations(ax, panel)


def _spread_direct_labels(values, lower=42, upper=98, min_gap=4.0):
    """Return non-overlapping y positions for direct labels in panel e."""
    order = np.argsort(values)
    placed = np.asarray(values, float)[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    if placed[-1] > upper:
        placed -= placed[-1] - upper
    if placed[0] < lower:
        placed += lower - placed[0]
    result = np.empty_like(placed)
    result[order] = placed
    return result


def draw_summary_panel(ax):
    """Summarise systematic turning-point migration across all systems."""
    line_records = []
    for panel in all_panel_data:
        targets, penetrations = [], []
        for target in panel['sorted_t']:
            knee_x, _ = (
                panel['method_knee_data'][target]['maximum_curvature'])
            if knee_x is not None:
                targets.append(target)
                penetrations.append(knee_x)
        style = SUMMARY_STYLE[panel['scenario']]
        ax.plot(
            targets, penetrations, color=style['color'], marker=style['marker'],
            markersize=SUMMARY_MARKERSIZE, markerfacecolor='white',
            markeredgewidth=SUMMARY_MARKEREDGEWIDTH,
            linewidth=SUMMARY_LINEWIDTH, zorder=4, clip_on=False,
        )
        if targets:
            line_records.append((panel['scenario'], targets[-1],
                                 penetrations[-1], style['color']))

    ax.set_xlim(60, 100)
    ax.set_ylim(40, 100)
    ax.set_xticks(sorted(COLOR_MAP))
    ax.set_yticks(np.arange(40, 101, 10))
    ax.set_xlabel('Renewable-consumption target (%)', fontsize=7)
    ax.set_ylabel('Turning-point renewable\npenetration (%)', fontsize=7)
    ax.tick_params(axis='both', which='major', length=2.5, width=0.6,
                   labelsize=6, direction='out', pad=2)
    ax.text(-0.045, 1.03, PANEL_LABELS[0], transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax.set_title('Turning-point migration', fontsize=8, fontweight='bold', pad=5)

    ax.annotate(
        '', xy=(96, 95.8), xytext=(64, 95.8),
        arrowprops=dict(arrowstyle='->', color='#595959', lw=0.65,
                        mutation_scale=7),
    )
    ax.text(80, 96.5, 'Tighter consumption constraint', fontsize=5.7,
            color='#595959', ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.5))

    if line_records:
        label_y = _spread_direct_labels([record[2] for record in line_records])
        for (name, x_end, y_end, color), y_text in zip(line_records, label_y):
            ax.plot([x_end - 0.3, 98.9], [y_end, y_text], color=color,
                    linewidth=0.55, alpha=0.8)
            ax.text(
                98.5, y_text, name, color=color, fontsize=6,
                fontweight='bold', ha='right', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9,
                          pad=0.3),
            )


def draw_figure():
    """Create a left-side summary with a 2×2 scenario grid on the right."""
    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1.16, 1.0, 1.0],
        left=0.085, right=0.975, bottom=0.12, top=0.82,
        hspace=0.43, wspace=0.42,
    )
    axes = [
        fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2]),
    ]
    for pidx, (ax, panel) in enumerate(zip(axes, all_panel_data)):
        draw_curve_panel(ax, panel, pidx + 1, labeled=True)

    summary_ax = fig.add_subplot(grid[:, 0])
    draw_summary_panel(summary_ax)

    target_handles = [
        Line2D([0], [0], color=COLOR_MAP[target], lw=1.5,
               label=f'{target}%')
        for target in sorted(COLOR_MAP)
    ]
    turning_point_handle = Line2D(
        [0], [0], linestyle='none', marker='o',
        markersize=TURNING_POINT_MARKERSIZE,
        markerfacecolor='#777777', markeredgecolor='black',
        markeredgewidth=TURNING_POINT_MARKEREDGEWIDTH,
        label='Flexibility turning point (maximum curvature)',
    )
    target_legend = fig.legend(
        handles=target_handles, loc='upper center', ncol=5, fontsize=6,
        frameon=False, bbox_to_anchor=(0.50, 0.975), handletextpad=0.45,
        columnspacing=1.15,
        title='Renewable-consumption target', title_fontsize=6.5,
    )
    fig.add_artist(target_legend)
    fig.legend(
        handles=[turning_point_handle], loc='upper center', ncol=1,
        fontsize=6, frameon=False, bbox_to_anchor=(0.50, 0.915),
        handletextpad=0.55,
    )
    return fig


def draw_clean_panel(panel):
    """Create one text-free curve-only scenario panel."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    draw_curve_panel(
        ax, panel, pidx=0, labeled=False,
        style_scale=CLEAN_STYLE_SCALE,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.99)
    return fig


def draw_clean_summary_panel():
    """Create a text-free version of panel e for external figure assembly."""
    fig, ax = plt.subplots(figsize=(2.5, 4.6))
    for panel in all_panel_data:
        targets, penetrations = [], []
        for target in panel['sorted_t']:
            knee_x, _ = (
                panel['method_knee_data'][target]['maximum_curvature'])
            if knee_x is not None:
                targets.append(target)
                penetrations.append(knee_x)
        style = SUMMARY_STYLE[panel['scenario']]
        ax.plot(
            targets, penetrations, color=style['color'],
            marker=style['marker'],
            markersize=SUMMARY_MARKERSIZE * CLEAN_STYLE_SCALE,
            markerfacecolor='white',
            markeredgewidth=(
                SUMMARY_MARKEREDGEWIDTH * CLEAN_STYLE_SCALE),
            linewidth=SUMMARY_LINEWIDTH * CLEAN_STYLE_SCALE,
            zorder=4, clip_on=False,
        )

    ax.set_xlim(60, 100)
    ax.set_ylim(40, 100)
    ax.set_xticks(sorted(COLOR_MAP))
    ax.set_yticks(np.arange(40, 101, 10))
    ax.tick_params(
        axis='both', which='major', length=2.5, width=0.6,
        direction='out', labelbottom=False, labelleft=False,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.05, top=0.99)
    return fig


# ═══ Step 3: Export the publication bundle ═══
fig_labeled = draw_figure()
main_output = os.path.join(RESULT_DIR, FIGURE_BASENAME)
fig_labeled.savefig(f'{main_output}.svg', bbox_inches='tight')
plt.close(fig_labeled)

caption_output = os.path.join(RESULT_DIR, 'Figure1_caption.txt')
with open(caption_output, 'w', encoding='utf-8') as caption_file:
    caption_file.write(FIGURE_CAPTION + '\n')

print(f'\nFigure title: {FIGURE_TITLE}')
print(f'Saved: {main_output}.svg')
print(f'Saved: {caption_output}')

for panel in all_panel_data:
    fig_clean = draw_clean_panel(panel)
    panel_output = os.path.join(CLEAN_DIR, f'{panel["slug"]}.svg')
    fig_clean.savefig(panel_output, bbox_inches='tight')
    plt.close(fig_clean)
    print(f'Saved: {panel_output}')

fig_clean_summary = draw_clean_summary_panel()
summary_panel_output = os.path.join(
    CLEAN_DIR, 'turning_point_migration.svg')
fig_clean_summary.savefig(summary_panel_output, bbox_inches='tight')
plt.close(fig_clean_summary)
print(f'Saved: {summary_panel_output}')

print('\nAll outputs completed.')
