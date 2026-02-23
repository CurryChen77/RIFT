import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
from pathlib import Path
from matplotlib.lines import Line2D

# ==========================================
# 1. Global style settings (ECCV-like aesthetics)
# ==========================================
# Use LaTeX-like serif fonts for a more academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. Data preparation (computed from results.csv)
# ==========================================
target_groups = ['plant', 'uniad', 'vad']
planner_point_colors = {
    'plant': '#830080',
    'uniad': "#e06d15",
    'vad': "#f5d664",
}
planner_point_markers = {
    'plant': 'o',
    'uniad': 'D',
    'vad': 's',
}
planner_display_names = {
    'plant': 'PlanT',
    'uniad': 'UniAD',
    'vad': 'VAD',
}


def format_method_name(method_name: str) -> str:
    label_map = {
        'pluto': 'Pluto',
        'ppo': 'PPO',
        'frea': 'FREA',
        'fppo_rs': 'FPPO-RS',
        'sft_pluto': 'SFT-Pluto',
        'rs_pluto': 'RS-Pluto',
        'rtr_pluto': 'RTR-Pluto',
        'ppo_pluto': 'PPO-Pluto',
        'reinforce_pluto': 'REINFORCE-Pluto',
        'grpo_pluto': 'GRPO-Pluto',
        'RIFT (ours)': 'RIFT-Pluto',
    }
    return label_map.get(method_name, method_name.replace('_', '-'))

def parse_mean(metric_cell: str) -> float:
    return float(metric_cell.split('±')[0].strip())


def load_metrics_by_group(csv_path: Path):
    metric_names = ['Driving Score (↑)', 'Route Completion (↑)', 'Infraction Penalty (↑)']
    results = {}
    current_group = None
    header = None

    with csv_path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('Group:'):
                current_group = line.split(':', 1)[1].strip().lower()
                header = None
                continue

            if current_group is None:
                continue

            row = next(csv.reader([line]))
            if row[0].strip() == 'Method':
                header = row
                continue

            if header is None:
                continue

            method_name = row[0].strip()
            idx = {name: header.index(name) for name in metric_names}
            results.setdefault(current_group, {})[method_name] = {
                metric: parse_mean(row[col_idx])
                for metric, col_idx in idx.items()
            }

    return results


root_dir = Path(__file__).resolve().parents[2]
csv_file = root_dir / 'assets' / 'results.csv'
group_metrics = load_metrics_by_group(csv_file)

common_method_set = (
    set(group_metrics['pdm_lite'].keys())
    .intersection(*[set(group_metrics[g].keys()) for g in target_groups])
)

# Keep original row order from PDM-Lite, and force RIFT to the far right.
pdm_lite_order = list(group_metrics['pdm_lite'].keys())
common_methods = [method for method in pdm_lite_order if method in common_method_set]
if 'RIFT (ours)' in common_methods:
    common_methods = [m for m in common_methods if m != 'RIFT (ours)'] + ['RIFT (ours)']

method_stats = []
for method_name in common_methods:
    rc_rel = []
    ip_rel = []
    rc_rel_by_planner = []
    ip_rel_by_planner = []

    for planner_group in target_groups:
        baseline = group_metrics['pdm_lite'][method_name]
        target = group_metrics[planner_group][method_name]

        delta_rc = baseline['Route Completion (↑)'] - target['Route Completion (↑)']
        delta_ip = baseline['Infraction Penalty (↑)'] - target['Infraction Penalty (↑)']

        rc_drop_pct = 100.0 * delta_rc / max(baseline['Route Completion (↑)'], 1e-8)
        ip_drop_pct = 100.0 * delta_ip / max(baseline['Infraction Penalty (↑)'], 1e-8)

        rc_rel.append(rc_drop_pct)
        ip_rel.append(ip_drop_pct)
        rc_rel_by_planner.append(rc_drop_pct)
        ip_rel_by_planner.append(ip_drop_pct)

    method_stats.append({
        'method': method_name,
        'rc_mean': float(np.mean(rc_rel)),
        'rc_std': float(np.std(rc_rel)),
        'ip_mean': float(np.mean(ip_rel)),
        'ip_std': float(np.std(ip_rel)),
        'rc_by_planner': rc_rel_by_planner,
        'ip_by_planner': ip_rel_by_planner,
    })

# Sort by delta_IP from small to large (largest at the right-most side).
method_stats.sort(key=lambda item: item['ip_mean'])

methods = [item['method'] for item in method_stats]
method_labels = [format_method_name(item['method']) for item in method_stats]
rc_means = np.array([item['rc_mean'] for item in method_stats])
ip_means = np.array([item['ip_mean'] for item in method_stats])
rc_stds = np.array([item['rc_std'] for item in method_stats])
ip_stds = np.array([item['ip_std'] for item in method_stats])

x = np.arange(len(methods))
width = 0.34  # Bar width

# ==========================================
# 3. Color palette (color-blind friendly, publication style)
# ==========================================
# Deep blue (DS), light gray-cyan (RC), coral red/orange (IP: risk/conflict emphasis)
color_rc = '#cb7b37'  # Coral orange (for RC, moderate emphasis)
color_ip = '#93186e'   # Deep magenta (for IP, stronger emphasis)

# ==========================================
# 4. Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6))


def compress_negative_scale(values, factor=0.2):
    arr = np.asarray(values, dtype=float)
    return np.where(arr >= 0.0, arr, arr * factor)


def inverse_compress_negative_scale(values, factor=0.2):
    arr = np.asarray(values, dtype=float)
    return np.where(arr >= 0.0, arr, arr / factor)


# Compress only the negative region while keeping all negative points visible.
negative_scale_factor = 0.2
ax.set_yscale(
    'function',
    functions=(
        lambda v: compress_negative_scale(v, factor=negative_scale_factor),
        lambda v: inverse_compress_negative_scale(v, factor=negative_scale_factor),
    ),
)

error_style = {
    'ecolor': '#7A7A7A',
    'elinewidth': 1.2,
    'alpha': 0.8,
    'capsize': 3,
    'capthick': 1.0,
}

rects_rc = ax.bar(
    x - width / 2,
    rc_means,
    width,
    yerr=rc_stds,
    error_kw=error_style,
    label='ΔRC / RC (PDM-Lite) (%)',
    color=color_rc,
    linewidth=0.9,
    alpha=0.7,
    zorder=3,
    # hatch='///',
    edgecolor='#FFFFFF',
)
rects_ip = ax.bar(
    x + width / 2,
    ip_means,
    width,
    yerr=ip_stds,
    error_kw=error_style,
    label='ΔIP / IP (PDM-Lite) (%)',
    color=color_ip,
    linewidth=0.9,
    alpha=0.8,
    zorder=3,
)

# Add planner-wise evidence points (3 planners) for each method
for idx, item in enumerate(method_stats):
    rc_x = np.full(len(target_groups), x[idx] - width / 2)
    ip_x = np.full(len(target_groups), x[idx] + width / 2)
    planner_offsets = np.array([-0.1, 0.0, 0.1])

    for planner_idx, planner_group in enumerate(target_groups):
        point_color = planner_point_colors[planner_group]
        point_marker = planner_point_markers[planner_group]
        ax.scatter(
            rc_x[planner_idx] + planner_offsets[planner_idx],
            item['rc_by_planner'][planner_idx],
            s=52,
            marker=point_marker,
            color=point_color,
            edgecolors='none',
            linewidths=0.0,
            alpha=0.8,
            zorder=4,
        )
        ax.scatter(
            ip_x[planner_idx] + planner_offsets[planner_idx],
            item['ip_by_planner'][planner_idx],
            s=52,
            marker=point_marker,
            color=point_color,
            edgecolors='none',
            linewidths=0.0,
            alpha=0.8,
            zorder=4,
        )

# Highlight RIFT bars to emphasize the target method
if 'RIFT (ours)' in methods:
    rift_index = methods.index('RIFT (ours)')
    for bar_collection in [rects_rc, rects_ip]:
        bar_collection[rift_index].set_linewidth(2)
        bar_collection[rift_index].set_edgecolor("#3D3C3D")

# ==========================================
# 5. Figure refinement
# ==========================================
# Compute range from bars, error bars, and planner points
raw_negative_floor = np.floor(min(
    0.0,
    float(np.min(rc_means - rc_stds)),
    float(np.min(ip_means - ip_stds)),
    min(min(item['rc_by_planner']) for item in method_stats),
    min(min(item['ip_by_planner']) for item in method_stats),
) - 0.8)
top_upper = max(
    float(np.max(rc_means + rc_stds)),
    float(np.max(ip_means + ip_stds)),
    max(max(item['rc_by_planner']) for item in method_stats),
    max(max(item['ip_by_planner']) for item in method_stats),
) + 1.0

ax.set_ylim(raw_negative_floor, top_upper)

# Set labels and ticks
ax.set_ylabel('Relative Drop (%)', fontsize=15, fontweight='bold', color="#3A3A3A")
ax.set_xticks(x)
tick_labels = ax.set_xticklabels(method_labels, rotation=30, ha='center', fontweight='bold', color="#3A3A3A")
for tick in tick_labels:
    tick_text = tick.get_text()
    if tick_text == 'RIFT-Pluto':
        tick.set_fontsize(14.5)
    else:
        name_len = len(tick_text)
        if name_len <= 5:
            tick.set_fontsize(13.5)
        elif name_len <= 9:
            tick.set_fontsize(12.5)
        elif name_len <= 12:
            tick.set_fontsize(11.5)
        elif name_len <= 16:
            tick.set_fontsize(10.5)
        else:
            tick.set_fontsize(9.8)
ax.tick_params(axis='y', labelsize=14)

# Use sparse ticks for a cleaner look
if raw_negative_floor < 0:
    neg_mid = 0.5 * raw_negative_floor
    neg_ticks = np.array([raw_negative_floor, neg_mid, 0.0])
else:
    neg_ticks = np.array([0.0])
pos_ticks = np.arange(5.0, np.ceil(top_upper / 5.0) * 5.0 + 0.001, 5.0)
ax.set_yticks(np.concatenate([neg_ticks, pos_ticks]))

# Add legends: bars + planner evidence points
legend_text_props = {'family': 'Times New Roman', 'size': 14}
bar_legend = ax.legend(
    loc='upper right',
    frameon=False,
    ncol=1,
    prop=legend_text_props,
)
planner_handles = [
    Line2D([0], [0], marker=planner_point_markers[group], linestyle='None', markersize=12,
           markerfacecolor=planner_point_colors[group],
           markeredgecolor='none', markeredgewidth=0.0,
           label=planner_display_names[group])
    for group in target_groups
]
planner_legend = ax.legend(
    handles=planner_handles,
    loc='upper left',
    frameon=False,
    ncol=3,
    prop=legend_text_props,
)
ax.add_artist(bar_legend)

# Add horizontal grid lines (behind bars)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# Remove top/right spines for a cleaner conference-style look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# # Add numeric labels above bars
# def autolabel(axis, rects):
#     for rect in rects:
#         height = rect.get_height()
#         if height >= 0:
#             axis.annotate(f'{height:.1f}',
#                           xy=(rect.get_x() + rect.get_width() / 2, height),
#                           xytext=(0, 3),
#                           textcoords="offset points", zorder=8,
#                           ha='center', va='bottom', fontsize=11, fontfamily='sans-serif')
#         else:
#             axis.annotate(f'{height:.1f}',
#                           xy=(rect.get_x() + rect.get_width() / 2, height),
#                           xytext=(0, -10),
#                           textcoords="offset points", zorder=8,
#                           ha='center', va='top', fontsize=11, fontfamily='sans-serif')


# autolabel(ax, rects_rc)
# autolabel(ax, rects_ip)

if raw_negative_floor < 0:
    # a thin band around y=0 to indicate axis transform
    ax.axhspan(-0.15, 0.15, color='#BBBBBB', alpha=0.18, zorder=1)

    ax.text(
        0.01, 0.015,
        f'Negative region compressed ×{negative_scale_factor:g}',
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=11, color='#444444'
    )

# Increase horizontal margins so rightmost label and bars are not cramped
ax.margins(x=0.02)

# Tight layout
fig.tight_layout()

# ==========================================
# 6. Save high-quality vector figure
# ==========================================
plt.savefig('assets/av_evaluation_degradation.pdf', format='pdf', dpi=500, bbox_inches='tight')
plt.savefig('assets/av_evaluation_degradation.png', format='png', dpi=500, bbox_inches='tight')

plt.close()