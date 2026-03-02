#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : plot_eval_result.py
@Date    : 2025/03/02
'''

import json
import importlib
import argparse
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D


from typing import Dict, List
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from matplotlib.legend_handler import HandlerPatch

METRICS = {
    'acc': r"CBV Acceleration $(\mathit{m/s^2})$",
    'jerk': r"CBV Jerk $(\mathit{m/s^3})$",
    'speed': r"CBV Speed $(\mathit{m/s})$",
    'EI': r"EI $(\uparrow)$",
    'RTTC': r"RTTC $(\downarrow)$",
    'ACT': r"ACT $(\downarrow)$",
}

ALGOS = {
    'frea': 'FREA',
    'ppo': 'PPO',
    'fppo_rs': 'FPPO-RS',
    'pluto': 'Pluto',
    'sft_pluto': 'SFT-Pluto',
    'grpo_pluto': 'GRPO-Pluto',
    'rift_pluto': 'RIFT-Pluto',
    'rtr_pluto': 'RTR-Pluto',
    'rs_pluto': 'RS-Pluto',
    'ppo_pluto': 'PPO-Pluto',
    'reinforce_pluto': 'REINFORCE-Pluto',
}

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 26,
    'legend.fontsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
})

# Custom handler to shrink Patch size
class SmallPatchHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        patch = Patch(
            facecolor=orig_handle.get_facecolor()[0],
            edgecolor=orig_handle.get_edgecolor()[0],
            lw=orig_handle.get_linewidth(),
            alpha=orig_handle.get_alpha(),
            transform=trans
        )
        # Set smaller bounds (smaller width & height)
        patch.set_bounds(xdescent, ydescent + height * 0.4, width * 0.5, height * 0.8)
        return [patch]

def parse_value(v):
    """
    Parse a value from string to appropriate data type (int, float, or original string).
    """
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (SyntaxError, ValueError):
        pass
    try:
        if '.' in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        pass
    return v

def load_labels_values_as_dict(json_path: str):
    """
    Load JSON file and parse labels and values into a dictionary.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = data.get("labels", [])
    values = data.get("values", [])
    results_dict = {}
    for label, value in zip(labels, values):
        results_dict[label] = parse_value(value)
    return results_dict

def parse_hist_dict(hist_dict):
    """
    Parse histogram dictionary into a list of tuples (left, right, count).
    """
    parsed = []
    for bin_str, count in hist_dict.items():
        try:
            left_str, right_str = bin_str.split('~')
            left_val = float(left_str)
            right_val = float(right_str)
            parsed.append((left_val, right_val, count))
        except:
            pass
    parsed.sort(key=lambda x: x[0])  # Sort by the left boundary of bins
    return parsed

def sample_from_hist(parsed_bins, n_samples=1000):
    """
    Generate random samples from histogram bins based on their counts.
    """
    total_count = sum(bin_item[2] for bin_item in parsed_bins)
    if total_count == 0:
        return np.zeros(n_samples)

    probs = [bin_item[2]/total_count for bin_item in parsed_bins]
    cum_probs = np.cumsum(probs)
    rand_vals = np.random.rand(n_samples)
    samples = np.zeros(n_samples)

    for i in range(n_samples):
        r = rand_vals[i]
        idx = np.searchsorted(cum_probs, r)
        left, right, _ = parsed_bins[idx]
        samples[i] = np.random.uniform(left, right)  # Sample uniformly within the bin
    return samples

def generate_2d_samples(acc_hist_dict, speed_hist_dict, n_samples=1000):
    """
    Generate 2D samples from acceleration and speed histograms.
    """
    acc_bins = parse_hist_dict(acc_hist_dict)
    speed_bins = parse_hist_dict(speed_hist_dict)
    acc_samples = sample_from_hist(acc_bins, n_samples)
    speed_samples = sample_from_hist(speed_bins, n_samples)
    return np.column_stack((acc_samples, speed_samples))  # Combine into 2D array (n_sample, 2)

def create_alpha_cmap(base_color, alpha, name='alpha_cmap'):
    """
    Create a colormap with inverted alpha (higher probability → lower alpha, fixed color).
    Args:
        base_color: Base color string (e.g., '#A80326')
    """
    rgb = mcolors.to_rgb(base_color)  # Convert the base color to RGB values
    cmap_dict = {
        'red':   [(0.0, rgb[0], rgb[0]), (1.0, rgb[0], rgb[0])],  # Fixed red channel
        'green': [(0.0, rgb[1], rgb[1]), (1.0, rgb[1], rgb[1])],  # Fixed green channel
        'blue':  [(0.0, rgb[2], rgb[2]), (1.0, rgb[2], rgb[2])],  # Fixed blue channel
        'alpha': [(0.0, 0.0, 0.0), (1.0, alpha, alpha)]
    }
    return mcolors.LinearSegmentedColormap(name, cmap_dict)

def _draw_kde_contour(ax, data, base_color, levels=10, alpha=0.8, density_threshold=0.00001):
    kde = gaussian_kde(data.T)  # Compute the kernel density estimate (KDE)
    x_min, x_max = data[:, 0].min(), data[:, 0].max()  # Get min and max of x-axis data
    y_min, y_max = data[:, 1].min(), data[:, 1].max()  # Get min and max of y-axis data

    pad_x = 0.5 * (x_max - x_min)  # Padding for x-axis
    pad_y = 0.5 * (y_max - y_min)  # Padding for y-axis
    xgrid = np.linspace(x_min - 0.1 * pad_x, x_max + 0.1 * pad_x, 200)  # Generate x-axis grid
    ygrid = np.linspace(y_min - 0.1 * pad_y, y_max + 0.1 * pad_y, 200)  # Generate y-axis grid
    X, Y = np.meshgrid(xgrid, ygrid)  # Create a meshgrid for contour plotting
    positions = np.vstack([X.ravel(), Y.ravel()])  # Flatten the grid for KDE evaluation
    Z = np.reshape(kde(positions).T, X.shape)  # Evaluate KDE and reshape to grid size

    # Normalize Z values to represent probability density (integral equals 1)
    Z_normalized = Z / Z.sum()

    # Check if the threshold is valid
    max_density = Z_normalized.max()
    if density_threshold >= max_density:
        density_threshold = max_density * 0.9  # Automatically adjust threshold to 90% of max value
        print(f"Warning: density_threshold too high. Adjusted to {density_threshold:.3f}")

    # Mask regions below the threshold
    Z_masked = np.where(Z_normalized >= density_threshold, Z_normalized, np.nan)

    # Generate contour levels (ensure at least 2 levels)
    valid_levels = np.linspace(density_threshold, max_density, levels)
    if len(valid_levels) < 2:
        valid_levels = np.linspace(density_threshold, max_density, 2)

    alpha_cmap = create_alpha_cmap(base_color, alpha)  # Create the custom colormap

    # Draw filled contours
    cs = ax.contourf(X, Y, Z_masked, levels=valid_levels, cmap=alpha_cmap)


def _style_main_axis_spines(ax):
    """Soften axis borders for cleaner PDF/vector rendering."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#A8A8A8')
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis='both', colors='#565656')

def plot_2d_with_marginals(main_ax,
                           data1, data2, 
                           algo_1, algo_2,
                           metric_1, metric_2,
                           color1='#b62230',  # Red color for data1
                           color2='#B192B4',  # Blue color for data2
                           scatter_interval=50,
                           cluster_centers=True):
    """
    Plot 2D scatter plot with marginal histograms and optional cluster centers.
    """
    # Define main plot and marginal plots
    divider = make_axes_locatable(main_ax)
    x_hist_ax = divider.append_axes("top", 1.2, pad=0.0, sharex=main_ax)
    y_hist_ax = divider.append_axes("right", 1.2, pad=0.0, sharey=main_ax)
    
    plt.setp(x_hist_ax.get_xticklabels(), visible=False)  # Hide x-axis labels for top histogram
    plt.setp(y_hist_ax.get_yticklabels(), visible=False)  # Hide y-axis labels for right histogram

    # Perform KMeans clustering to find cluster centers
    kmeans1 = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(data1)
    center1 = kmeans1.cluster_centers_[0]
    kmeans2 = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(data2)
    center2 = kmeans2.cluster_centers_[0]

    # Draw KDE contours for both datasets
    _draw_kde_contour(main_ax, data1, base_color=color1, levels=6, alpha=0.8)
    _draw_kde_contour(main_ax, data2, base_color=color2, levels=5, alpha=1.0)

    # # Scatter plot for both datasets
    # main_ax.scatter(data1[::scatter_interval, 0], data1[::scatter_interval, 1], s=100, alpha=0.5, c=color1)
    # main_ax.scatter(data2[::scatter_interval, 0], data2[::scatter_interval, 1], s=100, alpha=0.5, c=color2)

    if cluster_centers:
        # Draw gray lines for cluster centers
        main_ax.axvline(x=center1[0], color='gray', linestyle='-', linewidth=2)
        main_ax.axhline(y=center1[1], color='gray', linestyle='-', linewidth=2)

        # Mark cluster centers with circles
        main_ax.plot(center1[0], center1[1], marker='s', markersize=20, 
                     c=color1, mec='gray', mew=1.0, label=ALGOS[algo_1])
        main_ax.plot(center2[0], center2[1], marker='s', markersize=20, 
                     c=color2, mec='gray', mew=1.0, label=ALGOS[algo_2])
        
        distance = np.linalg.norm(np.array(center1) - np.array(center2))

        # Arrow Attributes
        arrow_width = 4.0 + distance * 0.2
        arrow_head_width = 0.35 + distance * 0.05
        arrow_head_length = 0.45 + distance * 0.1

        # Add Arrow
        arrow = FancyArrowPatch(
            posA=(center1[0], center1[1]),  # start point
            posB=(center2[0], center2[1]),  # end point
            color='white',                  # color
            arrowstyle=f'-|>, head_width={arrow_head_width}, head_length={arrow_head_length}',
            linewidth=arrow_width,
            zorder=10,                      # draw on top of other elements
            mutation_scale=15               # arrow size
        )
        main_ax.add_patch(arrow)

    # Set axis labels and legend
    main_ax.set_xlabel(METRICS[metric_1], fontsize=32)
    main_ax.set_ylabel(METRICS[metric_2], fontsize=32)
    main_ax.tick_params(axis='both', which='major', labelsize=28)
    _style_main_axis_spines(main_ax)
    main_ax.legend(loc='upper right')

    # Plot histograms for x-axis (top)
    # x_hist_ax.hist([data1[:,0], data2[:,0]],
    #                bins=30, color=[color1, color2],
    #                alpha=0.6, edgecolor='none', rwidth=2.0, stacked=True)
    x_hist_ax.hist(
        data1[:, 0], bins=30, color=color1, alpha=0.8,
        density=True, histtype='stepfilled', linewidth=0
    )
    x_hist_ax.hist(
        data2[:, 0], bins=30, color=color2, alpha=0.8,
        density=True, histtype='stepfilled', linewidth=0
    )
    x_hist_ax.tick_params(axis='y', left=False, labelleft=False)  # Hide y-axis ticks for top histogram
    
    # Plot histograms for y-axis (right)
    # y_hist_ax.hist([data1[:,1], data2[:,1]],
    #                bins=30, orientation='horizontal',
    #                color=[color1, color2], alpha=0.6,
    #                edgecolor='none', rwidth=2.0, stacked=True)
    y_hist_ax.hist(
        data1[:, 1], bins=30, orientation='horizontal',
        color=color1, alpha=0.8, density=True, histtype='stepfilled', linewidth=0
    )
    y_hist_ax.hist(
        data2[:, 1], bins=30, orientation='horizontal',
        color=color2, alpha=0.8, density=True, histtype='stepfilled', linewidth=0
    )
    y_hist_ax.tick_params(axis='x', bottom=False, labelbottom=False)  # Hide x-axis ticks for right histogram

    # Remove spines for marginal plots
    for spine in x_hist_ax.spines.values():
        spine.set_visible(False)
    for spine in y_hist_ax.spines.values():
        spine.set_visible(False)


def plot_all_cluster_centers(main_ax, sample_dict, metric_1, metric_2):
    """
    Plot cluster centers for all methods, grouped and colored by method types.
    """
    cb = importlib.import_module("palettable.cartocolors.sequential")
    cb1 = importlib.import_module("palettable.cmocean.sequential")

    group_algos = {
        'RLFT': ['rift_pluto', 'grpo_pluto', 'reinforce_pluto', 'ppo_pluto'],
        'SFT(+RLFT)': ['sft_pluto', 'rtr_pluto', 'rs_pluto'],
        'RL': ['frea', 'ppo', 'fppo_rs'],
        'IL': ['pluto'],
    }
    group_colors = {
        'RL': 'Magenta_5',
        'SFT(+RLFT)': 'Teal_4',
        'RLFT': 'Peach_4',
        'IL': 'Mint_4'
    }

    group_markers = {
        'RLFT': 's',  # square
        'SFT(+RLFT)': 'o',   # circle
        'RL': '^',    # triangle
        'IL': 'D',    # diamond
    }

    group_order = {
        'RL': 1,
        'SFT(+RLFT)': 2,
        'RLFT': 3,
        'IL': 4
    }

    # Compute cluster center for each method
    centers = {
        algo: KMeans(n_clusters=1, random_state=42, n_init='auto').fit(data).cluster_centers_[0]
        for algo, data in sample_dict.items()
    }

    # Plot cluster centers and group regions
    for group, algos in group_algos.items():
        pts = [centers[algo] for algo in algos if algo in centers]
        if pts:
            cmap = getattr(cb, group_colors[group])
            colors = list(reversed(cmap.hex_colors))
            pts = np.array(pts)
            poly_pts = np.vstack([pts, pts[0]])
            edge_color = colors[-1]
            main_ax.fill(poly_pts[:, 0], poly_pts[:, 1], color=edge_color, alpha=0.32, zorder=group_order[group])

            soft_edge_layers = [
                (18, 0.04),
                (12, 0.06),
                (8, 0.10),
                (4, 0.15),
            ]
            for linewidth, alpha in soft_edge_layers:
                main_ax.plot(
                    poly_pts[:, 0],
                    poly_pts[:, 1],
                    color=edge_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_joinstyle='round',
                    solid_capstyle='round',
                    zorder=group_order[group] + 0.05,
                )
            for i, algo in enumerate(algos):
                if algo in centers:
                    main_ax.plot(*centers[algo], marker=group_markers[group], markersize=12, color=colors[i], zorder=group_order[group])

    # Set axis labels
    main_ax.set_xlabel(METRICS[metric_1], fontsize=32)
    main_ax.set_ylabel(METRICS[metric_2], fontsize=32)
    main_ax.tick_params(axis='both', which='major', labelsize=28)
    _style_main_axis_spines(main_ax)

    # === Draw quarter elliptical arcs at the lower left of the data bounds ===
    # Use the lower left corner of the current data bounding box as the center
    xlims = main_ax.get_xlim()
    ylims = main_ax.get_ylim()
    x_center, y_center = xlims[0], ylims[0]
    # Compute maximum extents in x and y directions
    rx_max = xlims[1] - x_center
    ry_max = ylims[1] - y_center

    # Number of elliptical arcs to draw (adjustable)
    num_arcs = 5  
    # Generate s values in (0,1] to scale the ellipse; s=1 corresponds to the maximum ellipse
    s_values = np.linspace(0, 1, num_arcs + 1)[1:]
    
    # Use the Magenta_7 palette from palettable for arc colors
    arc_palette = getattr(cb1, 'Amp_16')
    # Reverse the palette colors to align with cluster plotting (if desired)
    arc_colors_full = list(arc_palette.hex_colors)
    arc_colors = arc_colors_full[:num_arcs]

    # Draw elliptical arcs: parameter theta goes from 0 to 90 degrees (first quadrant)
    theta = np.linspace(0, np.pi/2, 100)
    for i, s in enumerate(s_values):
        x_arc = x_center + s * rx_max * np.cos(theta)
        y_arc = y_center + s * ry_max * np.sin(theta)
        main_ax.plot(x_arc, y_arc, linestyle='--', color=arc_colors[i], linewidth=4, zorder=0, alpha=0.8)
    
    # Create a discrete colormap for the colorbar using the selected arc colors
    arc_cmap = ListedColormap(arc_colors)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=arc_cmap, norm=norm)
    sm.set_array([])

    # Add colorbar and remove numeric tick labels, only keep the label
    cbar = main_ax.figure.colorbar(sm, ax=main_ax, orientation='vertical', pad=0.01)
    cbar.set_ticks([])  # Remove numeric labels
    cbar.set_label('Interactivity', fontsize=32)
    cbar.outline.set_visible(False)

    # Build legend
    # legend 1
    group_handles = []
    for group in group_algos:
        cmap = getattr(cb, group_colors[group])
        color = list(reversed(cmap.hex_colors))[-1]
        marker = group_markers[group]
        handle = Line2D(
            [0], [0], marker=marker, color=color, linestyle='',
            markersize=12, label=group
        )
        group_handles.append(handle)

    legend1 = main_ax.legend(
        handles=group_handles,
        loc='lower right',
        title='Algorithm Type',
        title_fontsize=20,
        fontsize=18
    )
    legend1.get_title().set_fontweight('bold')
    main_ax.add_artist(legend1)  # keep the legend 1

    # legend 2
    algo_handles = []
    for group, algos in group_algos.items():
        cmap = getattr(cb, group_colors[group])
        colors = list(reversed(cmap.hex_colors))
        marker = group_markers[group]
        for i, algo in enumerate(algos):
            if algo in centers:
                handle = Line2D(
                    [0], [0], marker=marker, color=colors[i], linestyle='',
                    markersize=10, label=ALGOS[algo]
                )
                algo_handles.append(handle)

    legend2 = main_ax.legend(
        handles=algo_handles,
        loc='upper left',
        title='Algorithm',
        title_fontsize=20,
        fontsize=17
    )
    legend2.get_title().set_fontweight('bold')

    return


def find_simulation_results(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all 'simulation_results.json' files under folders matching the given prefix.

    Args:
        base_dir (Path): The base directory path.
        prefix (str): The prefix to match folder names.

    Returns:
        Dict[List[Path]]: A list of paths to valid 'simulation_results.json' files.
    """
    json_paths_dict = defaultdict(list)
    
    # Ensure the base directory exists
    if not base_dir.exists() or not base_dir.is_dir():
        raise ValueError(f"Invalid path: {base_dir}")

    # Iterate over all entries in the base directory
    for entry in base_dir.iterdir():
        # Check if the entry is a directory and its name starts with the given prefix
        if entry.is_dir() and entry.name.startswith(args.ego):
            json_file = entry / "simulation_results.json"
            if json_file.exists():
                cbv_name = entry.name.split('-')[1]
                json_paths_dict[cbv_name].append(json_file)
            else:
                print(f"Warning: Missing {json_file.name} in {entry}")

    return json_paths_dict

def merge_multi_seed_result(json_paths: List[str], n_samples=1000):
    merged_result = []
    for json_path in json_paths:
        data_dict = load_labels_values_as_dict(json_path)
        first_dist = data_dict.get(args.first_metric, {})
        second_dist = data_dict.get(args.second_metric, {})
        merged_result.append(generate_2d_samples(first_dist, second_dist, n_samples))
    return np.vstack(merged_result)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_algo', '-ma', type=str, default='rift_pluto')
    parser.add_argument('--other_algo', '-oa', type=list, default=['rtr_pluto', 'ppo'])
    parser.add_argument('--first_metric', '-fm', type=str, default='cbv_acc_distribution')
    parser.add_argument('--second_metric', '-sm', type=str, default='cbv_speed_distribution')
    parser.add_argument('--base_dir', type=str, default='log/eval')
    parser.add_argument('--ego', type=str, default='pdm_lite')
    parser.add_argument('--recog', type=str, default='rule')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)  # Replace with your actual directory path
    other_algo_set = set(args.other_algo)
    
    # find multi-seed simulation result
    data_dict = find_simulation_results(base_dir)

    # Load data from JSON files
    sample_dict = {}
    for algo, json_paths in data_dict.items():
        sample_dict[algo] = merge_multi_seed_result(json_paths)
    # reorder the sample_dict
    sample_dict = {
        k: sample_dict[k]
        for k in [args.main_algo] + [x for x in sample_dict if x != args.main_algo]
    }

    num_plot = len(args.other_algo) + 1
    fig, axes = plt.subplots(nrows=1, ncols=num_plot, figsize=(10 * num_plot, 10))

    main_algo = args.main_algo
    main_samples = sample_dict[main_algo]
    metric_1=args.first_metric.split('_')[1]
    metric_2=args.second_metric.split('_')[1]

    plot_index = 0
    for algo, samples in sample_dict.items():
        if algo in other_algo_set:
            plot_2d_with_marginals(
                axes[plot_index],
                main_samples, samples, 
                algo_1=main_algo, algo_2=algo,
                metric_1=metric_1, metric_2=metric_2,
                cluster_centers=True
            )
            plot_index += 1

    assert plot_index == num_plot - 1, "Mismatch between number of plots and number of algorithms"
    plot_all_cluster_centers(axes[plot_index], sample_dict, metric_1, metric_2)
    
    # Save the plot to a file
    output_dir = Path('assets')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f'{metric_1}-{metric_2}.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / f'{metric_1}-{metric_2}.pdf', dpi=600, bbox_inches='tight')
    plt.show()

    plt.close()
    print(f'>> Successfully saved plot to {output_dir}')
