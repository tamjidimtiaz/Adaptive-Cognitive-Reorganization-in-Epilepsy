"""Figure 1: intra-network topology (SAN, LN), inter-network FC and subcortical node strength."""
from itertools import combinations

import numpy as np
import pandas as pd
import bct
import matplotlib.pyplot as plt

from tools import functions as fn

GROUPS = ["HC", "ECN", "ECI"]
PAIRS = [("HC", "ECN"), ("HC", "ECI"), ("ECN", "ECI")]
INTRA_NETWORKS = {"SalVentAttn": "Salience Network", "Limbic": "Limbic Network"}
SUBCORTICAL_REGIONS = ["Hippocampus", "Thalamus"]
EDGES = [(fn.NET_LABELS[a], fn.NET_LABELS[b]) for a, b in combinations(fn.NETWORKS, 2)]


def compute(matrices, groups, nodes, density=fn.DENSITY):
    net_idx = fn.network_indices(nodes)
    sub_idx = fn.subcortical_indices(nodes)
    intra, inter, subc = [], [], []
    for group in GROUPS:
        for sub in groups[group]:
            W = bct.threshold_proportional(matrices[sub], density)
            for net in INTRA_NETWORKS:
                intra.append({"subject": sub, "group": group, "network": fn.NET_LABELS[net],
                              **fn.subnetwork_metrics(W, net_idx[net])})
            row = {"subject": sub, "group": group}
            for a, b in combinations(fn.NETWORKS, 2):
                row[f"{fn.NET_LABELS[a]}-{fn.NET_LABELS[b]}"] = W[np.ix_(net_idx[a], net_idx[b])].mean()
            inter.append(row)
            subc.append({"subject": sub, "group": group,
                         **{reg: max(bct.strengths_und(W[np.ix_(sub_idx[reg], sub_idx[reg])]).mean(), 0.0)
                            for reg in SUBCORTICAL_REGIONS}})
    return pd.DataFrame(intra), pd.DataFrame(inter), pd.DataFrame(subc)


def statistics(intra, inter, subc):
    intra_stats = pd.concat([
        fn.pairwise_tests(fn.group_values(df, metric, GROUPS), PAIRS, "student", "bonferroni")
          .assign(network=net, metric=metric)
        for net, df in intra.groupby("network", sort=False) for metric in fn.METRICS])
    inter_stats = pd.concat([
        fn.pairwise_tests(fn.group_values(inter, f"{a}-{b}", GROUPS), PAIRS, "student", "bonferroni")
          .assign(network_a=a, network_b=b)
        for a, b in EDGES])
    subc_stats = pd.concat([
        fn.pairwise_tests(fn.group_values(subc, reg, GROUPS), PAIRS, "mannwhitney", "bonferroni")
          .assign(region=reg, metric="Node Strength")
        for reg in SUBCORTICAL_REGIONS])
    first = lambda df, cols: df[cols + [c for c in df.columns if c not in cols]]
    return (first(intra_stats, ["network", "metric"]),
            first(inter_stats, ["network_a", "network_b"]),
            first(subc_stats, ["region", "metric"]))


def run(matrices, groups, nodes, dataset_dir, density=fn.DENSITY):
    intra, inter, subc = compute(matrices, groups, nodes, density)
    intra_stats, inter_stats, subc_stats = statistics(intra, inter, subc)
    for df, name in [(intra, "fig1a_intra_network_metrics.csv"),
                     (intra_stats, "fig1a_intra_network_stats.csv"),
                     (inter, "fig1b_inter_network_fc.csv"),
                     (inter_stats, "fig1b_inter_network_stats.csv"),
                     (subc, "fig1c_subcortical_node_strength.csv"),
                     (subc_stats, "fig1c_subcortical_stats.csv")]:
        fn.save_csv(df, dataset_dir, name)

    sig = inter_stats[inter_stats["p_corrected"] < 0.05]
    for r in sig.itertuples():
        print(f"  inter-network {r.network_a}-{r.network_b}: {r.group_a} vs {r.group_b}  "
              f"d={r.cohens_d:.2f}  p_fdr={r.p_corrected:.4f}")


def plot(dataset_dir, figure_dir, atlas_img=None, nodes=None):
    rng = np.random.default_rng(42)
    intra = fn.load_csv(dataset_dir, "fig1a_intra_network_metrics.csv")
    intra_stats = fn.load_csv(dataset_dir, "fig1a_intra_network_stats.csv")
    for net, title in INTRA_NETWORKS.items():
        label = fn.NET_LABELS[net]
        df = intra[intra["network"] == label]
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        for ax, metric in zip(axes.ravel(), fn.METRICS):
            values = fn.group_values(df, metric, GROUPS)
            fn.box_strip(ax, values, GROUPS, rng)
            pairs = intra_stats[(intra_stats["network"] == label) & (intra_stats["metric"] == metric)]
            fn.star_brackets(ax, pairs, GROUPS, df[metric].min(), df[metric].max())
            ax.set_xticklabels(GROUPS, fontsize=17)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_title(metric, fontsize=18)
            ax.set_ylabel(f"{metric} (a.u.)", fontsize=17)
            ax.grid(axis="y", ls="--", lw=0.6, alpha=0.4, color="#BBBBBB")
            ax.set_axisbelow(True)
        fig.suptitle(title, fontsize=24, fontweight="bold")
        fig.tight_layout()
        fn.save_figure(fig, figure_dir, f"fig1a_{label}")

    inter_stats = fn.load_csv(dataset_dir, "fig1b_inter_network_stats.csv")
    to_key = {v: k for k, v in fn.NET_LABELS.items()}
    inter_stats["network_a"] = inter_stats["network_a"].map(to_key)
    inter_stats["network_b"] = inter_stats["network_b"].map(to_key)
    thumbs = fn.network_thumbnails(atlas_img, nodes) if atlas_img is not None else None
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, patient in zip(axes, ["ECN", "ECI"]):
        edges = inter_stats[(inter_stats["group_a"] == "HC") & (inter_stats["group_b"] == patient)]
        fn.chord_plot(ax, edges, patient, thumbs)
        ax.set_title(f"Controls vs {patient}", fontsize=24)
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig1b_inter_network")

    subc = fn.load_csv(dataset_dir, "fig1c_subcortical_node_strength.csv")
    subc_stats = fn.load_csv(dataset_dir, "fig1c_subcortical_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, reg in zip(axes, SUBCORTICAL_REGIONS):
        fn.box_strip(ax, fn.group_values(subc, reg, GROUPS), GROUPS, rng)
        fn.star_brackets(ax, subc_stats[subc_stats["region"] == reg], GROUPS, subc[reg].min(), subc[reg].max())
        ax.set_xticklabels(GROUPS, fontsize=17)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_title("Node Strength", fontsize=18)
        ax.set_ylabel(f"{reg}\nNode Strength (a.u.)", fontsize=17)
        ax.grid(axis="y", ls="--", lw=0.6, alpha=0.4, color="#BBBBBB")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig1c_subcortical")
