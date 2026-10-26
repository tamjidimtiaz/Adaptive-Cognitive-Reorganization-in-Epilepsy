"""Figures 2-4: hub disruption index (K) and deviant betweenness-centrality nodes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import functions as fn

BC = "Betweenness Centrality"
MAIN = ["HC", "ECN", "ECI"]
DOMAINS = {"Language": ["HC", "Lang-Neutral", "Lang-Impaired"],
           "Memory": ["HC", "Mem-Neutral", "Mem-Impaired"]}


def hdi_pairs(groups):
    hc, neutral, impaired = groups
    return [(hc, neutral), (hc, impaired), (impaired, neutral)]


def bc_deviation(profiles, nodes, groups, patient_groups, n_perm):
    """HC against each patient group; permutation seeds follow the comparison order."""
    hc = np.vstack([profiles[s][BC] for s in groups["HC"]])
    tables = []
    for seed, g in enumerate(patient_groups, start=42):
        grp = np.vstack([profiles[s][BC] for s in groups[g]])
        table = fn.deviant_node_test(hc, grp, n_perm=n_perm, seed=seed)
        tables.append(nodes[["node", "label", "category"]].merge(table, on="node")
                      .assign(reference="HC", group=g))
    return pd.concat(tables, ignore_index=True)


def run(matrices, groups, nodes, dataset_dir, density=fn.DENSITY, n_perm=5000, n_jobs=1):
    needed = list(dict.fromkeys(s for g in MAIN + sum(DOMAINS.values(), []) for s in groups[g]))
    profiles = fn.nodal_profiles({s: matrices[s] for s in needed}, density, n_jobs)
    controls = groups["HC"]

    # Figure 2C
    patients = groups["ECN"] + groups["ECI"]
    K = {fn.K_NAMES[m]: fn.hub_disruption_index(profiles, controls, patients, m) for m in fn.METRICS}
    hdi = pd.DataFrame([{"subject": s, "group": g, **{k: K[k][s] for k in K}}
                        for g in MAIN for s in groups[g]])
    hdi_stats = pd.concat([fn.pairwise_tests(fn.group_values(hdi, k, MAIN), hdi_pairs(MAIN), "welch", "fdr_bh")
                           .assign(index=k) for k in K])
    fn.save_csv(hdi, dataset_dir, "fig2c_hub_disruption_index.csv")
    fn.save_csv(hdi_stats[["index"] + [c for c in hdi_stats if c != "index"]], dataset_dir, "fig2c_hdi_stats.csv")

    # Figure 2A/2B: subject closest to the group median K_S
    strength = "Node Strength"
    rep = {g: fn.median_subject(hdi[hdi["group"] == g], "K_S") for g in MAIN}
    maps = nodes[["node", "index", "label"]].assign(
        HC_average=fn.reference_profile(profiles, controls, strength),
        **{g: profiles[rep[g]][strength] for g in MAIN})
    fn.save_csv(maps, dataset_dir, "fig2a_node_strength_maps.csv")
    scatter = []
    for g in MAIN:
        ref = fn.reference_profile(profiles, controls, strength, exclude=rep[g] if g == "HC" else None)
        scatter.append(nodes[["node", "label"]].assign(
            group=g, subject=rep[g], controls_z=fn.zscore(ref), deviation=profiles[rep[g]][strength] - ref))
    fn.save_csv(pd.concat(scatter), dataset_dir, "fig2b_hdi_scatter.csv")

    # Figure 3
    fn.save_csv(bc_deviation(profiles, nodes, groups, MAIN[1:], n_perm), dataset_dir, "fig3_bc_deviant_nodes.csv")

    # Figure 4
    k_rows, k_stats, dev = [], [], []
    for domain, labels in DOMAINS.items():
        k_bc = fn.hub_disruption_index(profiles, controls, groups[labels[1]] + groups[labels[2]], BC)
        df = pd.DataFrame([{"domain": domain, "subject": s, "group": g, "K_BC": k_bc[s]}
                           for g in labels for s in groups[g]])
        k_rows.append(df)
        k_stats.append(fn.pairwise_tests(fn.group_values(df, "K_BC", labels), hdi_pairs(labels), "welch", "fdr_bh")
                       .assign(domain=domain, index="K_BC"))
        dev.append(bc_deviation(profiles, nodes, groups, labels[1:], n_perm).assign(domain=domain))
    k_stats = pd.concat(k_stats)
    fn.save_csv(pd.concat(k_rows), dataset_dir, "fig4_domain_kbc.csv")
    fn.save_csv(k_stats[["domain", "index"] + [c for c in k_stats if c not in ("domain", "index")]],
                dataset_dir, "fig4_domain_kbc_stats.csv")
    fn.save_csv(pd.concat(dev), dataset_dir, "fig4_domain_bc_deviant_nodes.csv")

    for r in hdi_stats.itertuples():
        print(f"  {r.index:5s} {r.group_a} vs {r.group_b}: d={r.cohens_d:.2f}  p_fdr={r.p_corrected:.4f}")


def k_boxes(axes, df, stats, labels, columns, rng):
    for ax, col in zip(axes, columns):
        values = fn.group_values(df, col, labels)
        fn.box_strip(ax, values, labels, rng, dark=True)
        pairs = stats[stats["index"] == col]
        fn.effect_brackets(ax, pairs, labels, df[col].min(), df[col].max())
        ax.set_xticklabels([g.replace("-", "-\n") for g in labels], fontsize=20)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel(fn.K_LABELS[col], fontsize=25)
        ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.5)


def deviation_maps(table, atlas_img, nodes, figure_dir, prefix, black_bg):
    for grp, sub in table.groupby("group", sort=False):
        d = sub["cohens_d"].where(sub["significant"].astype(bool), 0.0).fillna(0.0).to_numpy()
        if not np.any(d):
            continue
        vmax = np.abs(d).max()
        fig = fn.effect_size_slices(fn.parcel_image(d, atlas_img, nodes["index"]), vmax,
                                    black_bg=black_bg, n_ticks=5 if black_bg else 3)
        fn.save_figure(fig, figure_dir, f"{prefix}_HC_vs_{grp}_map")


def plot(dataset_dir, figure_dir, atlas_img=None, nodes=None):
    rng = np.random.default_rng(42)

    maps = fn.load_csv(dataset_dir, "fig2a_node_strength_maps.csv")
    if atlas_img is not None:
        titles = {"HC_average": "Reference: Average Controls", "HC": "Control Subject",
                  "ECN": "ECN Subject", "ECI": "ECI Subject"}
        for col, title in titles.items():
            v = maps[col].to_numpy()
            fig = fn.parcel_surface(fn.parcel_image(v, atlas_img, maps["index"]), title, v.min(), v.max())
            fn.save_figure(fig, figure_dir, f"fig2a_{col}")

    scatter = fn.load_csv(dataset_dir, "fig2b_hdi_scatter.csv")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, g in zip(axes, MAIN):
        s = scatter[scatter["group"] == g]
        name = "Control" if g == "HC" else g
        fn.hdi_scatter(ax, s["controls_z"].to_numpy(), s["deviation"].to_numpy(),
                       fn.GROUP_COLORS[g], f"{name} Subject")
        ax.tick_params(labelsize=14)
    axes[0].set_ylabel("Subject − <Controls> (a.u)", fontsize=16)
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig2b_hdi_scatter")

    hdi = fn.load_csv(dataset_dir, "fig2c_hub_disruption_index.csv")
    hdi_stats = fn.load_csv(dataset_dir, "fig2c_hdi_stats.csv")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    k_boxes(axes, hdi, hdi_stats, MAIN, list(fn.K_NAMES.values()), rng)
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig2c_hdi_boxplots")

    dev = fn.load_csv(dataset_dir, "fig3_bc_deviant_nodes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (grp, sub) in zip(axes, dev.groupby("group", sort=False)):
        fn.deviation_scatter(ax, sub, "HC", grp)
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig3_bc_deviation_scatter")
    fig, axes = plt.subplots(len(MAIN) - 1, 1, figsize=(11, 12))
    for ax, (grp, sub) in zip(axes, dev.groupby("group", sort=False)):
        fn.node_count_bars(ax, sub, f"HC vs {grp}", fn.GROUP_COLORS[grp], "Higher BC", "Lower BC")
    fig.tight_layout()
    fn.save_figure(fig, figure_dir, "fig3_bc_node_counts")
    if atlas_img is not None:
        deviation_maps(dev, atlas_img, nodes, figure_dir, "fig3", black_bg=False)

    kbc = fn.load_csv(dataset_dir, "fig4_domain_kbc.csv")
    kbc_stats = fn.load_csv(dataset_dir, "fig4_domain_kbc_stats.csv")
    dev = fn.load_csv(dataset_dir, "fig4_domain_bc_deviant_nodes.csv")
    for domain, labels in DOMAINS.items():
        tag = f"fig4_{domain.lower()}"
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        k_boxes([ax], kbc[kbc["domain"] == domain], kbc_stats[kbc_stats["domain"] == domain],
                labels, ["K_BC"], rng)
        fig.tight_layout()
        fn.save_figure(fig, figure_dir, f"{tag}_kbc")

        table = dev[dev["domain"] == domain]
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        for ax, (grp, sub) in zip(axes, table.groupby("group", sort=False)):
            fn.node_count_bars(ax, sub, f"HC vs {grp}", fn.GROUP_COLORS[grp],
                               f"Higher Centrality in {grp}", f"Lower Centrality in {grp}")
        fig.tight_layout()
        fn.save_figure(fig, figure_dir, f"{tag}_node_counts")
        if atlas_img is not None:
            deviation_maps(table, atlas_img, nodes, figure_dir, tag, black_bg=True)
