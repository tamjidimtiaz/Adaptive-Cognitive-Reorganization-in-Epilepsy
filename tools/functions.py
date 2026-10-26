import os
import glob
import tempfile
from itertools import combinations

import numpy as np
import pandas as pd
import nibabel as nib
import bct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MultipleLocator
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

DENSITY = 0.30
DOMAIN_COLS = ["language", "executive", "memory", "visuospatial"]
DOMAIN_SPLITS = {"Lang": "language", "Mem": "memory"}
MATRIX_EXTS = (".npy", ".csv", ".txt")
METRICS = ["Node Strength", "Betweenness Centrality",
           "Clustering Coefficient", "Global Efficiency"]
K_NAMES = dict(zip(METRICS, ["K_S", "K_BC", "K_CC", "K_GE"]))
K_LABELS = {"K_S": r"$K_s$", "K_BC": r"$K_{BC}$", "K_CC": r"$K_{CC}$", "K_GE": r"$K_{GE}$"}

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
CATEGORIES = NETWORKS + ["Subcortical", "Cerebellum"]
NET_LABELS = {"Vis": "VN", "SomMot": "SMN", "DorsAttn": "DAN", "SalVentAttn": "SAN",
              "Limbic": "LN", "Cont": "FPN", "Default": "DMN",
              "Subcortical": "Subc.", "Cerebellum": "Cereb."}
NET_COLORS = {"Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E",
              "SalVentAttn": "#C43AFA", "Limbic": "#D4A017", "Cont": "#E69422",
              "Default": "#CD3E4E"}

SUBCORTICAL = {
    "Thalamus":       [("ThalamusHCP", None)],
    "Hippocampus":    [("SubcorticalHCP", "Hippocampus")],
    "Amygdala":       [("SubcorticalHCP", "Amygdala"), ("CIT168Subcortical", r"-EXA")],
    "Striatum":       [("CIT168Subcortical", r"-Pu$|-Ca$|-NAC")],
    "GlobusPallidus": [("CIT168Subcortical", r"-GPe|-GPi")],
    "Midbrain":       [("CIT168Subcortical", r"-SNc|-SNr|-RN$|-VeP|-HN$|-HTH|-MN$|-STH")],
    "Cerebellum":     [("Cerebellum", None)],
}

GREEN, BLUE, RED = "#1A9A59", "#1B6CA8", "#C0392B"
GROUP_COLORS = {"HC": GREEN, "ECN": BLUE, "ECI": RED,
                "Lang-Neutral": BLUE, "Lang-Impaired": RED,
                "Mem-Neutral": BLUE, "Mem-Impaired": RED}
HIGHER_COLOR, LOWER_COLOR = "#174D84", "#E74C3C"


# ---------------------------------------------------------------- data

def file_subject_id(path):
    """sub-01_fc.npy -> sub-01; patient7.npy -> patient7"""
    return os.path.splitext(os.path.basename(path))[0].split("_")[0]


def matrix_files(folder):
    files = sorted(f for ext in MATRIX_EXTS for f in glob.glob(os.path.join(folder, f"*{ext}")))
    return {file_subject_id(f): f for f in files}


def read_metadata(path):
    meta = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_excel(path)
    missing = [c for c in ["subject_id"] + DOMAIN_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    meta["subject_id"] = meta["subject_id"].astype(str)
    return meta


def define_groups(meta, control_dir):
    """ECN: intact in all four domains. ECI: impaired in at least one.
    Domain groups use every patient with a score in that domain."""
    complete = meta.dropna(subset=DOMAIN_COLS)
    groups = {
        "HC": list(matrix_files(control_dir)),
        "ECN": complete.loc[(complete[DOMAIN_COLS] == 1).all(axis=1), "subject_id"].tolist(),
        "ECI": complete.loc[(complete[DOMAIN_COLS] == 0).any(axis=1), "subject_id"].tolist(),
    }
    for prefix, column in DOMAIN_SPLITS.items():
        score = pd.to_numeric(meta[column], errors="coerce")
        groups[f"{prefix}-Neutral"] = meta.loc[score == 1, "subject_id"].tolist()
        groups[f"{prefix}-Impaired"] = meta.loc[score == 0, "subject_id"].tolist()
    return groups


def load_matrix(path):
    mat = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    mat = np.nan_to_num(mat.astype(float), nan=0.0)
    np.fill_diagonal(mat, 0.0)
    return mat


def load_matrices(groups, epilepsy_dir, control_dir):
    """Load each subject once and drop subjects without a matrix from every group."""
    files = {"HC": matrix_files(control_dir), "patients": matrix_files(epilepsy_dir)}
    matrices = {}
    for group, subjects in groups.items():
        found = files["HC" if group == "HC" else "patients"]
        for sub in subjects:
            if sub in found and sub not in matrices:
                matrices[sub] = load_matrix(found[sub])
    groups = {g: list(dict.fromkeys(s for s in subs if s in matrices)) for g, subs in groups.items()}
    return matrices, groups


def read_atlas(tsv_path):
    nodes = pd.read_csv(tsv_path, sep="\t").sort_values("index").reset_index(drop=True)
    nodes["node"] = np.arange(len(nodes))
    nodes["category"] = np.where(nodes["atlas_name"] == "4S256", nodes["network_label"],
                                 np.where(nodes["atlas_name"] == "Cerebellum", "Cerebellum", "Subcortical"))
    return nodes[["node", "index", "label", "network_label", "atlas_name", "category"]]


def network_indices(nodes):
    return {net: nodes.loc[nodes["category"] == net, "node"].to_numpy() for net in NETWORKS}


def subcortical_indices(nodes):
    out = {}
    for region, parts in SUBCORTICAL.items():
        idx = []
        for atlas_name, pattern in parts:
            mask = nodes["atlas_name"] == atlas_name
            if pattern:
                mask &= nodes["label"].str.contains(pattern, regex=True)
            idx.extend(nodes.loc[mask, "node"])
        out[region] = np.array(idx)
    return out


def group_values(df, column, groups):
    return {g: df.loc[df["group"] == g, column].dropna().to_numpy() for g in groups}


def save_csv(df, folder, name):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, name), index=False)


def load_csv(folder, name):
    return pd.read_csv(os.path.join(folder, name))


# ---------------------------------------------------------------- graph metrics

def subnetwork_metrics(W, idx):
    """Metrics of a sub-network cut from a matrix already thresholded on the whole brain."""
    W = W[np.ix_(idx, idx)]
    A = bct.binarize(W)
    n = len(idx)
    return {"Node Strength": bct.strengths_und(W).mean(),
            "Betweenness Centrality": bct.betweenness_bin(A).mean() / max((n - 1) * (n - 2), 1),
            "Clustering Coefficient": bct.clustering_coef_bu(A).mean(),
            "Global Efficiency": bct.efficiency_wei(W)}


def nodal_metrics(mat, density=DENSITY):
    W = bct.threshold_proportional(mat, density)
    A = bct.binarize(W)
    n = len(A)
    return {"Node Strength": bct.strengths_und(W),
            "Betweenness Centrality": bct.betweenness_bin(A) / ((n - 1) * (n - 2)),
            "Clustering Coefficient": bct.clustering_coef_bu(A),
            "Global Efficiency": bct.efficiency_bin(A, local=True)}


def zscore(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def nodal_profiles(matrices, density=DENSITY, n_jobs=1):
    """Nodal metrics for every subject, z-scored across nodes within each subject."""
    subjects = list(matrices)
    values = Parallel(n_jobs=n_jobs)(delayed(nodal_metrics)(matrices[s], density) for s in subjects)
    return {s: {m: zscore(v[m]) for m in METRICS} for s, v in zip(subjects, values)}


# ---------------------------------------------------------------- hub disruption index

def reference_profile(profiles, controls, metric, exclude=None):
    return np.mean([profiles[s][metric] for s in controls if s != exclude], axis=0)


def hdi_slope(profile, reference):
    return stats.linregress(zscore(reference), profile - zscore(reference)).slope


def hub_disruption_index(profiles, controls, patients, metric):
    """K for each subject. Patients are compared with the mean of all controls; each control
    is compared with the mean of the remaining controls so it never enters its own reference."""
    full = reference_profile(profiles, controls, metric)
    K = {s: hdi_slope(profiles[s][metric], reference_profile(profiles, controls, metric, exclude=s))
         for s in controls}
    K.update({s: hdi_slope(profiles[s][metric], full) for s in patients})
    return K


def median_subject(df, column):
    vals = df[column].to_numpy()
    return df["subject"].iloc[np.argmin(np.abs(vals - np.median(vals)))]


# ---------------------------------------------------------------- statistics

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


TESTS = {
    "student": lambda a, b: stats.ttest_ind(a, b),
    "welch": lambda a, b: stats.ttest_ind(a, b, equal_var=False),
    "mannwhitney": lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided"),
}


def pairwise_tests(values, pairs, test="student", correction="bonferroni"):
    rows = []
    for g1, g2 in pairs:
        a, b = values[g1], values[g2]
        stat, p = TESTS[test](a, b)
        rows.append({"group_a": g1, "group_b": g2, "n_a": len(a), "n_b": len(b),
                     "mean_a": np.mean(a), "mean_b": np.mean(b),
                     "statistic": stat, "p_raw": p, "cohens_d": cohens_d(a, b)})
    out = pd.DataFrame(rows)
    out["p_corrected"] = multipletests(out["p_raw"].fillna(1.0), method=correction)[1]
    out["test"], out["correction"] = test, correction
    return out


def stars(p, four=True):
    if four and p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def permutation_test(ref, grp, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    pooled = np.vstack([ref, grp])
    n_ref = len(ref)
    observed = grp.mean(0) - ref.mean(0)
    count = np.zeros(ref.shape[1])
    for _ in range(n_perm):
        perm = rng.permutation(len(pooled))
        null = pooled[perm[n_ref:]].mean(0) - pooled[perm[:n_ref]].mean(0)
        count += np.abs(null) >= np.abs(observed)
    return observed, (count + 1) / (n_perm + 1)


def deviant_node_test(ref, grp, sd_thresh=2.0, n_perm=5000, seed=42, alpha=0.05):
    """Nodes whose group-mean difference lies beyond sd_thresh SD of all differences,
    then a label permutation test on those nodes with FDR correction."""
    ref_mean = ref.mean(0)
    diff = grp.mean(0) - ref_mean
    deviant = np.abs(diff - diff.mean()) > sd_thresh * diff.std()
    out = pd.DataFrame({"node": np.arange(ref.shape[1]), "ref_mean": ref_mean,
                        "difference": diff, "deviant": deviant,
                        "p_perm": np.nan, "p_fdr": np.nan, "significant": False, "cohens_d": np.nan})
    idx = np.where(deviant)[0]
    if len(idx):
        _, p = permutation_test(ref[:, idx], grp[:, idx], n_perm, seed)
        reject, p_fdr = multipletests(p, alpha=alpha, method="fdr_bh")[:2]
        sd = np.sqrt((ref[:, idx].var(0) + grp[:, idx].var(0)) / 2)
        out.loc[idx, "p_perm"] = p
        out.loc[idx, "p_fdr"] = p_fdr
        out.loc[idx, "significant"] = reject
        out.loc[idx, "cohens_d"] = diff[idx] / np.where(sd > 0, sd, np.inf)
    return out


# ---------------------------------------------------------------- brain images

def parcel_image(values, atlas_img, parcel_ids):
    labels = np.asarray(atlas_img.dataobj).astype(int)
    lut = np.zeros(labels.max() + 1)
    lut[np.asarray(parcel_ids, int)] = values
    return nib.Nifti1Image(lut[labels], atlas_img.affine)


def crop_white(img, pad=10):
    img = img[..., :3]
    content = ~np.all(img > 0.95, axis=2)
    rows, cols = np.where(content.any(1))[0], np.where(content.any(0))[0]
    if len(rows) == 0:
        return img
    return img[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]


def network_thumbnails(atlas_img, nodes):
    from nilearn import plotting
    labels = np.asarray(atlas_img.dataobj).astype(int)
    thumbs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for net in NETWORKS:
            mask = np.isin(labels, nodes.loc[nodes["category"] == net, "index"]).astype(float)
            r, g, b = to_rgb(NET_COLORS[net])
            cmap = LinearSegmentedColormap.from_list(net, [(r, g, b, 0), (r, g, b, 1)])
            disp = plotting.plot_glass_brain(nib.Nifti1Image(mask, atlas_img.affine),
                                             display_mode="z", colorbar=False, plot_abs=True,
                                             cmap=cmap, vmin=0, vmax=1, alpha=0.85)
            path = os.path.join(tmp, f"{net}.png")
            disp.savefig(path, dpi=150)
            disp.close()
            thumbs[net] = crop_white(mpimg.imread(path))
    return thumbs


# ---------------------------------------------------------------- plotting

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), facecolor=fig.get_facecolor())
    plt.close(fig)


def box_strip(ax, values, groups, rng, dark=False):
    for i, g in enumerate(groups):
        vals, color = values[g], GROUP_COLORS[g]
        if dark:
            bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=3.5), whiskerprops=dict(lw=2.5),
                            capprops=dict(lw=2.5), boxprops=dict(lw=2.2))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)
            ax.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals, color=color, s=35,
                       alpha=0.65, edgecolors="white", linewidths=0.4, zorder=3)
        else:
            ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                       medianprops=dict(color="white", lw=2.5),
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6, lw=1.3),
                       whiskerprops=dict(color=color, lw=1.3), capprops=dict(color=color, lw=1.8))
            ax.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals, color=color, s=32,
                       alpha=0.78, edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.55, len(groups) - 0.45)


def star_brackets(ax, pairs, groups, vmin, vmax, fontsize=20):
    """Red brackets for significant pairs only (Figure 1 style)."""
    span = vmax - vmin
    step = span * 0.18
    for level, r in enumerate(pairs.itertuples()):
        s = stars(r.p_corrected, four=False)
        if s == "ns":
            continue
        x1, x2 = groups.index(r.group_a), groups.index(r.group_b)
        y = vmax + span * 0.10 + step * level
        h = step * 0.3
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="#CC0000")
        ax.text((x1 + x2) / 2, y + h * 1.05, s, ha="center", va="bottom",
                fontsize=fontsize, color="#CC0000")
    ax.set_ylim(vmin - span * 0.08, vmax + span * (0.10 + 0.18 * len(pairs) + 0.05))


def effect_brackets(ax, pairs, groups, vmin, vmax):
    """Black brackets for every pair with Cohen's d above significant ones (Figure 2/4 style)."""
    span = vmax - vmin
    for k, r in enumerate(pairs.itertuples()):
        x1, x2 = groups.index(r.group_a), groups.index(r.group_b)
        y = vmax + span * (0.10 + k * 0.20)
        h = span * 0.025
        s = stars(r.p_corrected)
        sig = s != "ns"
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=2.5)
        if sig:
            ax.text((x1 + x2) / 2, y + h + span * 0.005, rf"$d={abs(r.cohens_d):.2f}$",
                    ha="center", va="bottom", fontsize=21, color="#333333")
        ax.text((x1 + x2) / 2, y + h - span * 0.008, s, ha="center", va="top",
                fontsize=23 if sig else 18, color="black" if sig else "#999999")
    ax.set_ylim(vmin - span * 0.08, vmax + span * (0.18 + 0.20 * len(pairs)))


def chord_plot(ax, edges, patient, thumbnails=None):
    """edges: rows with network_a, network_b, cohens_d (controls minus patients), p_corrected."""
    n = len(NETWORKS)
    angle = {net: np.pi / 2 - 2 * np.pi * i / n for i, net in enumerate(NETWORKS)}
    pos = {net: np.array([np.cos(a), np.sin(a)]) for net, a in angle.items()}
    ctrl = 0.25

    def curve(a, b, **kw):
        p0, p3 = pos[a], pos[b]
        path = Path([p0, p0 * ctrl, p3 * ctrl, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(mpatches.PathPatch(path, facecolor="none", capstyle="round", **kw))
        return (p0 + 3 * p0 * ctrl + 3 * p3 * ctrl + p3) / 8

    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="#E8E8E8", lw=1.0, zorder=0)
    sig = {(r.network_a, r.network_b): r for r in edges.itertuples() if r.p_corrected < 0.05}
    for a, b in combinations(NETWORKS, 2):
        if (a, b) not in sig:
            curve(a, b, edgecolor="#CCCCCC", lw=0.8, alpha=0.55, zorder=1)
    for (a, b), r in sig.items():
        color = "#CC0000" if r.cohens_d < 0 else GREEN
        mid = curve(a, b, edgecolor=color, lw=1.2 + 3.5 * abs(r.cohens_d), alpha=0.88, zorder=2)
        ax.text(*(mid * 1.12), stars(r.p_corrected, four=False), ha="center", va="center", fontsize=20)

    for net in NETWORKS:
        x, y = pos[net]
        if thumbnails:
            ab = AnnotationBbox(OffsetImage(thumbnails[net], zoom=0.22), (x, y), frameon=False, zorder=5)
            ax.add_artist(ab)
        else:
            ax.add_patch(plt.Circle((x, y), 0.11, facecolor=NET_COLORS[net], edgecolor="white", lw=3, zorder=5))
        ax.text(1.45 * np.cos(angle[net]), 1.45 * np.sin(angle[net]), NET_LABELS[net],
                ha="center", va="center", fontsize=18)

    ax.legend(handles=[mpatches.Patch(color="#CC0000", alpha=0.6, label=f"{patient} > Controls"),
                       mpatches.Patch(color=GREEN, alpha=0.6, label=f"{patient} < Controls")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=16)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.75, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def hdi_scatter(ax, x, y, color, title):
    fit = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, s=30, color=color, alpha=0.35, edgecolors="none")
    ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=2.2)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.text(0.97, 0.97, rf"$\mathbf{{K}}$ = {fit.slope:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=17,
            bbox=dict(facecolor="white", edgecolor="black", lw=1.5))
    ax.set_title(title, color=color, fontsize=18)
    ax.set_xlabel("<Controls> (a.u.)", fontsize=16)


def deviation_scatter(ax, table, ref, grp):
    x, y = table["ref_mean"].to_numpy(), table["difference"].to_numpy()
    sig = table["significant"].to_numpy(bool)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.3)
    ax.axhspan(min(y.min(), 0) * 1.15, 0, alpha=0.04, color=RED, zorder=0)
    ax.axhspan(0, max(y.max(), 0) * 1.15, alpha=0.04, color="#1A5276", zorder=0)
    ax.scatter(x[~sig], y[~sig], s=28, color="#CCCCCC", alpha=0.45, edgecolors="none")
    low, high = sig & (y < 0), sig & (y >= 0)
    ax.scatter(x[low], y[low], s=110, marker="v", color=RED, edgecolors="black",
               linewidths=0.8, zorder=5, label=f"Lower BC in {grp}")
    ax.scatter(x[high], y[high], s=110, marker="^", color="#1A5276", edgecolors="black",
               linewidths=0.8, zorder=5, label=f"Higher BC in {grp}")
    ax.set_xlabel(f"<{ref}>", fontsize=18)
    ax.set_ylabel(f"<{grp}> − <{ref}>", fontsize=18)
    ax.set_title(f"{ref} vs {grp}", fontsize=18)
    ax.legend(fontsize=13, frameon=True, framealpha=0.85, loc="lower left")


def node_count_bars(ax, table, title, title_color, higher_label, lower_label):
    sig = table[table["significant"].astype(bool)]
    high = [((sig["category"] == c) & (sig["difference"] >= 0)).sum() for c in CATEGORIES]
    low = [((sig["category"] == c) & (sig["difference"] < 0)).sum() for c in CATEGORIES]
    x = np.arange(len(CATEGORIES))
    w = 0.30
    ax.bar(x - w / 2 - 0.02, high, w, color=HIGHER_COLOR, alpha=0.85, edgecolor="white", label=higher_label, zorder=3)
    ax.bar(x + w / 2 + 0.02, low, w, color=LOWER_COLOR, alpha=0.85, edgecolor="white", label=lower_label, zorder=3)
    ax.axvline(len(NETWORKS) - 0.5, color="#AAAAAA", lw=1.8, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([NET_LABELS[c] for c in CATEGORIES], fontsize=16)
    ax.set_xlim(-0.7, len(CATEGORIES) - 0.3)
    ax.set_ylabel("No. of Significant\nNodes", fontsize=18)
    ax.set_title(title, color=title_color, fontsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", ls="--", lw=0.7, alpha=0.4, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", loc="upper right")


def effect_size_slices(img, vmax, black_bg=False, n_ticks=3):
    from nilearn import plotting
    bg, fg = ("black", "white") if black_bg else ("white", "black")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor=bg)
    for ax, mode in zip(axes, "xyz"):
        plotting.plot_stat_map(img, axes=ax, display_mode=mode, cut_coords=[-40, -20, 0, 20, 40],
                               cmap="RdBu_r", vmax=vmax, symmetric_cbar=True, colorbar=False,
                               draw_cross=False, annotate=False, black_bg=black_bg)
    cax = fig.add_axes([0.25, 0.03, 0.5, 0.025])
    cb = fig.colorbar(ScalarMappable(Normalize(-vmax, vmax), "RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(-vmax, vmax, n_ticks))
    cb.ax.tick_params(colors=fg, labelsize=16)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    return fig


def parcel_surface(img, title, vmin, vmax, cmap="RdYlBu"):
    from nilearn import plotting
    fig, _ = plotting.plot_img_on_surf(img, views=["lateral", "medial"], hemispheres=["left", "right"],
                                       cmap=cmap, vmin=vmin, vmax=vmax, symmetric_cbar=False,
                                       colorbar=True, cbar_tick_format="%.2f", inflate=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    return fig
