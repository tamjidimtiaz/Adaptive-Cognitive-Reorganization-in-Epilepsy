# Network topology and hub disruption in epilepsy with and without cognitive impairment

End-to-end pipeline for the resting-state functional connectivity analyses comparing healthy
controls (HC), patients with epilepsy and intact cognition (ECN), and patients with epilepsy and
impairment in at least one cognitive domain (ECI). Starting from preprocessed, harmonized
connectivity matrices, it runs every analysis, writes the results to `Dataset/` and draws the
figure panels in `figures/`.

## Repository layout

```
.
├── data/                 input (not tracked by git)
│   ├── epilepsy/         patient connectivity matrices
│   ├── controls/         control connectivity matrices
│   └── metadata.xlsx     patient domain scores
├── atlas/                4S256 parcellation (AtlasPack)
├── tools/
│   ├── functions.py      data loading, graph metrics, statistics, hub disruption index, plotting
│   ├── network_level.py  intra-network, inter-network and subcortical analyses (Figure 1)
│   └── hub_disruption.py hub disruption index and deviant-node analyses (Figures 2-4)
├── Dataset/              analysis results, written by the pipeline
├── run_pipeline.py       end-to-end pipeline
└── requirements.txt
```

## Installation

```bash
conda create -n ecn python=3.9
conda activate ecn
pip install -r requirements.txt
```

## Input data

Preprocessing and harmonization across scanning protocols are assumed to be done already.

- `data/epilepsy/` and `data/controls/`: one file per subject (`.npy`, or comma/space-delimited
  `.csv`/`.txt`) holding a 256 x 256 functional connectivity matrix with nodes ordered as in
  `atlas/atlas-4S256Parcels_dseg.tsv`. The subject ID is the file name up to the first underscore,
  so `sub-01_fc.npy` and `sub-01.npy` both give `sub-01`. Every file in `data/controls/` is used.
- `data/metadata.xlsx` (or `metadata.csv`): one row per patient with the columns below. Domain
  scores are binary (1 = intact, 0 = impaired).

| Column | Description |
|---|---|
| `subject_id` | matches the patient's matrix file name |
| `language` | language domain |
| `executive` | executive function domain |
| `memory` | memory domain |
| `visuospatial` | visuospatial domain |

Patients intact in all four domains form the ECN group; patients impaired in at least one form ECI.
For Figure 4, patients are also split by `language` or `memory` alone (`Lang-Neutral` /
`Lang-Impaired`, `Mem-Neutral` / `Mem-Impaired`). Patients missing any domain are left out of ECN
and ECI but still enter the domain splits where that score exists.

## Usage

```bash
python run_pipeline.py --n-jobs 8
```

Other locations can be given with `--data-dir`, `--metadata`, `--dataset-dir` and `--figure-dir`.
`--density` (default 0.30) and `--n-perm` (default 5000) set the threshold and permutation count.

## Analysis

2. **Intra-network (Figure 1A).** Node strength, betweenness centrality, clustering coefficient and
   global efficiency of the salience and limbic networks, cut from the whole-brain thresholded
   matrix. Two-sample t-tests, Bonferroni-corrected over the three group pairs.
3. **Inter-network (Figure 1B).** Mean connectivity between each pair of the seven networks.
   t-tests, Bonferroni-corrected over the three group pairs.
4. **Subcortical (Figure 1C).** Node strength within the hippocampus and thalamus. Mann-Whitney U
   tests, Bonferroni-corrected.
5. **Hub disruption index (Figure 2).** Nodal metrics are z-scored across the 256 nodes within each
   subject. K is the slope of (subject − reference) regressed on the z-scored reference profile,
   where the reference is the mean control profile. Each control is scored against the mean of the
   other controls (leave-one-out), so no subject is part of its own reference. Groups are compared
   with Welch t-tests and Bonferroni correction; effect sizes are Cohen's d.
6. **Deviant nodes (Figure 3).** Nodes whose group-mean betweenness difference from controls exceeds
   2 SD of all node differences are tested with 5000 label permutations and FDR correction.
7. **Cognitive domains (Figure 4).** K_BC and the deviant-node analysis repeated with patients split
   by the language score or the memory score alone.


## Outputs

| Figure | Results in `Dataset/` | Panels in `figures/` |
|---|---|---|
| 1A | `fig1a_intra_network_metrics.csv`, `fig1a_intra_network_stats.csv` | `fig1a_SAN`, `fig1a_LN` |
| 1B | `fig1b_inter_network_fc.csv`, `fig1b_inter_network_stats.csv` | `fig1b_inter_network` |
| 1C | `fig1c_subcortical_node_strength.csv`, `fig1c_subcortical_stats.csv` | `fig1c_subcortical` |
| 2A | `fig2a_node_strength_maps.csv` | `fig2a_*` |
| 2B | `fig2b_hdi_scatter.csv` | `fig2b_hdi_scatter` |
| 2C | `fig2c_hub_disruption_index.csv`, `fig2c_hdi_stats.csv` | `fig2c_hdi_boxplots` |
| 3 | `fig3_bc_deviant_nodes.csv` | `fig3_*` |
| 4 | `fig4_domain_kbc.csv`, `fig4_domain_kbc_stats.csv`, `fig4_domain_bc_deviant_nodes.csv` | `fig4_*` |

`atlas_nodes.csv` lists the label, network and category of each node. Figures are saved as PNG and PDF.

## Atlas

The 4S256 parcellation (Schaefer 200 cortical parcels plus subcortical, thalamic and cerebellar
parcels) is from [AtlasPack](https://github.com/PennLINC/AtlasPack).
