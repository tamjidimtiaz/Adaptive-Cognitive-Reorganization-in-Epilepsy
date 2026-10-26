import os
import glob
import argparse

import nibabel as nib

from tools import functions as fn
from tools import network_level, hub_disruption

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end network topology and hub disruption analysis.")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                   help="folder holding epilepsy/, controls/ and metadata.xlsx (or metadata.csv)")
    p.add_argument("--metadata", help="patient metadata file; defaults to <data-dir>/metadata.xlsx or .csv")
    p.add_argument("--atlas-dir", default=os.path.join(HERE, "atlas"))
    p.add_argument("--dataset-dir", default=os.path.join(HERE, "Dataset"))
    p.add_argument("--figure-dir", default=os.path.join(HERE, "figures"))
    p.add_argument("--density", type=float, default=fn.DENSITY)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def find_metadata(data_dir):
    hits = sorted(glob.glob(os.path.join(data_dir, "metadata.xlsx")) + glob.glob(os.path.join(data_dir, "metadata.csv")))
    if not hits:
        raise SystemExit(f"No metadata.xlsx or metadata.csv in {data_dir}; pass --metadata")
    return hits[0]


def main():
    args = parse_args()
    epilepsy_dir = os.path.join(args.data_dir, "epilepsy")
    control_dir = os.path.join(args.data_dir, "controls")
    metadata = args.metadata or find_metadata(args.data_dir)
    atlas_tsv = os.path.join(args.atlas_dir, "atlas-4S256Parcels_dseg.tsv")
    atlas_nii = os.path.join(args.atlas_dir, "tpl-MNI152NLin2009cAsym_atlas-4S256Parcels_res-02_dseg.nii.gz")
    fn.set_style()

    meta = fn.read_metadata(metadata)
    groups = fn.define_groups(meta, control_dir)
    matrices, groups = fn.load_matrices(groups, epilepsy_dir, control_dir)
    for g, subs in groups.items():
        print(f"{g:<14s} n = {len(subs)}")
    if len(groups["HC"]) < 3 or not groups["ECN"] or not groups["ECI"]:
        raise SystemExit("Need at least 3 controls and one patient in each of ECN and ECI; check file names against metadata subject_id")

    nodes = fn.read_atlas(atlas_tsv)
    fn.save_csv(nodes, args.dataset_dir, "atlas_nodes.csv")

    print("Network-level analysis (Figure 1)")
    network_level.run(matrices, groups, nodes, args.dataset_dir, args.density)
    print("Hub disruption analysis (Figures 2-4)")
    hub_disruption.run(matrices, groups, nodes, args.dataset_dir, args.density, args.n_perm, args.n_jobs)

    print("Plotting")
    atlas_img = nib.load(atlas_nii)
    network_level.plot(args.dataset_dir, args.figure_dir, atlas_img, nodes)
    hub_disruption.plot(args.dataset_dir, args.figure_dir, atlas_img, nodes)
    print(f"Results in {args.dataset_dir}, figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
