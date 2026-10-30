"""Summary statistics for the random surface-patch negative control.

Addresses the reviewer request for (i) patch size distributions and
(ii) overlap statistics versus the true ligand-binding sites.

Reads only artefacts already produced by ``pack_patches_lmdb.py`` (no
re-segmentation):

  * ``RESULTS/<ds>/lmdb/<ds>_patches_manifest.tsv``  — n_residues / n_atoms
  * ``RESULTS/<ds>/lmdb/<ds>_patches.lmdb``          — patch heavy-atom XYZ
  * ``reference/DATA/<raw>/<complex_id>/ligand_*.pdb`` — bound ligands
  * NATURAL_LIGANDS observed-pocket manifests (n_pocket_atoms) for reference

For every patch it computes the minimum heavy-atom distance between the
patch and any bound ligand of its complex, and the number of patch
residues that fall inside the observed site (any heavy atom ≤ 6 Å of a
ligand heavy atom — the same definition used to exclude the site before
patch growth, so this must be 0).

Statistics are reported for the screening population used in the paper
(nb52): pockets cross-dataset deduplicated on (pdb_id, lig_resname) with
priority COACH420 > scPDB > PDBbind > HOLO4K, restricted to valid ligands
present in the CCD library. Its complexes carry 252,222 patches. The same
table over all 277,306 generated patches is written alongside.

Outputs (``RESULTS/patch_qc/``):
  * ``patch_qc_per_patch.tsv``   — one row per patch (all generated patches)
  * ``patch_qc_summary.tsv``     — supplementary table, screening population
  * ``patch_qc_summary_all.tsv`` — same, all generated patches
  * ``fig_patch_qc.png/.pdf``    — size + distance distributions (screening population)

CLI:
    python scripts/patch_qc_stats.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_patches as sp

PROJECT = Path(__file__).resolve().parents[2] / 'DATA' / 'drugclip'
NAT_LIG = Path(__file__).resolve().parents[2] / "DATA" / "natural_ligands" / "RESULTS" / "pockets"
OUT_DIR = PROJECT / "RESULTS" / "patch_qc"

DEDUP_PRIORITY = ["coach420", "scpdb", "pdbbind2020", "holo4k"]  # nb49 / nb52 order

# dataset -> (raw dir, observed-pocket manifest, display label)
DATASETS = {
    "coach420":    ("reference/DATA/coach420/raw",    "coach420_pocket_manifest.tsv",    "COACH420"),
    "holo4k":      ("reference/DATA/holo4k/raw",      "holo4k_pocket_manifest.tsv",      "HOLO4K"),
    "scpdb":       ("reference/DATA/sc-pdb/raw",      "sc-pdb_pocket_manifest.tsv",      "scPDB"),
    "pdbbind2020": ("reference/DATA/pdbbind2020/raw", "pdbbind2020_pocket_manifest.tsv", "PDBbind"),
}
LIGAND_CUTOFF = 6.0   # Å, observed-site definition used in pack_patches_lmdb.py
ADJ_CUTOFF = 8.0      # Å, "adjacent to the site" for reporting


def _patch_residue_ids(atom_names, coords):
    """Split a patch's flat atom list into residues (each residue starts at 'N')."""
    starts = [i for i, a in enumerate(atom_names) if a == "N"] + [len(atom_names)]
    if starts[0] != 0:
        starts = [0] + starts
    return [(s, e) for s, e in zip(starts[:-1], starts[1:]) if e > s]


def per_patch_stats(ds: str) -> pd.DataFrame:
    raw_dir, _, _ = DATASETS[ds]
    raw_dir = PROJECT / raw_dir
    man = pd.read_csv(PROJECT / f"RESULTS/{ds}/lmdb/{ds}_patches_manifest.tsv", sep="\t")
    env = lmdb.open(str(PROJECT / f"RESULTS/{ds}/lmdb/{ds}_patches.lmdb"),
                    subdir=False, readonly=True, lock=False)

    lig_cache: dict = {}
    rows = []
    with env.begin() as txn:
        for r in tqdm(man.itertuples(index=False), total=len(man), desc=ds):
            cid = r.complex_id
            if cid not in lig_cache:
                ligs = sorted((raw_dir / cid).glob("ligand_*.pdb"))
                xyz = [sp.load_ligand_atoms(p) for p in ligs]
                xyz = [x for x in xyz if x.size]
                lig_cache = {cid: cKDTree(np.vstack(xyz)) if xyz else None}
            tree = lig_cache[cid]

            rec = pickle.loads(txn.get(str(r.key).encode()))
            names = rec["pocket_atoms"]
            xyz = np.asarray(rec["pocket_coordinates"], dtype=np.float64)
            heavy = np.array([not n.startswith("H") for n in names])
            names = [n for n, h in zip(names, heavy) if h]
            xyz = xyz[heavy]

            if tree is None:
                dmin = dcen = np.nan
                n_overlap = 0
            else:
                d, _ = tree.query(xyz, k=1)
                dmin = float(d.min())
                dcen = float(tree.query(xyz.mean(0), k=1)[0])
                n_overlap = sum(
                    int(d[s:e].min() <= LIGAND_CUTOFF)
                    for s, e in _patch_residue_ids(names, xyz)
                )
            rows.append({
                "dataset": ds, "complex_id": cid, "identifier": r.identifier,
                "n_residues": r.n_residues, "n_atoms": r.n_atoms,
                "n_atoms_lmdb": len(names),
                "min_dist_to_ligand": dmin,
                "centroid_dist_to_ligand": dcen,
                "n_residues_in_site": n_overlap,
            })
    return pd.DataFrame(rows)


def screening_population() -> pd.DataFrame:
    """Pockets of the nb52 pooled screen: dedup on (pdb_id, lig_resname) by
    DEDUP_PRIORITY, then valid & in_library. Returns dataset/pocket/entry."""
    raw = pd.concat(
        [pd.read_parquet(PROJECT / f"RESULTS/benchmark/{ds}_patches_ccd_screen.parquet")
         for ds in DATASETS],
        ignore_index=True,
    )
    keep = (raw.groupby(["pdb_id", "lig_resname"])["dataset"]
               .agg(lambda s: next(d for d in DEDUP_PRIORITY if d in set(s)))
               .rename("keep"))
    raw = raw.merge(keep, left_on=["pdb_id", "lig_resname"], right_index=True)
    pop = raw[(raw.dataset == raw.keep) & raw.valid & raw.in_library]
    out = []
    for ds, (_, mf, _) in DATASETS.items():
        man = pd.read_csv(NAT_LIG / mf, sep="\t").set_index("pocket")
        sub = pop.loc[pop.dataset == ds, ["dataset", "pocket_id"]].copy()
        sub["entry"] = man.loc[sub.pocket_id, "entry"].to_numpy()
        sub["n_pocket_atoms"] = man.loc[sub.pocket_id, "n_pocket_atoms"].to_numpy()
        sub["ligand_file"] = man.loc[sub.pocket_id, "ligand_file"].to_numpy()
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def observed_site_residues(pop: pd.DataFrame) -> pd.DataFrame:
    """Residue count of each observed site: standard amino acids with any heavy
    atom ≤ 6 Å of a heavy atom of that pocket's ligand (the DrugCLIP pocket
    definition used for n_pocket_atoms). Cached to observed_site_residues.tsv."""
    from Bio.PDB.Polypeptide import is_aa

    cache = OUT_DIR / "observed_site_residues.tsv"
    if cache.exists():
        return pd.read_csv(cache, sep="\t")
    rows = []
    for (ds, entry), grp in tqdm(pop.groupby(["dataset", "entry"]), desc="observed sites"):
        cx_dir = PROJECT / DATASETS[ds][0] / entry
        residues, _ = sp.load_structure_residues(cx_dir / "protein.pdb")
        residues = {k: r for k, r in residues.items() if is_aa(r.name, standard=True)}
        for r in grp.itertuples(index=False):
            lig = sp.load_ligand_atoms(cx_dir / r.ligand_file)
            hits = sp.residues_within_ligand(residues, lig, cutoff=LIGAND_CUTOFF)
            rows.append({
                "dataset": ds, "pocket_id": r.pocket_id,
                "n_site_residues": len(hits),
                "n_site_atoms_recomputed": sum(len(residues[k].heavy_atoms) for k in hits),
                "n_pocket_atoms": r.n_pocket_atoms,
            })
    out = pd.DataFrame(rows)
    out.to_csv(cache, sep="\t", index=False)
    return out


def _q(s: pd.Series) -> str:
    q1, m, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{m:.0f} [{q1:.0f}–{q3:.0f}]"


def _qf(s: pd.Series) -> str:
    q1, m, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{m:.1f} [{q1:.1f}–{q3:.1f}]"


def summarise(df: pd.DataFrame, label: str, pockets: pd.Series,
              site_res: pd.Series | None = None) -> dict:
    per_cx = df.groupby(["dataset", "complex_id"]).size()
    out = {
        "Dataset": label,
        "Complexes": len(per_cx),
        "Patches": len(df),
        "Patches / complex, median [IQR]": _q(per_cx),
        "Heavy atoms / patch, median [IQR]": _q(df.n_atoms),
        "Heavy atoms / patch, min–max": f"{df.n_atoms.min()}–{df.n_atoms.max()}",
        "Residues / patch, median [IQR]": _q(df.n_residues),
        "Residues / patch, min–max": f"{df.n_residues.min()}–{df.n_residues.max()}",
        "Patches < 40 atoms (%)": f"{100 * (df.n_atoms < 40).mean():.2f}",
        "Patches > 128 atoms (%)": f"{100 * (df.n_atoms > 128).mean():.1f}",
        "Observed-site heavy atoms, median [IQR]": _q(pockets),
        "Patch residues within 6 Å of ligand": int(df.n_residues_in_site.sum()),
        "Min. patch–ligand distance (Å), median [IQR]": _qf(df.min_dist_to_ligand),
        "Min. patch–ligand distance (Å), min": f"{df.min_dist_to_ligand.min():.2f}",
        f"Patches ≤ {ADJ_CUTOFF:.0f} Å from ligand (%)":
            f"{100 * (df.min_dist_to_ligand <= ADJ_CUTOFF).mean():.1f}",
        "Patch centroid–ligand distance (Å), median [IQR]": _qf(df.centroid_dist_to_ligand),
    }
    if site_res is not None:
        out["Observed-site residues, median [IQR]"] = _q(site_res)
    return out


def plot(df: pd.DataFrame, pockets: dict, out: Path) -> None:
    order = list(DATASETS)
    labels = {k: v[2] for k, v in DATASETS.items()}
    colours = dict(zip(order, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    bins = np.arange(0, 520, 10)
    for ds in order:
        ax[0].hist(df.loc[df.dataset == ds, "n_atoms"], bins=bins, density=True,
                   histtype="step", lw=1.4, color=colours[ds], label=labels[ds])
    allp = pd.concat(pockets.values())
    ax[0].hist(allp, bins=bins, density=True, histtype="stepfilled", alpha=0.2,
               color="k", label="observed sites (pooled)")
    for x, ls in [(40, ":"), (128, "--"), (511, "-.")]:
        ax[0].axvline(x, c="grey", ls=ls, lw=0.8)
    ax[0].set_xlabel("heavy atoms per patch"); ax[0].set_ylabel("density")
    ax[0].set_title("a  Patch size (atoms)", loc="left")
    ax[0].legend(fontsize=8, frameon=False)

    rbins = np.arange(0, df.n_residues.max() + 2, 1)
    for ds in order:
        ax[1].hist(df.loc[df.dataset == ds, "n_residues"], bins=rbins, density=True,
                   histtype="step", lw=1.4, color=colours[ds], label=labels[ds])
    ax[1].set_xlabel("residues per patch"); ax[1].set_ylabel("density")
    ax[1].set_title("b  Patch size (residues)", loc="left")

    for ds in order:
        d = np.sort(df.loc[df.dataset == ds, "min_dist_to_ligand"].dropna())
        ax[2].plot(d, np.arange(1, len(d) + 1) / len(d), color=colours[ds], lw=1.4,
                   label=labels[ds])
    ax[2].axvline(LIGAND_CUTOFF, c="k", ls="--", lw=0.8)
    ax[2].text(LIGAND_CUTOFF + 0.5, 0.05, f"site exclusion ({LIGAND_CUTOFF:.0f} Å)", fontsize=8)
    ax[2].set_xlim(0, 60)
    ax[2].set_xlabel("min. heavy-atom distance patch → bound ligand (Å)")
    ax[2].set_ylabel("cumulative fraction of patches")
    ax[2].set_title("c  Separation from ligand site", loc="left")

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_patch_path = OUT_DIR / "patch_qc_per_patch.tsv"
    if per_patch_path.exists():
        df = pd.read_csv(per_patch_path, sep="\t")
    else:
        df = pd.concat([per_patch_stats(ds) for ds in DATASETS], ignore_index=True)
        df.to_csv(per_patch_path, sep="\t", index=False)

    mismatch = (df.n_atoms != df.n_atoms_lmdb).sum()
    print(f"manifest vs LMDB atom-count mismatches: {mismatch}")

    pop = screening_population()
    entries = set(zip(pop.dataset, pop.entry))
    in_pop = pd.Series(list(zip(df.dataset, df.complex_id))).isin(entries).to_numpy()

    all_pockets = {
        ds: pd.read_csv(NAT_LIG / mf, sep="\t")["n_pocket_atoms"]
        for ds, (_, mf, _) in DATASETS.items()
    }
    pop_pockets = {ds: pop.loc[pop.dataset == ds, "n_pocket_atoms"] for ds in DATASETS}

    site = observed_site_residues(pop)
    site_res = {ds: site.loc[site.dataset == ds, "n_site_residues"] for ds in DATASETS}

    for tag, sub, pockets, sres in [("", df[in_pop], pop_pockets, site_res),
                                    ("_all", df, all_pockets, None)]:
        rows = [summarise(sub[sub.dataset == ds], DATASETS[ds][2], pockets[ds],
                          sres[ds] if sres else None) for ds in DATASETS]
        rows.append(summarise(sub, "Pooled", pd.concat(pockets.values()),
                              site.n_site_residues if sres else None))
        summary = pd.DataFrame(rows)
        summary.insert(2, "Ligand pockets", [len(pockets[ds]) for ds in DATASETS]
                       + [sum(len(v) for v in pockets.values())])
        summary.set_index("Dataset").T.rename_axis("Statistic").to_csv(
            OUT_DIR / f"patch_qc_summary{tag}.tsv", sep="\t")
        print(f"\n=== patch_qc_summary{tag}.tsv")
        print(summary.set_index("Dataset").T.to_string())

    plot(df[in_pop], pop_pockets, OUT_DIR / "fig_patch_qc")


if __name__ == "__main__":
    main()
