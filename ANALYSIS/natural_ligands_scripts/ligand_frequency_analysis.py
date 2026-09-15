"""Is the de-leaked screening signal real generalisation, or shared-ligand
vocabulary recognition?

The screening retrieves a pocket's bound ligand out of ~49k CCD compounds. The
bound ligands skew toward common cofactors (ATP, HEM, NAD, ...) that appear in
many training complexes paired with OTHER pockets. So even a protein-de-leaked
pocket's answer ligand may have been heavily seen in training. This script:

  1. counts each ligand's frequency in the training subset (BioLiP rows whose
     PDB id is in the DRUGCLIP training union) -> n distinct training PDBs;
  2. attaches that to each de-leaked eval pocket's bound ligand (lig_resname);
  3. stratifies retrieval metrics (top-1, EF1%, median rank) by that frequency.

If enrichment holds only for common-ligand pockets and collapses for rare/absent
ones, the surviving signal is pocket-class + shared-vocabulary recognition, not
transferable binding prediction.

De-leakage levels reuse the ladder definitions (PDB-id + receptor sequence
identity); metrics reuse screening.subset_metrics on the saved per_pocket
parquets.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / 'DATA' / 'natural_ligands'
sys.path.insert(0, str(ROOT / "scripts"))
import screening as sc  # noqa: E402

TRAIN_JSON = ROOT / "DATA" / "DRUGCLIP_BIOLIP_TRAINING.json"
BIOLIP = ROOT / "DATA" / "BioLiP.txt"
SCREEN_DIR = ROOT / "RESULTS" / "screening"
OUT_DIR = ROOT / "RESULTS" / "deleakage"
SEQ_ID_TSV = OUT_DIR / "pocket_seq_identity.tsv"
FREQ_TSV = OUT_DIR / "train_ligand_freq.tsv"

DATASETS = ["coach420", "scpdb", "pdbbind2020", "holo4k"]
LIB = "CCD"
BIN_LABELS = ["0 (absent)", "1-4", "5-24", "25-99", "100+"]


def training_union() -> set[str]:
    folds = json.loads(TRAIN_JSON.read_text())
    return {x.lower().strip() for v in folds.values() for x in v}


def lib_size() -> int:
    with h5py.File(sc.MOLS_DIR / f"{LIB}.h5", "r") as f:
        return int(f["mol_reps"].shape[0])


def train_ligand_freq() -> pd.DataFrame:
    """ligand CCD code -> (#training binding-site rows, #distinct training PDBs)."""
    if FREQ_TSV.exists():
        # keep_default_na=False so ligand codes like "NA" (sodium) stay strings.
        return pd.read_csv(FREQ_TSV, sep="\t", keep_default_na=False)
    union = training_union()
    rows: dict[str, int] = defaultdict(int)
    pdbs: dict[str, set] = defaultdict(set)
    with open(BIOLIP) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 5 or c[0].lower() not in union:
                continue
            lig = c[4].strip().upper()
            if lig:
                rows[lig] += 1
                pdbs[lig].add(c[0].lower())
    df = pd.DataFrame({
        "ligand": list(rows),
        "n_rows": [rows[k] for k in rows],
        "n_pdbs": [len(pdbs[k]) for k in rows],
    }).sort_values("n_pdbs", ascending=False)
    df.to_csv(FREQ_TSV, sep="\t", index=False)
    return df


def load_pockets() -> pd.DataFrame:
    union = training_union()
    seq = pd.read_csv(SEQ_ID_TSV, sep="\t")
    best = dict(zip(seq["pocket"], seq["best_pident"]))
    freq = train_ligand_freq().set_index("ligand")["n_pdbs"]
    frames = []
    for ds in DATASETS:
        pp = pd.read_parquet(SCREEN_DIR / f"{ds}__{LIB}__official.parquet")
        pp["dataset"] = ds
        pp["in_train"] = pp["pdb_id"].str.lower().str.strip().isin(union)
        pp["best_pident"] = pp["pocket"].map(best).fillna(0.0)
        pp["lig_freq"] = (pp["lig_resname"].str.upper().map(freq)
                          .fillna(0).astype(int))
        frames.append(pp)
    return pd.concat(frames, ignore_index=True)


def level_mask(pp: pd.DataFrame, level: str) -> pd.Series:
    if level == "L0":
        return pd.Series(True, index=pp.index)
    nt = ~pp["in_train"]
    if level == "L1":
        return nt
    if level.startswith("L2<"):
        return nt & (pp["best_pident"] < float(level[3:]))
    raise ValueError(level)


def freq_bin(n: int) -> str:
    if n == 0:
        return "0 (absent)"
    if n <= 4:
        return "1-4"
    if n <= 24:
        return "5-24"
    if n <= 99:
        return "25-99"
    return "100+"


def stratify(pp: pd.DataFrame, level: str, n_mols: int) -> pd.DataFrame:
    sub = pp[level_mask(pp, level)].copy()
    sub["bin"] = sub["lig_freq"].map(freq_bin)
    rows = []
    for b in BIN_LABELS:
        m = sc.subset_metrics(sub[sub["bin"] == b], n_mols)
        if m.get("n_in_library", 0) == 0:
            rows.append({"bin": b, "n_in_library": 0})
            continue
        rows.append({"bin": b, "n_in_library": m["n_in_library"],
                     "top1_acc": m["top1_acc"], "ef@1%": m["ef@1%"],
                     "recall@10": m["recall@10"], "median_rank": m["median_rank"]})
    out = pd.DataFrame(rows)
    out.insert(0, "level", level)
    return out


# Disjoint ligand-frequency bins (manuscript grid, row 2), finer than
# BIN_LABELS above: each pocket falls in exactly ONE bin, so a bin's metric
# isn't diluted by pooling in everything less-frequent (as a cumulative
# "freq<=t" ladder point would) — see the module-level docstring on
# deleakage_analysis.SI_BIN_ORDER for why that matters.
FREQ_BIN_ORDER = ["all", ">100", "76-100", "51-75", "26-50", "6-25", "2-5",
                  "1", "0 (unseen)"]
FREQ_BIN_LABELS = ["SI<30%\n(all freq)", ">100", "76-100", "51-75", "26-50",
                   "6-25", "2-5", "1", "0\n(unseen)"]


def freq_bin_fine(n: int) -> str:
    if n == 0:
        return "0 (unseen)"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 25:
        return "6-25"
    if n <= 50:
        return "26-50"
    if n <= 75:
        return "51-75"
    if n <= 100:
        return "76-100"
    return ">100"


def build_freq_bins_by_dataset(pp: pd.DataFrame, n_mols: int) -> pd.DataFrame:
    """Per-dataset disjoint ligand-frequency bins for the manuscript grid
    (row 2), within L2<0.3: 'all' as a reference point, then that population
    partitioned into non-overlapping frequency bins.
    """
    base = pp[level_mask(pp, "L2<0.3")].copy()
    base["freq_bin"] = base["lig_freq"].map(freq_bin_fine)
    rows = []
    for ds, sub in base.groupby("dataset"):
        rows.append({"dataset": ds, "bin": "all", **sc.subset_metrics(sub, n_mols)})
        for b in FREQ_BIN_ORDER[1:]:
            m = sc.subset_metrics(sub[sub["freq_bin"] == b], n_mols)
            rows.append({"dataset": ds, "bin": b, **m})
    return pd.DataFrame(rows)


def main() -> None:
    pp = load_pockets()
    n_mols = lib_size()
    freq = train_ligand_freq()

    print("=== top training ligands (by #distinct training PDBs) ===")
    print(freq.head(15).to_string(index=False))

    ev = pp[pp["in_library"]]
    print("\n=== bound-ligand training frequency of EVAL pockets, by level ===")
    for level in ["L0", "L1", "L2<0.3"]:
        s = ev[level_mask(ev, level)]
        share = (s["lig_freq"] >= 100).mean() if len(s) else float("nan")
        print(f"{level:>7}: n_in_lib={len(s):>5}  "
              f"median lig_freq={s['lig_freq'].median():>6.0f}  "
              f"share binding a 100+ ligand={share:6.1%}")

    print("\n=== retrieval metrics by bound-ligand training frequency (pooled) ===")
    for level in ["L1", "L2<0.3"]:
        tab = stratify(pp, level, n_mols)
        fmt = tab.copy()
        for c in ["top1_acc", "recall@10"]:
            if c in fmt:
                fmt[c] = (fmt[c] * 100).round(1)
        for c in ["ef@1%", "median_rank"]:
            if c in fmt:
                fmt[c] = fmt[c].round(1)
        print(f"\n--- {level} ---")
        print(fmt.to_string(index=False))

    # rank-correlation: does a more frequent bound ligand => better rank?
    print("\n=== Spearman( log1p(lig_freq) , gt_rank ) on in-library pockets ===")
    for level in ["L1", "L2<0.3"]:
        s = ev[level_mask(ev, level)].copy()
        s = s[s["gt_rank"].notna()]
        rho = (pd.Series(np.log1p(s["lig_freq"].to_numpy()))
               .corr(pd.Series(s["gt_rank"].astype(float).to_numpy()),
                     method="spearman"))
        print(f"{level:>7}: n={len(s):>5}  rho={rho:+.3f}  "
              "(negative => frequent ligand ranks better)")


if __name__ == "__main__":
    main()
