"""Refine the novelty test: instead of exact-CCD-code absence, measure how
chemically novel each de-leaked pocket's bound ligand is relative to the
training ligand set, via ECFP4 Tanimoto to its NEAREST training ligand.

A ligand with a novel CCD code but Tanimoto ~0.95 to a training ligand is, to
the model, seen chemistry; only low-Tanimoto bound ligands are genuinely novel
scaffolds. Stratifying retrieval metrics by this nearest-training Tanimoto
(within the protein-de-leaked sets) isolates de novo chemical generalisation.

Training ligands = CCD codes appearing in the DrugCLIP training subset
(train_ligand_freq.tsv); SMILES from the CCD components manifest (same library
that is screened). Reuses pooling/level logic from ligand_frequency_analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

ROOT = Path(__file__).resolve().parents[2] / 'DATA' / 'natural_ligands'
sys.path.insert(0, str(ROOT / "scripts"))
import screening as sc  # noqa: E402
from ligand_frequency_analysis import (  # noqa: E402
    load_pockets, level_mask, lib_size, train_ligand_freq,
)

RDLogger.DisableLog("rdApp.*")
CCD_MANIFEST = ROOT / "RESULTS" / "components_pub_unimol_manifest.tsv"
OUT_DIR = ROOT / "RESULTS" / "deleakage"
FP_RADIUS, FP_BITS = 2, 2048           # ECFP4
LEVELS = ["L1", "L2<0.3"]
# nearest-training-ligand Tanimoto bins, novel -> identical
TS_EDGES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
TS_LABELS = ["<0.3 (novel)", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-<1.0",
             "=1.0 (identical)"]


def code_to_fp() -> dict[str, object]:
    man = pd.read_csv(CCD_MANIFEST, sep="\t", dtype=str)
    out = {}
    for code, smi in zip(man["identifier"], man["smiles"]):
        if not isinstance(smi, str) or not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            out[str(code).upper()] = AllChem.GetMorganFingerprintAsBitVect(
                m, FP_RADIUS, FP_BITS)
    return out


def nearest_training_ts(fp_map: dict) -> dict[str, float]:
    """eval CCD code -> max ECFP4 Tanimoto to any training ligand."""
    train_codes = [c.upper() for c in train_ligand_freq()["ligand"]]
    train_fps = [fp_map[c] for c in set(train_codes) if c in fp_map]
    print(f"training ligands with a fingerprint: {len(train_fps)} "
          f"(of {len(set(train_codes))} unique training codes)")
    ts = {}
    for code, fp in fp_map.items():
        sims = BulkTanimotoSimilarity(fp, train_fps)
        ts[code] = float(max(sims)) if sims else 0.0
    return ts


def ts_bin(x: float) -> str:
    if np.isnan(x):
        return "no_fp"
    for lab, lo, hi in zip(TS_LABELS, TS_EDGES, TS_EDGES[1:] + [1.0001]):
        if lab == "=1.0 (identical)":
            if x >= 1.0:
                return lab
        elif lo <= x < hi:
            return lab
    return "=1.0 (identical)" if x >= 1.0 else TS_LABELS[0]


def annotate_nn_ts(pp: pd.DataFrame, fp_map: dict | None = None) -> pd.DataFrame:
    """Add `lig`, `nn_ts`, `ts_bin` columns to a pooled per-pocket frame."""
    if fp_map is None:
        fp_map = code_to_fp()
    nn = nearest_training_ts(fp_map)
    out = pp.copy()
    out["lig"] = out["lig_resname"].str.upper()
    out["nn_ts"] = out["lig"].map(nn)
    out["ts_bin"] = out["nn_ts"].map(ts_bin)
    return out


def stratify_ts(ev: pd.DataFrame, level: str, n_mols: int) -> pd.DataFrame:
    """Per-TS-bin retrieval metrics on `ev` (must already have `ts_bin` and
    `lig_freq` columns), within the given de-leakage level."""
    sub = ev[level_mask(ev, level)]
    rows = []
    for lab in TS_LABELS:
        sel = sub[sub["ts_bin"] == lab]
        m = sc.subset_metrics(sel, n_mols)
        if m.get("n_in_library", 0) == 0:
            rows.append({"level": level, "ts_bin": lab, "n": 0})
            continue
        rows.append({
            "level": level, "ts_bin": lab, "n": m["n_in_library"],
            "top1_%": round(m["top1_acc"] * 100, 1),
            "recall@10_%": round(m["recall@10"] * 100, 1),
            "ef@1%": round(m["ef@1%"], 1),
            "median_rank": round(m["median_rank"], 0),
            "med_ligfreq": int(sel["lig_freq"].median()),
        })
    return pd.DataFrame(rows)


# Disjoint TS bins for the manuscript grid (row 3), reusing ts_bin/TS_LABELS
# above but reordered lenient -> strict (identical -> novel) to match rows
# 1-2's left-to-right direction. Disjoint, not cumulative, for the same
# dilution reason as SI_BIN_ORDER / FREQ_BIN_ORDER: a real effect confined to
# one TS stratum should show up directly, not averaged into a "TS<t" pool.
TS_BIN_ORDER = ["all", *reversed(TS_LABELS)]
TS_BIN_LABELS = ["freq==0\n(all TS)", *reversed(TS_LABELS)]


def build_ts_bins_by_dataset(ev: pd.DataFrame, n_mols: int) -> pd.DataFrame:
    """Per-dataset disjoint novelty bins for the manuscript grid (row 3),
    within L2<0.3 pockets whose bound ligand is entirely training-unseen
    (lig_freq==0): 'all' as a reference point, then that population
    partitioned into non-overlapping nearest-training-ligand Tanimoto bins.
    Needs `nn_ts`/`ts_bin` columns (see `annotate_nn_ts`).
    """
    base = ev[level_mask(ev, "L2<0.3") & (ev["lig_freq"] == 0)]
    rows = []
    for ds, sub in base.groupby("dataset"):
        rows.append({"dataset": ds, "bin": "all", **sc.subset_metrics(sub, n_mols)})
        for lab in reversed(TS_LABELS):
            m = sc.subset_metrics(sub[sub["ts_bin"] == lab], n_mols)
            rows.append({"dataset": ds, "bin": lab, **m})
    return pd.DataFrame(rows)


def main() -> None:
    pp = load_pockets()
    n_mols = lib_size()
    fp_map = code_to_fp()
    pp = annotate_nn_ts(pp, fp_map=fp_map)
    ev = pp[pp["in_library"]]
    miss = ev["nn_ts"].isna().mean()
    print(f"eval in-library pockets without a ligand fingerprint: {miss:.1%}\n")

    for level in LEVELS:
        tab = stratify_ts(ev, level, n_mols)
        print(f"=== {level}: retrieval by nearest-training-ligand Tanimoto ===")
        print(tab.drop(columns=["level"]).to_string(index=False))
        s = ev[level_mask(ev, level)]
        s = s[s["nn_ts"].notna() & s["gt_rank"].notna()]
        rho = (pd.Series(s["nn_ts"].to_numpy())
               .corr(pd.Series(s["gt_rank"].astype(float).to_numpy()),
                     method="spearman"))
        print(f"Spearman(nn_ts, gt_rank) = {rho:+.3f}  "
              f"(negative => chemically-seen ligand ranks better)\n")


if __name__ == "__main__":
    main()
