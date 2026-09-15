"""DrugCLIP-style virtual screening: pocket embeddings vs compound libraries.

Shared logic for the per-dataset screening notebooks (`NOTEBOOKS/15_screen_*`).
Kept in a module (not duplicated across notebooks) so the scoring/metric code
has a single source of truth and stays diff-reviewable.

Data model
----------
Compound libraries — `RESULTS/mols/<LIB>.h5`:
  * ``mol_reps``  (N, 768) float32 = 6 fold blocks of 128, each block already
    L2-normalised by the DTWG encoder.
  * ``fold0..fold5`` (N,) bool — per-fold validity masks.
  * h5 row order is the **lexicographic** order of the lmdb integer-string
    keys (``'0','1','10','100',...``), NOT numeric manifest order. The
    manifest must be sorted by ``str(lmdb_key)`` before pairing it with the
    h5 rows (see ``load_mol_library``). The manifest ``identifier`` column
    is, for CCD, the PDB chemical-component code — the same vocabulary as a
    pocket's bound-ligand ``lig_resname``.

Pocket sets — `RESULTS/pockets/<DS>_pocket_reps.pkl`:
  * pickled ``(names, arr)``; ``arr`` (6, P, 128) float32, each (fold, pocket)
    vector already L2-normalised.
  * ``names`` is a permutation of the pocket-manifest ``pocket`` column, so
    join on the string, never on row order.

Because every vector is unit-norm, cosine similarity == dot product.

Ground truth only exists for the **CCD** library: a pocket's ``lig_resname``
is looked up among the CCD identifiers. Peptide / multi-residue ligands
(hyphenated resnames) have no single-component row and are reported as
``in_library == False`` and excluded from accuracy denominators. For COCONUT /
LOTUS there is no PDB-code identifier, so screening still runs but the
rank-of-bound-ligand metrics are not defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / 'DATA' / 'natural_ligands'
MOLS_DIR = ROOT / "RESULTS" / "mols"
POCKETS_DIR = ROOT / "RESULTS" / "pockets"

N_FOLDS = 6
EMB_DIM = 128

# library name -> (h5 file, unimol manifest, identifier column)
MOL_LIBRARIES = {
    "CCD": (
        MOLS_DIR / "CCD.h5",
        ROOT / "RESULTS" / "components_pub_unimol_manifest.tsv",
        "identifier",
    ),
    "COCONUT_human": (
        MOLS_DIR / "COCONUT_human.h5",
        ROOT / "RESULTS" / "coconut_homo_sapiens_unimol_manifest.tsv",
        "identifier",
    ),
    "LOTUS": (
        MOLS_DIR / "LOTUS.h5",
        ROOT / "RESULTS" / "lotus_2021_unimol_manifest.tsv",
        "lotus_id",
    ),
    # Bound-ligand control libraries (built by NOTEBOOKS/16, encoded on the
    # cluster). One entry per pocket, in the *exact crystallographic pose*;
    # the identifier is the pocket key, so screen(..., gt="pocket") asks
    # "does a pocket rank its own true-pose ligand #1?".
    **{
        f"bound_{ds}": (
            MOLS_DIR / f"bound_{ds}.h5",
            MOLS_DIR / f"bound_{ds}_manifest.tsv",
            "name",
        )
        for ds in ("coach420", "scpdb", "pdbbind2020", "holo4k")
    },
}

# pocket-set name -> (reps pickle, pocket manifest). Note the scpdb/sc-pdb
# spelling split between the reps file and the manifest.
POCKET_DATASETS = {
    "coach420": (
        POCKETS_DIR / "coach420_pocket_reps.pkl",
        POCKETS_DIR / "coach420_pocket_manifest.tsv",
    ),
    "scpdb": (
        POCKETS_DIR / "scpdb_pocket_reps.pkl",
        POCKETS_DIR / "sc-pdb_pocket_manifest.tsv",
    ),
    "pdbbind2020": (
        POCKETS_DIR / "pdbbind2020_pocket_reps.pkl",
        POCKETS_DIR / "pdbbind2020_pocket_manifest.tsv",
    ),
    "holo4k": (
        POCKETS_DIR / "holo4k_pocket_reps.pkl",
        POCKETS_DIR / "holo4k_pocket_manifest.tsv",
    ),
}

# Scoring modes. Faithful to Jia et al., Science 2026 (eads9530), "Ensemble
# of multiple models and pockets during screening", and the repo's
# retrieval_multi_folds / screening_utils.py.
#
#   official : six-fold cross-validation ensemble. Each fold k is one full
#              model that encodes BOTH pocket and molecule, giving one
#              prediction cos(pocket_k, mol_k). The six predictions are
#              combined by MEAN pooling (no cross-fold pairing — fold i
#              pocket is never matched with fold j mol). This is the
#              canonical per-pocket screening score.
#   fold0..fold5 : single-fold diagnostic — that one model's
#              cos(pocket_k, mol_k). Use to inspect per-model behaviour.
#
# The adjusted robust z-score (eq. 4: (x-median)/(1.48*MAD)) followed by
# max pooling is the paper's recipe for combining MULTIPLE POCKET
# CONFORMATIONS of one target, NOT for folds. It is rank-preserving within
# a single pocket, so it does not affect per-pocket bound-ligand ranking;
# `screen(..., group_by_pdb=True)` applies it when a target has several
# pocket rows.
SCORING_MODES = ["official"] + [f"fold{k}" for k in range(N_FOLDS)]
DEFAULT_MODE = "official"


@dataclass
class MolLibrary:
    name: str
    emb: np.ndarray          # (N, 6, 128) float32, per-fold unit norm
    fold_valid: np.ndarray   # (N, 6) bool
    ids: np.ndarray          # (N,) object — identifier per row
    manifest: pd.DataFrame   # full manifest, row-aligned to emb


@dataclass
class PocketSet:
    name: str
    names: list[str]         # (P,) pocket ids
    emb: np.ndarray          # (6, P, 128) float32, per-fold unit norm
    manifest: pd.DataFrame   # manifest reindexed to `names` order
    lig_resname: np.ndarray  # (P,) object — bound-ligand code (ground truth)


def load_mol_library(name: str) -> MolLibrary:
    h5_path, manifest_path, id_col = MOL_LIBRARIES[name]
    with h5py.File(h5_path, "r") as f:
        reps = f["mol_reps"][:]                       # (N, 768)
        valid = np.stack([f[f"fold{k}"][:] for k in range(N_FOLDS)], axis=1)
    n = reps.shape[0]
    emb = reps.reshape(n, N_FOLDS, EMB_DIM).astype(np.float32, copy=False)

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    if len(manifest) != n:
        raise ValueError(
            f"{name}: manifest rows ({len(manifest)}) != h5 rows ({n}); "
            "row alignment assumption broken."
        )
    # The cluster encoder wrote mol_reps (and the fold masks) in LEXICOGRAPHIC
    # order of the lmdb integer-string keys ('0','1','10','100',...), not
    # numeric manifest order. Re-sort the manifest the same way so manifest
    # row i <-> emb[i]. The fold masks come straight from the h5 and are
    # already in this order.
    manifest = manifest.sort_values(
        "lmdb_key", key=lambda s: s.astype(str)).reset_index(drop=True)
    ids = manifest[id_col].to_numpy(dtype=object)
    return MolLibrary(name=name, emb=emb, fold_valid=valid, ids=ids,
                      manifest=manifest)


def concat_libraries(*libs: MolLibrary, name: str | None = None
                      ) -> MolLibrary:
    """Stack several libraries into one screening library.

    Used to append a bound-ligand control set to a base library (e.g. CCD)
    so a pocket is ranked against the full background *plus* its own
    true-pose ligand. Manifests are outer-joined on column union.
    """
    emb = np.concatenate([l.emb for l in libs], axis=0)
    fold_valid = np.concatenate([l.fold_valid for l in libs], axis=0)
    ids = np.concatenate([l.ids for l in libs])
    manifest = pd.concat([l.manifest for l in libs], ignore_index=True)
    return MolLibrary(
        name=name or "+".join(l.name for l in libs),
        emb=emb, fold_valid=fold_valid, ids=ids, manifest=manifest,
    )


def load_pocket_set(name: str) -> PocketSet:
    reps_path, manifest_path = POCKET_DATASETS[name]
    with open(reps_path, "rb") as fh:
        names, arr = pd.read_pickle(fh)
    names = [str(x) for x in names]
    arr = np.asarray(arr, dtype=np.float32)           # (6, P, 128)

    man = pd.read_csv(manifest_path, sep="\t", dtype=str)
    man = man.set_index("pocket")
    missing = [p for p in names if p not in man.index]
    if missing:
        raise ValueError(
            f"{name}: {len(missing)} pocket(s) in reps absent from manifest, "
            f"e.g. {missing[:3]}"
        )
    man = man.loc[names].reset_index()
    return PocketSet(
        name=name,
        names=names,
        emb=arr,
        manifest=man,
        lig_resname=man["lig_resname"].to_numpy(dtype=object),
    )


def fold_scores(ps: PocketSet, lib: MolLibrary, idx: np.ndarray,
                fold: int) -> np.ndarray:
    """(len(idx), N) single-fold prediction: cos(pocket_k, mol_k) — fold k's
    model encodes both sides, so this is one self-contained model's score."""
    return ps.emb[fold, idx, :] @ lib.emb[:, fold, :].T


def ensemble_scores(ps: PocketSet, lib: MolLibrary, idx: np.ndarray
                     ) -> np.ndarray:
    """(len(idx), N) canonical score: MEAN pooling of the six same-model
    predictions (paper "Ensemble of multiple models ... combined using mean
    pooling"). No cross-fold pairing."""
    acc = None
    for k in range(N_FOLDS):
        s = ps.emb[k, idx, :] @ lib.emb[:, k, :].T
        acc = s if acc is None else acc + s
    return acc / N_FOLDS


def subset_metrics(per_pocket: pd.DataFrame, n_mols: int,
                   topk=(1, 5, 10), top_pct=(0.01, 0.05)) -> dict:
    """Aggregate retrieval metrics over a ``per_pocket`` frame or any subset
    of one. Reuses the per-pocket ``gt_rank`` / ``hit@`` columns that
    ``screen`` already wrote, so an arbitrary pocket subset (e.g. a de-leaked
    hold-out) can be re-scored from a saved per_pocket parquet without
    re-running the screen. ``n_mols`` is the library size the screen was run
    against (needed for the EF normalisation / top-pct cut).
    """
    ev = per_pocket[per_pocket["in_library"]]
    out = {"n_pockets": len(per_pocket), "n_in_library": int(len(ev))}
    if not len(ev):
        return out
    ranks = ev["gt_rank"].astype(int).to_numpy()
    pct_k = {p: max(1, int(round(p * n_mols))) for p in top_pct}
    out["top1_acc"] = float((ranks == 1).mean())
    for k in topk:
        out[f"recall@{k}"] = float(ev[f"hit@{k}"].mean())
    for p in top_pct:
        recall_p = float(ev[f"hit@{p:.0%}"].mean())
        out[f"recall@{p:.0%}"] = recall_p
        # Enrichment factor: with one ground-truth active per pocket,
        # EF_x% = recall@x% / (screened fraction). The screened fraction is
        # pct_k/n_mols (the actual top-k cut, rounding-aware), so EF is capped
        # at n_mols/pct_k ≈ 1/x%.
        out[f"ef@{p:.0%}"] = recall_p * n_mols / pct_k[p]
    out["median_rank"] = float(np.median(ranks))
    out["mean_rank"] = float(ranks.mean())
    return out


def adjusted_robust_z(scores: np.ndarray) -> np.ndarray:
    """Eq. 4: (x - median) / (1.48 * MAD), per row (per pocket). Used before
    max-pooling across multiple pocket conformations of one target."""
    med = np.median(scores, axis=1, keepdims=True)
    mad = np.median(np.abs(scores - med), axis=1, keepdims=True)
    return (scores - med) / (1.48 * mad + 1e-6)


def screen(ps: PocketSet, lib: MolLibrary, mode: str = DEFAULT_MODE,
           topk=(1, 5, 10), top_pct=(0.01, 0.05),
           keep_top=10, chunk=256, gt: str = "lig_resname"
           ) -> tuple[pd.DataFrame, dict]:
    """Run the screen and score retrieval of each pocket's bound ligand.

    gt : how a pocket's ground-truth library row is identified.
        "lig_resname" — match the pocket's bound-ligand 3-letter code to a
        library identifier (CCD-style; the cognate may be any instance).
        "pocket"      — match the pocket's own key to a library identifier.
        Use with a bound-ligand control library (NOTEBOOK 16) so the test
        is the *exact* crystallographic pose of *this* complex's ligand.

    Returns
    -------
    per_pocket : DataFrame, one row per pocket
        pocket, pdb_id, lig_resname, in_library, gt_rank (1-based, NA if the
        ligand is not a library row), gt_score, hit@k / hit@pct booleans,
        best_hit_id, best_hit_score, top_hits (list of (id, score)).
    summary : dict
        n_pockets, n_in_library, mode, library, and over the in-library
        subset: top1_acc, recall@k, recall@pct, ef@pct (enrichment factor,
        = recall@pct / screened-fraction), median_rank, mean_rank.
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"mode {mode!r} not in {SCORING_MODES}")
    if gt not in ("lig_resname", "pocket"):
        raise ValueError(f"gt must be 'lig_resname' or 'pocket', got {gt!r}")

    n_mols = lib.emb.shape[0]
    id_to_row: dict[str, int] = {}
    for i, mid in enumerate(lib.ids):
        id_to_row.setdefault(mid, i)  # first occurrence wins

    pct_k = {p: max(1, int(round(p * n_mols))) for p in top_pct}
    pdb_id = ps.manifest["pdb_id"].to_numpy(dtype=object)

    rows = []
    for start in range(0, len(ps.names), chunk):
        sel = np.arange(start, min(start + chunk, len(ps.names)))
        if mode == "official":
            sims = ensemble_scores(ps, lib, sel)             # (c, N)
        else:
            sims = fold_scores(ps, lib, sel, int(mode[4:]))  # (c, N)

        kk = min(keep_top, n_mols)
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r, prow in enumerate(sel):
            srow = sims[r]
            order = part[r][np.argsort(-srow[part[r]])]
            top_hits = [(str(lib.ids[j]), float(srow[j])) for j in order]

            lig = ps.lig_resname[prow]
            key = ps.names[prow] if gt == "pocket" else lig
            grow = id_to_row.get(key)
            rec = {
                "pocket": ps.names[prow],
                "pdb_id": pdb_id[prow],
                "lig_resname": lig,
                "in_library": grow is not None,
                "best_hit_id": top_hits[0][0],
                "best_hit_score": top_hits[0][1],
                "top_hits": top_hits,
            }
            if grow is None:
                rec["gt_rank"] = pd.NA
                rec["gt_score"] = pd.NA
            else:
                gs = float(srow[grow])
                # 1-based rank, ties resolved best-case (strictly-greater count)
                rank = int((srow > gs).sum()) + 1
                rec["gt_score"] = gs
                rec["gt_rank"] = rank
                for k in topk:
                    rec[f"hit@{k}"] = rank <= k
                for p in top_pct:
                    rec[f"hit@{p:.0%}"] = rank <= pct_k[p]
            rows.append(rec)

    per_pocket = pd.DataFrame(rows)
    summary = {
        "library": lib.name,
        "pocket_set": ps.name,
        "mode": mode,
        "gt": gt,
        "n_missing_ligand": int((~per_pocket["in_library"]).sum()),
        "n_library_mols": n_mols,
    }
    summary.update(subset_metrics(per_pocket, n_mols, topk=topk,
                                  top_pct=top_pct))
    return per_pocket, summary


def compare_modes(ps: PocketSet, lib: MolLibrary, modes=None,
                  **kw) -> pd.DataFrame:
    """Summary-row-per-mode table: the canonical ``official`` method vs each
    single-checkpoint diagnostic."""
    modes = modes or SCORING_MODES
    return pd.DataFrame([screen(ps, lib, mode=m, **kw)[1] for m in modes])
