"""Pool all four screening datasets (coach420, scpdb, pdbbind2020, holo4k)
into one combined population per bin, instead of one line per dataset — the
per-dataset disjoint-bin grid (manuscript_figures.plot_grid) leaves several
bins with single-digit n, especially for coach420. Pooling multiplies the
per-bin sample size, at the cost of one real complication: the same PDB
complex (same pdb_id + bound-ligand code) is sometimes curated into more
than one of these datasets, so naively concatenating them double- or
triple-counts it and skews the pooled metric. `dedup_cross_dataset` fixes
that: for a (pdb_id, ligand) pair spanning more than one dataset, keep only
the copy from the first dataset in DATASET_PRIORITY, drop the rest. Repeats
of the same pair WITHIN one dataset are left alone (that's a modelling
choice for that dataset, not a pooling artifact). All bin-building here
operates on the deduped pool only — a diagnostic elsewhere (the notebook's
"how much does duplication matter" cell) showed the raw-vs-deduped gap is
small, so a redundant second line isn't worth the clutter.

Uses the same disjoint-bin philosophy as manuscript_figures (a bin shows
that stratum's own metric, unmixed with everything cumulatively pooled
around it — see manuscript_figures module docstring), with coarser bins per
the reduced-bin request, each row going lenient -> strict left to right:

  row 1  within L1 (same-PDB removed), 4 SI%-to-training bins:
         70-100% -> 50-70% -> 30-50% -> <=30% (novel)
  row 2  within row 1's <=30% (novel) bin, ligand_frequency_analysis'
         existing 5 bins reversed: 100+ -> 25-99 -> 5-24 -> 1-4 ->
         0 (absent)
  row 3  within row 2's 0 (absent) bin, ligand_similarity_analysis' 6 bins
         reversed and the top two merged (0.9-<1.0 and =1.0 (identical) ->
         one 0.9-1.0 bin): 0.9-1.0 -> 0.7-0.9 -> 0.5-0.7 -> 0.3-0.5 ->
         <0.3 (novel)
"""
from __future__ import annotations

import pandas as pd

import ligand_frequency_analysis as lf
import ligand_similarity_analysis as ls
import screening as sc

DATASET_PRIORITY = ["coach420", "scpdb", "pdbbind2020", "holo4k"]

SI_BIN_ORDER = ["same PDB", "70-100%", "50-70%", "30-50%", "<=30% (novel)"]

# row 2: ligand-frequency bins, lenient (common) -> strict (absent)
FREQ_BIN_ORDER = ["100+", "25-99", "5-24", "1-4", "0 (absent)"]

# row 3: nearest-training-ligand Tanimoto bins, lenient (identical) ->
# strict (novel); 0.9-<1.0 and =1.0 (identical) merged into one 0.9-1.0 bin
TS_BIN_ORDER = ["0.9-1.0", "0.7-0.9", "0.5-0.7", "0.3-0.5", "<0.3 (novel)"]
_TS_MERGE = {"0.9-1.0": {"0.9-<1.0", "=1.0 (identical)"}}


def si_bin_coarse(best_pident: float) -> str:
    if best_pident > 0.70:
        return "70-100%"
    if best_pident > 0.50:
        return "50-70%"
    if best_pident > 0.30:
        return "30-50%"
    return "<=30% (novel)"


def load_pooled() -> pd.DataFrame:
    """All four datasets concatenated (ligand_frequency_analysis.load_pockets
    already does this), with in_train/best_pident/lig_freq annotated."""
    return lf.load_pockets()


def dedup_cross_dataset(pp: pd.DataFrame,
                         priority: list[str] = DATASET_PRIORITY) -> pd.DataFrame:
    """For each (pdb_id, ligand) pair spanning more than one dataset, keep
    only the rows from the first dataset in `priority` that has it; rows
    from every other dataset for that pair are dropped. A pair confined to
    a single dataset is untouched, including any within-dataset repeats.
    """
    key = pd.Series(list(zip(pp["pdb_id"].str.lower().str.strip(),
                              pp["lig_resname"].str.upper())), index=pp.index)
    winner = pp.groupby(key)["dataset"].transform(
        lambda s: next(d for d in priority if d in set(s)))
    return pp[pp["dataset"] == winner]


def l0_metrics(n_mols: int, pp: pd.DataFrame | None = None) -> dict:
    """Full deduped pool, no filtering at all — the "whole dataset" baseline
    for row 1's dashed reference line (rows 2/3 use their own first ("all")
    bin as their reference instead, since that IS the previous row's
    strictest bin)."""
    if pp is None:
        pp = load_pooled()
    return sc.subset_metrics(dedup_cross_dataset(pp), n_mols)


def build_si_bins_combined(n_mols: int, pp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row 1: combined (pooled, deduped) disjoint SI% bins, one row per bin.
    Together these 5 bins exactly partition the whole deduped dataset
    (l0_metrics): "same PDB" is the pdb_id-in-training (memorized) subset
    itself — not "training duplicates removed" — and the remaining 4 bins
    split everything else (pdb_id NOT in training) by receptor SI% to the
    nearest training receptor.
    """
    if pp is None:
        pp = load_pooled()
    deduped = dedup_cross_dataset(pp)
    same_pdb = deduped[deduped["in_train"]]
    rows = [{"bin": "same PDB", **sc.subset_metrics(same_pdb, n_mols)}]
    l1 = deduped[~deduped["in_train"]].copy()
    l1["si_bin"] = l1["best_pident"].map(si_bin_coarse)
    for b in SI_BIN_ORDER[1:]:
        m = sc.subset_metrics(l1[l1["si_bin"] == b], n_mols)
        rows.append({"bin": b, **m})
    return pd.DataFrame(rows)


def build_freq_bins_combined(n_mols: int, pp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row 2: combined (pooled, deduped) disjoint ligand-frequency bins,
    within row 1's <=30% (novel) bin."""
    if pp is None:
        pp = load_pooled()
    deduped = dedup_cross_dataset(pp)
    l1 = deduped[~deduped["in_train"]].copy()
    novel = l1[l1["best_pident"].map(si_bin_coarse) == "<=30% (novel)"].copy()
    rows = [{"bin": "all", **sc.subset_metrics(novel, n_mols)}]
    novel["freq_bin"] = novel["lig_freq"].map(lf.freq_bin)
    for b in FREQ_BIN_ORDER:
        m = sc.subset_metrics(novel[novel["freq_bin"] == b], n_mols)
        rows.append({"bin": b, **m})
    return pd.DataFrame(rows)


def build_ts_bins_combined(n_mols: int, pp_ts: pd.DataFrame) -> pd.DataFrame:
    """Row 3: combined (pooled, deduped) disjoint nearest-training-ligand
    Tanimoto bins, within row 2's 0 (absent) bin. `pp_ts` must already have
    `nn_ts`/`ts_bin` columns (see ligand_similarity_analysis.annotate_nn_ts) —
    computed once on the raw pool before dedup, since nn_ts depends only on
    the ligand code, not on which row survives dedup.
    """
    deduped = dedup_cross_dataset(pp_ts)
    l1 = deduped[~deduped["in_train"]].copy()
    l1["si_bin"] = l1["best_pident"].map(si_bin_coarse)
    novel = l1[l1["si_bin"] == "<=30% (novel)"]
    unseen = novel[novel["lig_freq"] == 0]
    rows = [{"bin": "all", **sc.subset_metrics(unseen, n_mols)}]
    for b in TS_BIN_ORDER:
        labels = _TS_MERGE.get(b, {b})
        m = sc.subset_metrics(unseen[unseen["ts_bin"].isin(labels)], n_mols)
        rows.append({"bin": b, **m})
    return pd.DataFrame(rows)
