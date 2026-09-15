"""Plot + table helpers for the combined (all-datasets-pooled) de-leakage
grid built by combined_deleakage_analysis. Same 3x3 layout as
manuscript_figures.plot_grid (rows = SI% / ligand-freq / novelty bins,
columns = EF1% / top-1% / median rank), one deduped-pooled line per panel,
plus a dashed reference line per row: row 1's is the whole (unfiltered)
dataset (`combined_deleakage_analysis.l0_metrics`); rows 2/3's is that row's
own "all" bin, which is exactly the previous row's strictest bin — so the
dashed line always answers "how does this bin compare to the population it
was carved out of". That "all" bin is NOT plotted as a data point on rows
2/3 (the dashed line already shows its value — a second, redundant marker
would just repeat it). Axes are shared per column (`sharey`) so all three
rows read off the same scale.

Every column also gets a dotted random-chance reference line, closed-form
given the library size (n_mols): EF1%=1 by definition; top-1% = 1/n_mols
(probability the one true active lands exactly at rank 1 under a uniform
random ranking); median rank = (n_mols+1)/2 (median of a discrete uniform
rank over 1..n_mols).

Tick labels spell out the exact interval each bin covers (matching the
manuscript's own bracket convention: `>70%`, `(50,70]%`, ... `[0.9,1.0]`,
`[0.7,0.9)`, ...) rather than a loose "70-100%"-style range.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_deleakage_analysis import FREQ_BIN_ORDER, SI_BIN_ORDER, TS_BIN_ORDER

LINE_COLOR = "#1b6ea8"

GRID_METRICS = [("ef@1%", r"EF$_{1\%}$", False),
                ("top1_acc", "Top-1 recall (%)", False),
                ("median_rank", "Bound ligand median rank", True)]

SI_BIN_LABELS = ["Same\nPDB", ">70%", "(50,70] %", "(30,50] %", "≤30%\n(novel)"]
FREQ_BIN_LABELS = ["≥100", "[25,99]", "[5,24]", "[1,4]", "0\n(absent)"]
TS_BIN_LABELS = ["[0.9,1.0]", "[0.7,0.9)", "[0.5,0.7)", "[0.3,0.5)",
                 "<0.3\n(novel)"]

# table-only variants that also show the "all" reference bin (dropped from
# the plotted line on rows 2/3, but still useful as a denominator in the
# counts/performance tables)
FREQ_BIN_LABELS_FULL = ["SI≤30%\n(all freq)", *FREQ_BIN_LABELS]
FREQ_BIN_ORDER_FULL = ["all", *FREQ_BIN_ORDER]
TS_BIN_LABELS_FULL = ["freq==0\n(all TS)", *TS_BIN_LABELS]
TS_BIN_ORDER_FULL = ["all", *TS_BIN_ORDER]


def _random_baseline(col: str, n_mols: int) -> float:
    """Chance-level value for `col`, given a library of n_mols candidates
    and exactly one true active per pocket."""
    if col == "ef@1%":
        return 1.0
    if col == "top1_acc":
        return 100.0 / n_mols  # already in percent, matches the *100 below
    if col == "median_rank":
        return (n_mols + 1) / 2.0
    raise ValueError(col)


def _draw_panel(ax, d: pd.DataFrame, x_order: list[str], x_labels: list[str],
                 n_mols: int, col: str, label: str, logy: bool, baseline,
                 xtick_rotation: float = 0, xtick_ha: str = "center"):
    """Draw one metric panel (one column of the grid, or one standalone
    figure) onto `ax`. `d` is the bin dataframe indexed by `bin`, already
    reindexed to `x_order` is NOT assumed — reindexing happens here so this
    can be reused standalone. `xtick_rotation`/`xtick_ha` let standalone
    panels rotate labels (to fit the same font size as the y-axis in a
    narrower panel) without affecting the combined grid's unrotated ticks."""
    x_pos = np.arange(len(x_order))
    dd = d.reindex(x_order)
    y = dd[col].to_numpy(dtype=float)
    n = dd["n_in_library"].to_numpy(dtype=float)
    ok = np.isfinite(y) & (n > 0)
    yy = y * 100 if col == "top1_acc" else y
    base_y = baseline[col] * 100 if col == "top1_acc" else baseline[col]
    ax.axhline(base_y, color=LINE_COLOR, ls="--", lw=1.2, alpha=0.6)
    ax.plot(x_pos[ok], yy[ok], marker="o", ms=4, lw=1.5, color=LINE_COLOR)
    ax.set_ylabel(label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha=xtick_ha,
                       fontsize=plt.rcParams["ytick.labelsize"])
    ax.set_box_aspect(1)
    ax.axhline(_random_baseline(col, n_mols), color="k", ls=":", lw=0.7)
    if logy:
        ax.set_yscale("log")


def _draw_row(axes_row, df: pd.DataFrame, x_order: list[str],
              x_labels: list[str], n_mols: int, baseline=None,
              baseline_bin: str | None = None):
    """baseline: dict/Series with ef@1%/top1_acc/median_rank for the dashed
    reference line. If None, looked up from `df` (NOT restricted to
    `x_order`) at `baseline_bin` — lets the baseline be a bin that isn't
    itself plotted as a data point.
    """
    full = df.set_index("bin")
    if baseline is None:
        baseline = full.loc[baseline_bin]
    for ax, (col, label, logy) in zip(axes_row, GRID_METRICS):
        _draw_panel(ax, full, x_order, x_labels, n_mols, col, label, logy,
                    baseline, xtick_rotation=30, xtick_ha="center")


def plot_grid_combined(si_bins: pd.DataFrame, freq_bins: pd.DataFrame,
                        ts_bins: pd.DataFrame, l0: dict, n_mols: int,
                        figsize=(11, 11)):
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharey="col")

    _draw_row(axes[0], si_bins, SI_BIN_ORDER, SI_BIN_LABELS, n_mols, baseline=l0)
    _draw_row(axes[1], freq_bins, FREQ_BIN_ORDER, FREQ_BIN_LABELS, n_mols,
              baseline_bin="all")
    _draw_row(axes[2], ts_bins, TS_BIN_ORDER, TS_BIN_LABELS, n_mols,
              baseline_bin="all")

    for ax, (_, label, _logy) in zip(axes[0], GRID_METRICS):
        ax.set_title(label)

    row_titles = ["SI% bins\n(of whole dataset, pooled)",
                  "ligand-freq bins\n(within SI≤30%, pooled)",
                  "novelty bins\n(within freq==0, pooled)"]
    for ax, title in zip(axes[:, 0], row_titles):
        ax.annotate(title, xy=(-0.45, 0.5), xycoords="axes fraction",
                    rotation=90, va="center", ha="center",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def save_panels_combined(si_bins: pd.DataFrame, freq_bins: pd.DataFrame,
                          ts_bins: pd.DataFrame, l0: dict, n_mols: int,
                          out_dir, prefix: str = "26_panel", dpi: int = 600,
                          figsize=(3.6, 3.6)) -> list:
    """Save each of the 9 grid panels (3 rows x 3 metrics) as its own PNG at
    `dpi`, instead of one combined 3x3 figure — same data/styling as
    `plot_grid_combined`, just one panel per file. Y-limits are still forced
    to match across the 3 panels of a metric column (what `sharey="col"` gave
    us in the combined grid), by drawing a throwaway sharey grid first and
    copying its per-column limits onto the standalone panels."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_data = {
        "si": (si_bins.set_index("bin"), SI_BIN_ORDER, SI_BIN_LABELS, l0),
        "freq": (freq_bins.set_index("bin"), FREQ_BIN_ORDER, FREQ_BIN_LABELS,
                 None),
        "ts": (ts_bins.set_index("bin"), TS_BIN_ORDER, TS_BIN_LABELS, None),
    }

    ref_fig, ref_axes = plt.subplots(3, 3, sharey="col")
    for ax_row, (d, x_order, x_labels, baseline) in zip(ref_axes, row_data.values()):
        baseline_row = baseline if baseline is not None else d.loc["all"]
        for ax, (col, label, logy) in zip(ax_row, GRID_METRICS):
            _draw_panel(ax, d, x_order, x_labels, n_mols, col, label, logy,
                        baseline_row, xtick_rotation=30, xtick_ha="center")
    col_ylim = [ref_axes[0, c].get_ylim() for c in range(3)]
    plt.close(ref_fig)

    paths = []
    for row_name, (d, x_order, x_labels, baseline) in row_data.items():
        baseline_row = baseline if baseline is not None else d.loc["all"]
        for c, (col, label, logy) in enumerate(GRID_METRICS):
            fig, ax = plt.subplots(figsize=figsize)
            _draw_panel(ax, d, x_order, x_labels, n_mols, col, label, logy,
                        baseline_row, xtick_rotation=30, xtick_ha="center")
            ax.set_ylim(*col_ylim[c])
            fig.tight_layout()
            metric_tag = col.replace("@", "").replace("%", "pct").replace("_", "")
            out = out_dir / f"{prefix}_{row_name}_{metric_tag}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)
    return paths


def _clean(labels: list[str]) -> list[str]:
    return [lab.replace("\n", " ") for lab in labels]


def _bin_frame(df: pd.DataFrame, order: list[str], labels: list[str],
               l0: dict | None = None) -> pd.DataFrame:
    """`df` reindexed to `order`/`labels`, optionally with an `l0` (whole
    dataset) reference row prepended — used for both the counts and
    performance tables so the "full dataset" row always matches."""
    d = df.set_index("bin").reindex(order)
    lbls = list(labels)
    if l0 is not None:
        l0_row = pd.DataFrame([l0], index=["__l0__"])
        d = pd.concat([l0_row, d])
        lbls = ["whole dataset\n(L0)"] + lbls
    d.index = _clean(lbls)
    return d


def pocket_counts_combined(si_bins: pd.DataFrame, freq_bins: pd.DataFrame,
                            ts_bins: pd.DataFrame, l0: dict
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """bin -> surviving in-library pocket-ligand pairs (n_in_library), one
    table per row of `plot_grid_combined` (rows 2/3 also include the "all"
    reference bin; row 1 also gets the whole-dataset L0 reference — rows
    2/3 don't repeat it since they already have their own "all" reference,
    which IS the previous row's strictest bin, not the global L0)."""
    def as_row(df, order, labels, l0_arg=None):
        return _bin_frame(df, order, labels, l0=l0_arg)[["n_in_library"]].T

    return (as_row(si_bins, SI_BIN_ORDER, SI_BIN_LABELS, l0_arg=l0),
            as_row(freq_bins, FREQ_BIN_ORDER_FULL, FREQ_BIN_LABELS_FULL),
            as_row(ts_bins, TS_BIN_ORDER_FULL, TS_BIN_LABELS_FULL))


def performance_combined(si_bins: pd.DataFrame, freq_bins: pd.DataFrame,
                          ts_bins: pd.DataFrame, l0: dict
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same bins/columns as `pocket_counts_combined`, but rows are the
    actual retrieval metrics (EF1%, top-1%, median rank) instead of n."""
    def as_rows(df, order, labels, l0_arg=None):
        d = _bin_frame(df, order, labels, l0=l0_arg)
        out = pd.DataFrame({
            "EF1%": d["ef@1%"].round(1),
            "top-1 (%)": (d["top1_acc"] * 100).round(2),
            "median rank": d["median_rank"].round(1),
        }, index=d.index)
        return out.T

    return (as_rows(si_bins, SI_BIN_ORDER, SI_BIN_LABELS, l0_arg=l0),
            as_rows(freq_bins, FREQ_BIN_ORDER_FULL, FREQ_BIN_LABELS_FULL),
            as_rows(ts_bins, TS_BIN_ORDER_FULL, TS_BIN_LABELS_FULL))
