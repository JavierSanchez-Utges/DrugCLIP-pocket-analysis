# DrugCLIP-pocket-analysis

<!-- TODO: replace with the published paper link once available -->
This is the GitHub repository underlying the analysis for our paper "Interrogating contrastive learning embeddings for structure-based virtual screening: a case study on DrugCLIP" (Utgés _et al._, 2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22768064.svg)](https://doi.org/10.5281/zenodo.22768064)

## Overview

This work benchmarks DrugCLIP-style pocket embeddings under a series of pocket-source
perturbations — apo/holo/AlphaFold-predicted structures, P2Rank-detected pockets, and random
surface patches — against experimental, ligand-defined pockets, and asks how much retrieval
performance degrades as the pocket definition moves away from the ideal case. It draws on three
underlying projects:

- **DrugCLIP pocket-source robustness** (`CHEMBL_DRUGCLIP`) — apo/holo/AFDB structure and
  P2Rank/random-patch pocket-detection perturbations, scored against experimental pockets and
  pooled across the coach420 / holo4k / pdbbind2020 / scPDB datasets.
- **Pocket-matching benchmark** (`POCKET_MATCHING_BENCHMARK`) — the PROSPECCTS pocket-similarity
  benchmark, comparing DrugCLIP-derived scores against PocketVec, NRGRank and EPoCS, plus naive
  sequence/ligand-identity baselines.
- **Natural-ligand embedding analysis** (`NATURAL_LIGANDS`) — pocket/ligand embedding geometry,
  embedding-vs-structural-similarity relationships, and training-set de-leakage analysis for the
  screening benchmark.

## Repository structure

```
DrugCLIP-pocket-analysis/
├── README.md
├── LICENSE
├── environment.yml
├── ANALYSIS/                     # all notebooks, flat, plus two small helper-code folders
│   ├── 00_prospeccts_dataset_summary.ipynb
│   ├── 02_benchmark_methods.ipynb
│   ├── 04_naive_baselines.ipynb
│   ├── 20_visualise_pockets_by_ligand.ipynb
│   ├── 24_embedding_vs_tanimoto_unique_pairs.ipynb
│   ├── 25_pocket_embedding_vs_rmsd_visualisations.ipynb
│   ├── 26_combined_deleakage.ipynb
│   ├── 45_ahoj_apo_vs_experimental_scores_pooled.ipynb
│   ├── 46_ahoj_holo_vs_experimental_scores_pooled.ipynb
│   ├── 47_afdb_apo_holo_key_plots.ipynb
│   ├── 48_scatter_color_exploration.ipynb
│   ├── 49_p2rank_patches_pooled.ipynb
│   ├── 50_p2rank_patches_key_plots.ipynb
│   ├── 51_p2rank_scatter_threshold_exploration.ipynb
│   ├── 52_pooled_screening_key_plots.ipynb
│   ├── naive_baselines_lib.py         # helper module for notebook 04
│   └── natural_ligands_scripts/       # helper modules for notebooks 26 and 52
└── DATA/                          # curated inputs — the minimal set each notebook needs
    ├── drugclip/RESULTS/...            # from CHEMBL_DRUGCLIP
    ├── pocket_benchmark/DATA/...       # from POCKET_MATCHING_BENCHMARK
    └── natural_ligands/{DATA,RESULTS}/...  # from NATURAL_LIGANDS
```

Notebooks are numbered as in their original projects and kept flat in one folder; the number
prefix indicates origin (`00`/`02`/`04` = pocket-matching benchmark, `20`s = natural-ligand
analysis, `40`s = DrugCLIP pocket-source robustness). Each notebook resolves its own data paths
relative to the repo root (`Path.cwd().parent`), so they run unmodified straight after cloning —
no local path configuration needed.

## Data

This repository holds **code only** — `DATA/` is git-ignored and not pushed to GitHub. The data
itself (the **minimal reproducing input set** for these 15 notebooks, ~5.6 GB.

<!-- TODO: replace with the real Zenodo DOI once the deposit is published -->
**Zenodo DOI: TODO**

To reproduce the analysis, download the Zenodo archive and extract it as `DATA/` at the repo root
(i.e. so `DATA/drugclip/`, `DATA/pocket_benchmark/` and `DATA/natural_ligands/` sit directly under
this repo, alongside `ANALYSIS/`) — every notebook resolves its paths relative to the repo root, so
no further configuration is needed.

## Environment

`environment.yml` reflects the actual conda environment (`myNewEnv`) these analyses were run in —
exact package versions exported from that environment, not a guessed list.

```bash
conda env create -f environment.yml
conda activate drugclip_pocket_analysis_env
```

## External dependencies

These notebooks compare against, or build on data produced by, the following external methods,
tools and databases. They aren't Python packages (so aren't in `environment.yml`) — this is for
provenance/citation, not something you need to install to run the notebooks as shipped (their
outputs are already baked into `DATA/`), except where noted.

1. [P2Rank](https://github.com/rdk/p2rank) — pocket-detection method whose predictions are
   compared throughout (PDB and AlphaFold-model structures). Krivák, *et al.* "P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure." *J Cheminform.* 2018. [doi:10.1186/s13321-018-0285-8](https://link.springer.com/article/10.1186/s13321-018-0285-8)
2. [DrugCLIP](https://github.com/bowen-gao/DrugCLIP) — the contrastive pocket/ligand embedding
   model this whole analysis evaluates. Jia, *et al.* "Deep contrastive learning enables genome-wide virtual screening." *Science* 2026. [doi:10.1126/science.ads9530](https://www.science.org/doi/10.1126/science.ads9530)
3. [PocketVec](https://github.com/sbnb-irb/pocketvec) — pocket-comparison method compared in
   notebook 02. Comajuncosa-Creus *et al.* "Comprehensive detection and characterization of human
   druggable pockets through binding site descriptors." *Nat Commun.* 2024.
   [doi:10.1038/s41467-024-52146-3](https://www.nature.com/articles/s41467-024-52146-3).
4. [NRGRank](https://github.com/NRGlab/NRGRank) — pocket/ligand scoring method compared in notebook 02.
   [bioRxiv 2025.02.17.638675](https://www.biorxiv.org/content/10.1101/2025.02.17.638675v1.full).
5. [EPoCS](https://github.com/tugceoruc/epocs) — ESM-2-based pocket cross-similarity method
   compared in notebook 02. Oruç, *et al.* "Mapping the space of protein binding sites with
   sequence-based protein language models." *Bioinformatics.* 2025;41(6):btaf284.
   [doi:10.1093/bioinformatics/btaf284](https://academic.oup.com/bioinformatics/article/41/6/btaf284/8176567).
6. [US-align](https://github.com/pylelab/USalign) — structural alignment tool used upstream to
   compute the SC-RMSD values shipped in `DATA/` (not re-run by these notebooks). Zhang, *et al.*
   "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular
   complexes." *Nat Methods.* 2022.
   [doi:10.1038/s41592-022-01585-1](https://www.nature.com/articles/s41592-022-01585-1).
7. [MMseqs2](https://github.com/soedinglab/MMseqs2) — used upstream to compute pocket/receptor
   sequence identity (shipped in `DATA/`, not re-run by these notebooks). Hauser, _et al._ "MMseqs software suite for fast and deep clustering and searching of large protein sequence sets." *Bioinformatics* 2016. [doi:10.1093/bioinformatics/btw006](https://academic.oup.com/bioinformatics/article/32/9/1323/1744460)

## Citation

<!-- TODO: add citation once published -->
