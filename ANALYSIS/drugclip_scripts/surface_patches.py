"""Surface-patch segmentation for AFDB models + P2Rank pockets.

For a given AFDB mmCIF model and its P2Rank `<model>.cif_residues.csv`,
partition the protein **surface** (relative SASA above a permissive cutoff)
into:

  * pocket regions, as defined by P2Rank (one patch per pocket),
  * non-pocket patches, generated here by greedy region-growing on the
    Cα contact graph with a heavy-atom budget per patch.

Buried residues (not surface-exposed) receive no label. Pockets and patches
are strictly disjoint and there is no buffer between them: every surface
residue ends up either in a pocket or in a patch.

Outputs (under --out-dir, one set per model):
  * ``<model>.patches.tsv``        — per-residue assignment
  * ``<model>.patches.defattr``    — ChimeraX defattr (attr ``patch_id``)
  * ``<model>.patches.cxc``        — ChimeraX command file (color + name)
  * ``<model>.patches.json``       — sidecar metadata, ready for DrugCLIP

CLI:

    python scripts/surface_patches.py \\
        --cif DATA/afdb/models/AF-XXXX-F1-model_v6.cif \\
        --p2rank-residues RESULTS/chembl/p2rank/out/AF-XXXX-...cif_residues.csv \\
        --out-dir RESULTS/chembl/patches/

Or use ``--batch`` to process every model in a directory using its matching
P2Rank residues file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree

# Tien et al. (2013) "max" empirical SASA reference for relative SASA.
# Values in Å² for the residue X in the tripeptide GLY-X-GLY.
MAX_SASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Resolved per the user's clarification: only "pocket atoms" toward the
# DrugCLIP 511 budget, i.e. heavy atoms of the residues making up the
# pocket / patch (no hydrogens). Source: DrugCLIP pocket featurizer.
DEFAULT_ATOM_BUDGET = 128
DEFAULT_SASA_REL_CUTOFF = 0.02  # relative SASA (Tien et al. max)
DEFAULT_SASA_ABS_CUTOFF = 0.0   # Å²  — any residue with non-zero SASA
DEFAULT_CONTACT_CUTOFF = 8.0    # Cα–Cα Å for the residue contact graph
DEFAULT_MIN_PATCH_ATOMS = 40    # post-merge: patches smaller than this get absorbed
DEFAULT_MERGE_ATOM_BUDGET = 511 # merged patch heavy-atom ceiling (DrugCLIP cap)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

ResKey = Tuple[str, int]  # (chain_id, residue_seq_num)


@dataclass
class ResidueInfo:
    key: ResKey
    name: str
    ca: np.ndarray            # Cα coordinate (3,)
    heavy_atoms: List[Atom]   # all non-H atoms
    sasa_total: float = 0.0   # Å²
    sasa_rel: float = 0.0     # 0–1+
    pocket: int = 0           # 0 → not in a P2Rank pocket
    patch: int = 0            # 0 → unassigned (buried or not yet placed)
    is_surface: bool = False

    @property
    def n_heavy(self) -> int:
        return len(self.heavy_atoms)


def _make_parser(path: Path):
    """Pick the right Biopython parser based on file extension."""
    suf = path.suffix.lower()
    if suf in (".pdb", ".ent"):
        return PDBParser(QUIET=True)
    return MMCIFParser(QUIET=True)


def load_structure_residues(struct_path: Path) -> Tuple[Dict[ResKey, ResidueInfo], object]:
    """Parse a .cif or .pdb file into our ResidueInfo dict + raw Structure.

    Skips heteroatoms (HET / waters) — those define ligands, not protein
    residues. Heavy atoms only; hydrogens excluded.
    """
    parser = _make_parser(struct_path)
    structure = parser.get_structure("m", str(struct_path))

    residues: Dict[ResKey, ResidueInfo] = {}
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, resseq, _ = res.id
                if hetflag.strip():
                    continue  # skip hetero / water
                if "CA" not in res:
                    continue
                heavy = [a for a in res.get_atoms() if a.element != "H"]
                if not heavy:
                    continue
                ca = res["CA"].coord.astype(np.float64)
                key = (chain.id, resseq)
                residues[key] = ResidueInfo(
                    key=key, name=res.resname,
                    ca=ca, heavy_atoms=heavy,
                )
        break  # use first model only
    return residues, structure


def load_ligand_atoms(ligand_path: Path) -> np.ndarray:
    """Load heavy-atom coordinates from a ligand PDB/CIF.

    Returns an (N, 3) array. Hydrogens skipped. Accepts both standard
    ``HETATM`` ligand files and ``ATOM``-record ligand files.
    """
    parser = _make_parser(ligand_path)
    struct = parser.get_structure("lig", str(ligand_path))
    coords: List[np.ndarray] = []
    for atom in struct.get_atoms():
        if (atom.element or "").strip().upper() == "H":
            continue
        coords.append(atom.coord.astype(np.float64))
    return np.vstack(coords) if coords else np.empty((0, 3))


def residues_within_ligand(
    residues: Dict[ResKey, ResidueInfo],
    ligand_coords: np.ndarray,
    cutoff: float,
) -> List[ResKey]:
    """Protein residues with any heavy atom within ``cutoff`` Å of any ligand atom."""
    if ligand_coords.size == 0:
        return []
    lig_tree = cKDTree(ligand_coords)
    hits: List[ResKey] = []
    for key, r in residues.items():
        rc = np.vstack([a.coord for a in r.heavy_atoms])
        d, _ = lig_tree.query(rc, k=1)
        if d.min() <= cutoff:
            hits.append(key)
    return hits


def annotate_sasa(structure, residues: Dict[ResKey, ResidueInfo]) -> None:
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(structure, level="R")
    for model in structure:
        for chain in model:
            for res in chain:
                key = (chain.id, res.id[1])
                if key not in residues:
                    continue
                sasa = float(res.sasa) if hasattr(res, "sasa") else 0.0
                ref = MAX_SASA_TIEN.get(res.resname.upper(), 200.0)
                ri = residues[key]
                ri.sasa_total = sasa
                ri.sasa_rel = sasa / ref if ref > 0 else 0.0
        break


def load_p2rank_residues(
    residues_csv: Path,
    residues: Dict[ResKey, ResidueInfo],
) -> None:
    df = pd.read_csv(residues_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for _, row in df.iterrows():
        chain = str(row["chain"]).strip()
        resnum = int(row["residue_label"])
        key = (chain, resnum)
        if key in residues:
            residues[key].pocket = int(row["pocket"])


# ----------------------------------------------------------------------
# Geometry / graph
# ----------------------------------------------------------------------

def build_contact_graph(
    residues: Sequence[ResidueInfo],
    cutoff: float = DEFAULT_CONTACT_CUTOFF,
) -> Dict[ResKey, List[ResKey]]:
    """Cα–Cα adjacency (residues within ``cutoff`` Å of each other)."""
    keys = [r.key for r in residues]
    coords = np.vstack([r.ca for r in residues])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff)
    adj: Dict[ResKey, List[ResKey]] = {k: [] for k in keys}
    for i, j in pairs:
        adj[keys[i]].append(keys[j])
        adj[keys[j]].append(keys[i])
    return adj


# ----------------------------------------------------------------------
# Greedy region growing
# ----------------------------------------------------------------------

def grow_patches(
    surface_residues: Dict[ResKey, ResidueInfo],
    *,
    atom_budget: int = DEFAULT_ATOM_BUDGET,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
    start_patch_id: int = 1,
) -> Dict[ResKey, int]:
    """Assign every non-pocket surface residue to a patch id (≥ start_patch_id).

    Algorithm: pick the unassigned residue farthest (Cα distance) from any
    already-assigned (pocket *or* prior-patch) residue as a seed. From the
    seed, grow by repeatedly adding the unassigned neighbour whose Cα is
    closest to the current patch centroid, stopping when the next residue
    would exceed the heavy-atom budget. Then start a new patch.
    """
    pool = {k: r for k, r in surface_residues.items() if r.pocket == 0}
    if not pool:
        return {}

    # Anchors = surface residues already in a pocket (used for seed distance).
    anchors = [r.ca for r in surface_residues.values() if r.pocket > 0]
    anchor_arr = np.vstack(anchors) if anchors else np.empty((0, 3))

    adj = build_contact_graph(list(pool.values()), cutoff=contact_cutoff)

    assignments: Dict[ResKey, int] = {}
    patch_id = start_patch_id

    pool_keys = list(pool.keys())
    pool_coords = np.vstack([pool[k].ca for k in pool_keys])
    key_to_idx = {k: i for i, k in enumerate(pool_keys)}
    unassigned_mask = np.ones(len(pool_keys), dtype=bool)

    if anchor_arr.size:
        min_dist_to_assigned = cKDTree(anchor_arr).query(pool_coords, k=1)[0]
    else:
        min_dist_to_assigned = np.full(len(pool_keys), np.inf)

    while unassigned_mask.any():
        # Seed: max min-distance to anything already assigned.
        cand_idx = np.where(unassigned_mask)[0]
        seed_local = cand_idx[np.argmax(min_dist_to_assigned[cand_idx])]
        seed = pool_keys[seed_local]

        patch_members: List[ResKey] = [seed]
        atom_count = pool[seed].n_heavy
        unassigned_mask[seed_local] = False
        centroid = pool[seed].ca.copy()

        # Frontier = unassigned neighbours of the patch.
        frontier = {n for n in adj[seed] if unassigned_mask[key_to_idx[n]]}

        while frontier:
            # Pick frontier residue closest to current centroid.
            best = min(frontier, key=lambda k: np.linalg.norm(pool[k].ca - centroid))
            best_n = pool[best].n_heavy
            if atom_count + best_n > atom_budget:
                # Try the smallest frontier residue as a last fit.
                smallest = min(frontier, key=lambda k: pool[k].n_heavy)
                if atom_count + pool[smallest].n_heavy > atom_budget:
                    break
                best = smallest
                best_n = pool[best].n_heavy

            patch_members.append(best)
            atom_count += best_n
            centroid = (centroid * (len(patch_members) - 1) + pool[best].ca) / len(patch_members)
            unassigned_mask[key_to_idx[best]] = False
            frontier.discard(best)
            for nb in adj[best]:
                if unassigned_mask[key_to_idx[nb]]:
                    frontier.add(nb)

        for k in patch_members:
            assignments[k] = patch_id

        # Update min-distance for remaining unassigned residues.
        if unassigned_mask.any():
            new_coords = np.vstack([pool[k].ca for k in patch_members])
            d = cKDTree(new_coords).query(pool_coords, k=1)[0]
            min_dist_to_assigned = np.minimum(min_dist_to_assigned, d)
        patch_id += 1

    return assignments


def merge_small_patches(
    surface_residues: Dict[ResKey, ResidueInfo],
    assignments: Dict[ResKey, int],
    *,
    min_patch_atoms: int = DEFAULT_MIN_PATCH_ATOMS,
    merge_atom_budget: int = DEFAULT_MERGE_ATOM_BUDGET,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
) -> Dict[ResKey, int]:
    """Absorb every patch under ``min_patch_atoms`` heavy atoms into its
    closest graph-adjacent patch, provided the merged patch stays at or
    below ``merge_atom_budget`` heavy atoms.

    Iterates until no further merge fires. Patches with no adjacent
    patch (genuinely isolated stragglers) or whose every neighbour would
    blow the budget are left as-is. Patch ids are renumbered 1..N at the
    end, preserving the original sort order of the survivors.
    """
    if min_patch_atoms <= 0 or not assignments:
        return dict(assignments)

    pool = {k: r for k, r in surface_residues.items() if r.pocket == 0}
    adj = build_contact_graph(list(pool.values()), cutoff=contact_cutoff)
    assignments = dict(assignments)

    while True:
        members: Dict[int, List[ResKey]] = defaultdict(list)
        for k, pid in assignments.items():
            members[pid].append(k)
        atom_counts = {
            pid: sum(pool[k].n_heavy for k in keys)
            for pid, keys in members.items()
        }
        centroids = {
            pid: np.mean(np.vstack([pool[k].ca for k in keys]), axis=0)
            for pid, keys in members.items()
        }
        small_sorted = sorted(
            [pid for pid, n in atom_counts.items() if n < min_patch_atoms],
            key=lambda p: atom_counts[p],
        )
        if not small_sorted:
            break

        merged_any = False
        absorbed: set = set()
        for small_pid in small_sorted:
            if small_pid in absorbed:
                continue
            neighbours: set = set()
            for k in members[small_pid]:
                for nb in adj.get(k, []):
                    nb_pid = assignments.get(nb)
                    if nb_pid and nb_pid != small_pid:
                        neighbours.add(nb_pid)
            candidates = [
                (pid, float(np.linalg.norm(centroids[pid] - centroids[small_pid])))
                for pid in neighbours
                if pid not in absorbed
                and atom_counts[pid] + atom_counts[small_pid] <= merge_atom_budget
            ]
            if not candidates:
                continue
            target_pid = min(candidates, key=lambda x: x[1])[0]
            for k in members[small_pid]:
                assignments[k] = target_pid
            atom_counts[target_pid] += atom_counts[small_pid]
            # Update centroid as size-weighted mean.
            n_small = len(members[small_pid])
            n_target = len(members[target_pid])
            centroids[target_pid] = (
                centroids[target_pid] * n_target
                + centroids[small_pid] * n_small
            ) / (n_target + n_small)
            members[target_pid].extend(members[small_pid])
            absorbed.add(small_pid)
            merged_any = True

        if not merged_any:
            break

    # Renumber so patch ids are contiguous 1..N.
    survivors = sorted(set(assignments.values()))
    remap = {old: new for new, old in enumerate(survivors, start=1)}
    return {k: remap[v] for k, v in assignments.items()}


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------

def _patch_color(idx: int, n: int) -> Tuple[float, float, float]:
    """Distinct HSV-spaced color for patch idx out of n."""
    import colorsys
    h = (idx * 0.6180339887) % 1.0  # golden-ratio hue spacing
    s, v = 0.55, 0.95
    return colorsys.hsv_to_rgb(h, s, v)


def write_per_residue_tsv(residues: Dict[ResKey, ResidueInfo], out_path: Path) -> None:
    rows = []
    for r in residues.values():
        label = (
            f"pocket{r.pocket}" if r.pocket > 0
            else (f"patch{r.patch}" if r.patch > 0 else "")
        )
        rows.append({
            "chain": r.key[0],
            "resnum": r.key[1],
            "resname": r.name,
            "sasa_total": round(r.sasa_total, 2),
            "sasa_rel": round(r.sasa_rel, 3),
            "is_surface": r.is_surface,
            "pocket": r.pocket,
            "patch": r.patch,
            "label": label,
            "n_heavy": r.n_heavy,
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def write_defattr(residues: Dict[ResKey, ResidueInfo], out_path: Path) -> None:
    """ChimeraX defattr: numeric ``patch_id`` per residue.

    Values:
      * 0           — not assigned (buried or skipped)
      * 1..Npocket  — P2Rank pockets (use ``pocket_id`` instead)
      * we encode pocket as negative ids so they're distinguishable visually.

    For simplicity we emit two attributes: ``pocket_id`` and ``patch_id``.
    """
    lines_pocket = ["attribute: pocket_id", "match mode: 1-to-1", "recipient: residues"]
    lines_patch = ["attribute: patch_id", "match mode: 1-to-1", "recipient: residues"]
    for r in residues.values():
        spec = f"/{r.key[0]}:{r.key[1]}"
        lines_pocket.append(f"\t{spec}\t{r.pocket}")
        lines_patch.append(f"\t{spec}\t{r.patch}")
    out_path.write_text("\n".join(lines_pocket + [""] + lines_patch) + "\n")


def write_cxc(
    residues: Dict[ResKey, ResidueInfo],
    out_path: Path,
    *,
    cif_path: Optional[Path] = None,
    ligand_paths: Optional[Sequence[Path]] = None,
) -> None:
    """ChimeraX script that opens the model (if given), colors pockets and
    patches, and creates named selections for each. If ``ligand_paths`` are
    provided they are opened too and shown as sticks.
    """
    lines: List[str] = []
    if cif_path is not None:
        lines.append(f"open {cif_path}")
    lines += [
        "hide #1 target a",
        "show #1 cartoons",
        "color #1 #cccccc",
        "surface #1",
        "transparency 50 target s",
    ]
    # Open bound ligands (if any) as separate models, sticks display.
    for i, lp in enumerate(ligand_paths or [], start=2):
        lines.append(f"open {lp.absolute()}")
        lines.append(f"style #{i} stick")
        lines.append(f"color #{i} byhetero")

    pockets = defaultdict(list)
    patches = defaultdict(list)
    for r in residues.values():
        if r.pocket > 0:
            pockets[r.pocket].append(r.key)
        elif r.patch > 0:
            patches[r.patch].append(r.key)

    n_patches = len(patches)
    legend_entries: List[Tuple[str, str]] = []  # (hex, label)
    for pid in sorted(pockets):
        spec = _residue_spec(pockets[pid])
        rgb = _patch_color(pid * 7, max(n_patches, 1))
        hexc = _rgb_to_hex(rgb)
        n_res = len(pockets[pid])
        lines.append(f"name frozen pocket{pid} {spec}")
        lines.append(f"color pocket{pid} {hexc}")
        legend_entries.append((hexc, f"pocket{pid} ({n_res}r)"))
    for j, pid in enumerate(sorted(patches)):
        spec = _residue_spec(patches[pid])
        rgb = _patch_color(j + 1, n_patches)
        hexc = _rgb_to_hex(rgb)
        n_res = len(patches[pid])
        lines.append(f"name frozen patch{pid} {spec}")
        lines.append(f"color patch{pid} {hexc}")
        legend_entries.append((hexc, f"patch{pid} ({n_res}r)"))

    # Aggregate named selections — one for all pocket residues, one for all
    # patch residues. Lets you do `select pockets` / `select patches` etc.
    all_pocket_keys = [k for k, r in residues.items() if r.pocket > 0]
    all_patch_keys  = [k for k, r in residues.items() if r.patch  > 0]
    if all_pocket_keys:
        lines.append(f"name frozen pockets {_residue_spec(all_pocket_keys)}")
    if all_patch_keys:
        lines.append(f"name frozen patches {_residue_spec(all_patch_keys)}")

    # On-screen legend via 2dlabels — one per entry, colored to match.
    # `key` proved fragile with many entries, so use plain text labels.
    if legend_entries:
        y = 0.96
        dy = 0.026
        for i, (hex_color, label) in enumerate(legend_entries):
            name = f"legend{i:02d}"
            lines.append(
                f'2dlabels create {name} text "■ {label}" '
                f'color {hex_color} xpos 0.01 ypos {y:.3f} size 14'
            )
            y -= dy

    lines.append("view")
    out_path.write_text("\n".join(lines) + "\n")


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{0:02x}{1:02x}{2:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def _residue_spec(keys: Iterable[ResKey]) -> str:
    """Compact ChimeraX residue spec like ``/A:10-12,15,30-32``."""
    by_chain: Dict[str, List[int]] = defaultdict(list)
    for chain, resnum in keys:
        by_chain[chain].append(resnum)
    parts: List[str] = []
    for chain in sorted(by_chain):
        nums = sorted(set(by_chain[chain]))
        runs: List[str] = []
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            runs.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
            i = j + 1
        parts.append(f"/{chain}:" + ",".join(runs))
    return " ".join(parts)


def write_metadata_json(
    residues: Dict[ResKey, ResidueInfo],
    out_path: Path,
    *,
    cif_path: Optional[Path] = None,
    ligand_paths: Optional[Sequence[Path]] = None,
    params: Optional[dict] = None,
) -> None:
    pocket_groups: Dict[int, List[ResidueInfo]] = defaultdict(list)
    patch_groups: Dict[int, List[ResidueInfo]] = defaultdict(list)
    for r in residues.values():
        if r.pocket > 0:
            pocket_groups[r.pocket].append(r)
        elif r.patch > 0:
            patch_groups[r.patch].append(r)

    def summarize(group: List[ResidueInfo]) -> dict:
        coords = np.vstack([np.vstack([a.coord for a in r.heavy_atoms]) for r in group])
        center = coords.mean(axis=0)
        radius = float(np.linalg.norm(coords - center, axis=1).max())
        return {
            "n_residues": len(group),
            "n_heavy_atoms": int(sum(r.n_heavy for r in group)),
            "residues": [
                {"chain": r.key[0], "resnum": r.key[1], "resname": r.name}
                for r in sorted(group, key=lambda x: (x.key[0], x.key[1]))
            ],
            "center_xyz": [float(c) for c in center],
            "bounding_radius": radius,
        }

    payload = {
        "model": str(cif_path) if cif_path else None,
        "ligands": [str(p) for p in (ligand_paths or [])],
        "params": params or {},
        "n_residues_total": len(residues),
        "n_surface_residues": sum(1 for r in residues.values() if r.is_surface),
        "pockets": {str(pid): summarize(g) for pid, g in sorted(pocket_groups.items())},
        "patches": {str(pid): summarize(g) for pid, g in sorted(patch_groups.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

@dataclass
class PatchParams:
    sasa_rel_cutoff: float = DEFAULT_SASA_REL_CUTOFF
    sasa_abs_cutoff: float = DEFAULT_SASA_ABS_CUTOFF
    atom_budget: int = DEFAULT_ATOM_BUDGET
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF
    min_patch_atoms: int = DEFAULT_MIN_PATCH_ATOMS
    merge_atom_budget: int = DEFAULT_MERGE_ATOM_BUDGET


def process_model(
    cif_path: Path,
    p2rank_residues_csv: Path,
    out_dir: Path,
    params: PatchParams = PatchParams(),
) -> Dict[ResKey, ResidueInfo]:
    residues, structure = load_structure_residues(cif_path)
    annotate_sasa(structure, residues)
    load_p2rank_residues(p2rank_residues_csv, residues)

    # Mark surface residues. Permissive: relative SASA above cutoff *or*
    # absolute SASA above a small Å² floor (rescues small residues whose
    # relative SASA is low but which still poke out of the surface), plus
    # any residue P2Rank assigned to a pocket.
    for r in residues.values():
        r.is_surface = (
            r.sasa_rel > params.sasa_rel_cutoff
            or r.sasa_total > params.sasa_abs_cutoff
            or r.pocket > 0
        )

    surface_residues = {k: r for k, r in residues.items() if r.is_surface}
    patch_assignments = grow_patches(
        surface_residues,
        atom_budget=params.atom_budget,
        contact_cutoff=params.contact_cutoff,
    )
    patch_assignments = merge_small_patches(
        surface_residues,
        patch_assignments,
        min_patch_atoms=params.min_patch_atoms,
        merge_atom_budget=params.merge_atom_budget,
        contact_cutoff=params.contact_cutoff,
    )
    for k, pid in patch_assignments.items():
        residues[k].patch = pid

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cif_path.stem  # AF-XXXX-F1-model_v6
    write_per_residue_tsv(residues, out_dir / f"{stem}.patches.tsv")
    write_defattr(residues, out_dir / f"{stem}.patches.defattr")
    write_cxc(residues, out_dir / f"{stem}.patches.cxc", cif_path=cif_path.absolute())
    write_metadata_json(
        residues, out_dir / f"{stem}.patches.json",
        cif_path=cif_path,
        params={
            "sasa_rel_cutoff": params.sasa_rel_cutoff,
            "sasa_abs_cutoff": params.sasa_abs_cutoff,
            "atom_budget": params.atom_budget,
            "contact_cutoff": params.contact_cutoff,
            "min_patch_atoms": params.min_patch_atoms,
            "merge_atom_budget": params.merge_atom_budget,
        },
    )
    return residues


def compute_complex_patches(
    protein_path: Path,
    ligand_paths: Sequence[Path],
    params: PatchParams = PatchParams(),
    *,
    ligand_cutoff: float = 6.0,
) -> Dict[ResKey, ResidueInfo]:
    """In-memory patch segmentation for one (protein, ligands) complex.

    The observed pocket = union of protein residues whose heavy atoms come
    within ``ligand_cutoff`` Å of *any* heavy atom of *any* bound ligand.
    All other surface residues are partitioned into patches by the same
    greedy + merge pipeline used for AFDB models. Returns the residue
    dict with ``pocket`` / ``patch`` / ``is_surface`` fields populated —
    no files written.
    """
    residues, structure = load_structure_residues(protein_path)
    annotate_sasa(structure, residues)

    # Union all ligand heavy-atom coords.
    coord_chunks: List[np.ndarray] = []
    for lp in ligand_paths:
        c = load_ligand_atoms(lp)
        if c.size:
            coord_chunks.append(c)
    if coord_chunks:
        ligand_coords = np.vstack(coord_chunks)
        for k in residues_within_ligand(residues, ligand_coords, cutoff=ligand_cutoff):
            residues[k].pocket = 1

    # Same surface criterion as the AFDB pipeline.
    for r in residues.values():
        r.is_surface = (
            r.sasa_rel > params.sasa_rel_cutoff
            or r.sasa_total > params.sasa_abs_cutoff
            or r.pocket > 0
        )

    surface_residues = {k: r for k, r in residues.items() if r.is_surface}
    patch_assignments = grow_patches(
        surface_residues,
        atom_budget=params.atom_budget,
        contact_cutoff=params.contact_cutoff,
    )
    patch_assignments = merge_small_patches(
        surface_residues,
        patch_assignments,
        min_patch_atoms=params.min_patch_atoms,
        merge_atom_budget=params.merge_atom_budget,
        contact_cutoff=params.contact_cutoff,
    )
    for k, pid in patch_assignments.items():
        residues[k].patch = pid
    return residues


def process_complex(
    protein_path: Path,
    ligand_paths: Sequence[Path],
    out_dir: Path,
    params: PatchParams = PatchParams(),
    *,
    ligand_cutoff: float = 6.0,
    output_stem: Optional[str] = None,
) -> Dict[ResKey, ResidueInfo]:
    """Bench-dataset entry point (COACH420 / HOLO4K / PDBBIND / scPDB).

    Thin wrapper around :func:`compute_complex_patches` that also writes
    the per-complex .tsv / .defattr / .cxc / .json outputs.
    """
    residues = compute_complex_patches(
        protein_path=protein_path,
        ligand_paths=ligand_paths,
        params=params,
        ligand_cutoff=ligand_cutoff,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or protein_path.parent.name or protein_path.stem
    write_per_residue_tsv(residues, out_dir / f"{stem}.patches.tsv")
    write_defattr(residues, out_dir / f"{stem}.patches.defattr")
    write_cxc(
        residues, out_dir / f"{stem}.patches.cxc",
        cif_path=protein_path.absolute(),
        ligand_paths=[lp.absolute() for lp in ligand_paths],
    )
    write_metadata_json(
        residues, out_dir / f"{stem}.patches.json",
        cif_path=protein_path,
        ligand_paths=ligand_paths,
        params={
            "sasa_rel_cutoff": params.sasa_rel_cutoff,
            "sasa_abs_cutoff": params.sasa_abs_cutoff,
            "atom_budget": params.atom_budget,
            "contact_cutoff": params.contact_cutoff,
            "min_patch_atoms": params.min_patch_atoms,
            "merge_atom_budget": params.merge_atom_budget,
            "ligand_cutoff": ligand_cutoff,
            "n_ligands": len(ligand_paths),
        },
    )
    return residues


def _default_p2rank_residues_for(cif_path: Path, p2rank_dir: Path) -> Path:
    return p2rank_dir / f"{cif_path.name}_residues.csv"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cif", type=Path, help="AFDB mmCIF model")
    p.add_argument("--p2rank-residues", type=Path, help="P2Rank *_residues.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-cif-dir", type=Path, help="Directory of .cif models")
    p.add_argument("--batch-p2rank-dir", type=Path, help="Directory of P2Rank outputs")
    p.add_argument("--sasa-rel-cutoff", type=float, default=DEFAULT_SASA_REL_CUTOFF)
    p.add_argument("--sasa-abs-cutoff", type=float, default=DEFAULT_SASA_ABS_CUTOFF,
                   help="absolute SASA Å² floor; any residue at or above is surface")
    p.add_argument("--atom-budget", type=int, default=DEFAULT_ATOM_BUDGET)
    p.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF)
    p.add_argument("--min-patch-atoms", type=int, default=DEFAULT_MIN_PATCH_ATOMS,
                   help="patches under this heavy-atom count are merged into their closest neighbour; 0 disables merging")
    p.add_argument("--merge-atom-budget", type=int, default=DEFAULT_MERGE_ATOM_BUDGET,
                   help="upper heavy-atom limit for a merged patch (DrugCLIP cap)")
    args = p.parse_args(argv)

    params = PatchParams(
        sasa_rel_cutoff=args.sasa_rel_cutoff,
        sasa_abs_cutoff=args.sasa_abs_cutoff,
        atom_budget=args.atom_budget,
        contact_cutoff=args.contact_cutoff,
        min_patch_atoms=args.min_patch_atoms,
        merge_atom_budget=args.merge_atom_budget,
    )

    if args.batch_cif_dir:
        assert args.batch_p2rank_dir, "--batch-p2rank-dir required with --batch-cif-dir"
        cifs = sorted(args.batch_cif_dir.glob("*.cif"))
        from tqdm import tqdm
        for cif in tqdm(cifs):
            p2r = _default_p2rank_residues_for(cif, args.batch_p2rank_dir)
            if not p2r.exists():
                continue
            try:
                process_model(cif, p2r, args.out_dir, params)
            except Exception as e:
                print(f"[warn] {cif.name}: {e}")
    else:
        assert args.cif and args.p2rank_residues, "Provide --cif and --p2rank-residues"
        process_model(args.cif, args.p2rank_residues, args.out_dir, params)


if __name__ == "__main__":
    main()
