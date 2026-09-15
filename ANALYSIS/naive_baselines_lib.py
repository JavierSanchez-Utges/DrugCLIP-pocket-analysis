"""
Helper functions for the naive/baseline pocket-matching scores in
04_naive_baselines.ipynb:

  1. same_protein   - 1/0, full receptor sequence identical
  2. same_ligand     - 1/0, bound ligand is the same chemical compound
  3. pocket_seqid    - sequence identity of pocket-lining residues after
                       global alignment of the two full receptor sequences
  4. ligand_tanimoto - Morgan-fingerprint Tanimoto similarity of the two
                       bound ligands

Ligand identity is recovered by round-tripping to RCSB where possible: the
split "*_LIG.pdb" files have their HETATM residue name rewritten to a
generic "LIG", but chain + resSeq survive the split, so the true
chemical-component id can be read off the original deposited PDB file at
the same (chain, resSeq). The recovered id is then used to fetch canonical
SMILES from the RCSB Chemical Component Dictionary.

Two PROSPECCTS subsets use structure ids that are not real, resolvable PDB
codes: NMR_structures (cz00A, di00A, ...) and roughly 83% of the
decoy_structures pool (A100A, A101A, ... - synthetic decoy placeholders
constructed by the benchmark, never deposited to the PDB). For these, and
for any other id where the RCSB round-trip fails for some other reason
(chain/resSeq renumbered since deposition, etc.), ligand identity falls
back to local, coordinate-based bond perception (RDKit `proximityBonding`)
on the stripped ligand atoms - no internet needed, but bond orders/
stereochemistry are approximate. `resolve_ligand_identity()` tries RCSB
first and applies this fallback automatically.

All network calls are cached to disk under DATA/rcsb_cache/ so repeat runs
are free.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

_CANDIDATE_ROOTS = [
    Path(__file__).resolve().parent.parent / 'DATA' / 'pocket_benchmark',
]


def resolve_benchmark_dir() -> Path:
    for root in _CANDIDATE_ROOTS:
        if (root / 'DATA' / 'PROSPECCTS_pairs').is_dir():
            return root.resolve()
    raise FileNotFoundError('Could not locate POCKET_MATCHING_BENCHMARK root (no DATA/PROSPECCTS_pairs found).')


BENCHMARK_DIR = resolve_benchmark_dir()
DATA_DIR = BENCHMARK_DIR / 'DATA'
SPLIT_PDBS_DIR = DATA_DIR / 'PROSPECCTS_split_pdbs'
RESULTS_DIR = BENCHMARK_DIR / 'RESULTS'
CACHE_DIR = DATA_DIR / 'rcsb_cache'
PDB_CACHE_DIR = CACHE_DIR / 'pdb'
CHEMCOMP_CACHE_DIR = CACHE_DIR / 'chemcomp'
for d in (PDB_CACHE_DIR, CHEMCOMP_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# dkey -> subdirectory of DATA/PROSPECCTS_split_pdbs holding <id>_protein.pdb
# / <id>_LIG.pdb for that dataset. Mirrors the `nrgrank_subdir`/`epocs_name`
# fields of the DATASETS dict in 02_benchmark_methods.ipynb.
DKEY_SPLIT_SUBDIR = {
    'D1':   'identical_structures/identical_structures',
    'D1.2': 'identical_structures_similar_ligands/identical_structures_similar_ligands',
    'D2':   'NMR_structures/NMR_structures',
    'D3':   'decoy/decoy_rational_structures',
    'D4':   'decoy/decoy_shape_structures',
    'D5':   'kahraman_structures/kahraman_structures',
    'D5.2': 'kahraman_structures/kahraman_structures',
    'D6':   'barelier_structures/barelier_structures',
    'D7':   'review_structures/review_structures',
}
NMR_DKEYS = {'D2'}

# --------------------------------------------------------------------------
# fixed-column PDB parsing
# --------------------------------------------------------------------------

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}


def parse_protein_residues(pdb_path: Path):
    """Parse ATOM records of a single-chain protein PDB.

    Returns (seq, index_map, atoms_by_res):
      seq          - one-letter sequence string, in file order
      index_map    - dict[(chain, resseq, icode)] -> 0-based index into seq
      atoms_by_res - dict[(chain, resseq, icode)] -> Nx3 float array of all
                      (non-alt/altloc-A) atom coordinates for that residue
    """
    seq_chars = []
    index_map = {}
    coords_by_res: dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM  '):
                continue
            altloc = line[16]
            if altloc not in (' ', 'A'):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode)
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            coords_by_res.setdefault(key, []).append((x, y, z))

            atom_name = line[12:16].strip()
            if atom_name == 'CA' and key not in index_map:
                index_map[key] = len(seq_chars)
                seq_chars.append(THREE_TO_ONE.get(resname, 'X'))

    atoms_by_res = {k: np.asarray(v, dtype=float) for k, v in coords_by_res.items()}
    return ''.join(seq_chars), index_map, atoms_by_res


def parse_hetatm_groups(pdb_path: Path):
    """Group HETATM records of a *_LIG.pdb file by (chain, resseq, icode).

    Returns dict[key] -> list of (atom_name, element, x, y, z).
    """
    groups: dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            altloc = line[16]
            if altloc not in (' ', 'A'):
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode)
            atom_name = line[12:16].strip()
            element = line[76:78].strip() or atom_name[0]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            groups.setdefault(key, []).append((atom_name, element, x, y, z))
    return groups


def primary_ligand_group(pdb_path: Path):
    """Pick the largest HETATM group in a *_LIG.pdb (multi-fragment ligand files
    are rare; this discards small satellite groups, e.g. mis-split ions)."""
    groups = parse_hetatm_groups(pdb_path)
    if not groups:
        return None, []
    key = max(groups, key=lambda k: len(groups[k]))
    return key, groups[key]


# --------------------------------------------------------------------------
# pocket residues (distance cutoff from primary ligand)
# --------------------------------------------------------------------------

POCKET_CUTOFF_A = 6.0


def compute_pocket_residues(atoms_by_res: dict, ligand_atoms: list, cutoff: float = POCKET_CUTOFF_A):
    """Return the set of residue keys with any atom within `cutoff` A of any
    ligand atom."""
    if not ligand_atoms or not atoms_by_res:
        return set()
    lig_xyz = np.asarray([(a[2], a[3], a[4]) for a in ligand_atoms], dtype=float)
    pocket = set()
    for key, res_xyz in atoms_by_res.items():
        d = np.linalg.norm(res_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1)
        if d.min() <= cutoff:
            pocket.add(key)
    return pocket


# --------------------------------------------------------------------------
# RCSB round-trip: recover the real ligand chemical-component id
# --------------------------------------------------------------------------

_PDB_CODE_RE = re.compile(r'^[0-9][A-Za-z0-9]{3}$')


def looks_like_pdb_code(code: str) -> bool:
    """PDB codes are 4 chars, always starting with a digit. Synthetic ids
    used for NMR ensembles (cz00, di00, ...) and decoy placeholders (A100,
    A101, ...) don't match this, so there's no point hitting RCSB for them."""
    return bool(_PDB_CODE_RE.match(code))


_HTTP_TIMEOUT = 20
_HTTP_RETRIES = 3


def _http_get(url: str) -> bytes | None:
    for attempt in range(_HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_original_pdb(pdb_code: str) -> Path | None:
    """Download (or reuse cached) the original deposited PDB file."""
    pdb_code = pdb_code.upper()
    if not looks_like_pdb_code(pdb_code):
        return None
    cache_path = PDB_CACHE_DIR / f'{pdb_code}.pdb'
    miss_marker = PDB_CACHE_DIR / f'{pdb_code}.missing'
    if cache_path.is_file():
        return cache_path
    if miss_marker.is_file():
        return None
    data = _http_get(f'https://files.rcsb.org/download/{pdb_code}.pdb')
    if data is None:
        miss_marker.touch()
        return None
    cache_path.write_bytes(data)
    return cache_path


def recover_ligand_code(pdb_code: str, chain: str, resseq: str) -> str | None:
    """Look up the real 3-5 char chemical-component id for (chain, resseq) in
    the original PDB file."""
    original = fetch_original_pdb(pdb_code)
    if original is None:
        return None
    target_resseq = resseq.strip()
    with open(original) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            if line[21] != chain:
                continue
            if line[22:26].strip() != target_resseq:
                continue
            return line[17:20].strip()
    return None


def prefetch(codes, fetch_fn, max_workers: int = 16, label: str = 'items'):
    """Warm the disk cache for a collection of codes in parallel (network I/O
    bound, so threads are fine despite the GIL)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    codes = sorted(set(codes))
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_fn, c): c for c in codes}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 250 == 0 or done == len(codes):
                print(f'  prefetched {done}/{len(codes)} {label}')


def fetch_chemcomp_smiles(ligand_code: str) -> str | None:
    """Fetch (or reuse cached) canonical SMILES for a chemical-component id
    from the RCSB Chemical Component Dictionary."""
    cache_path = CHEMCOMP_CACHE_DIR / f'{ligand_code}.json'
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text())
    else:
        data = _http_get(f'https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code}')
        if data is None:
            cache_path.write_text('{}')
            return None
        payload = json.loads(data)
        cache_path.write_text(json.dumps(payload))

    descriptors = payload.get('pdbx_chem_comp_descriptor', [])
    smiles_candidates = [d['descriptor'] for d in descriptors if d.get('type') in ('SMILES_CANONICAL', 'SMILES')]
    for smi in smiles_candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return smi
    return None


# --------------------------------------------------------------------------
# ligand mols / fingerprints
# --------------------------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles)


def mol_from_local_bond_perception(ligand_atoms: list):
    """Fallback when the RCSB round-trip isn't possible: build an RDKit mol
    straight from the stripped local coordinates via distance-based bond
    perception (no internet, no bond orders from hydrogens - approximate by
    construction)."""
    if len(ligand_atoms) < 2:
        return None
    lines = ['HEADER', 'COMPND']
    for i, (name, element, x, y, z) in enumerate(ligand_atoms, start=1):
        elem = (element or name[0]).strip()[:2].rjust(2)
        lines.append(
            f'HETATM{i:5d} {name:<4s} LIG A   1    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem}'
        )
    lines.append('END')
    block = '\n'.join(lines)
    mol = Chem.MolFromPDBBlock(block, sanitize=True, removeHs=True, proximityBonding=True)
    if mol is None:
        mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True, proximityBonding=True)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
    return mol


def morgan_fp(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_NBITS)
    except Exception:
        return None


def tanimoto(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return float('nan')
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def resolve_ligand_identity(pdb_code: str, lig_key, ligand_atoms: list) -> dict:
    """RCSB round-trip first (authoritative chem-comp id + CCD SMILES);
    local bond perception as a fallback whenever that isn't possible
    (synthetic id, download miss, or chain/resSeq no longer matches the
    current deposition)."""
    lig_code = None
    mol = None
    if lig_key is not None and looks_like_pdb_code(pdb_code):
        chain, resseq, _icode = lig_key
        lig_code = recover_ligand_code(pdb_code, chain, resseq)
        if lig_code:
            smiles = fetch_chemcomp_smiles(lig_code)
            mol = mol_from_smiles(smiles) if smiles else None
    if mol is None:
        mol = mol_from_local_bond_perception(ligand_atoms)
    return {'lig_code': lig_code, 'fp': morgan_fp(mol)}


def same_ligand_score(struct_a: dict, struct_b: dict, exact_tanimoto: float = 0.999) -> float:
    """1/0 same bound ligand. Prefers exact chem-comp id match (authoritative,
    available when both sides resolved via RCSB); falls back to a
    near-exact fingerprint match when at least one side only has a locally
    bond-perceived mol. NaN if neither side has any usable ligand info."""
    if struct_a['lig_code'] and struct_b['lig_code']:
        return 1.0 if struct_a['lig_code'] == struct_b['lig_code'] else 0.0
    tan = tanimoto(struct_a['fp'], struct_b['fp'])
    if np.isnan(tan):
        return float('nan')
    return 1.0 if tan >= exact_tanimoto else 0.0


# --------------------------------------------------------------------------
# pocket sequence identity via global alignment
# --------------------------------------------------------------------------

_BLOSUM62 = substitution_matrices.load('BLOSUM62')
_ALIGNER = PairwiseAligner()
_ALIGNER.substitution_matrix = _BLOSUM62
_ALIGNER.mode = 'global'
_ALIGNER.open_gap_score = -10
_ALIGNER.extend_gap_score = -0.5


def pocket_sequence_identity(seq_a: str, seq_b: str, pocket_idx_a: set, pocket_idx_b: set) -> float:
    """Sequence identity restricted to pocket-lining positions.

    Aligns the two full sequences globally, then for each protein's pocket
    residues checks whether the aligned partner position (if not a gap) is
    an identical residue. The two one-sided fractions are averaged for a
    symmetric score.
    """
    if not seq_a or not seq_b:
        return float('nan')
    if seq_a == seq_b:
        return 1.0

    alignment = _ALIGNER.align(seq_a, seq_b)[0]
    blocks_a, blocks_b = alignment.aligned  # aligned (non-gap-on-either-side) blocks

    map_a_to_b: dict[int, int] = {}
    for (sa, ea), (sb, eb) in zip(blocks_a, blocks_b):
        for offset in range(ea - sa):
            map_a_to_b[sa + offset] = sb + offset
    map_b_to_a = {v: k for k, v in map_a_to_b.items()}

    def one_sided(pocket_idx, mapping, seq_from, seq_to):
        aligned = [(i, mapping[i]) for i in pocket_idx if i in mapping]
        if not aligned:
            return float('nan')
        matches = sum(1 for i, j in aligned if seq_from[i] == seq_to[j])
        return matches / len(aligned)

    frac_a = one_sided(pocket_idx_a, map_a_to_b, seq_a, seq_b)
    frac_b = one_sided(pocket_idx_b, map_b_to_a, seq_b, seq_a)

    fracs = [f for f in (frac_a, frac_b) if not np.isnan(f)]
    if not fracs:
        return float('nan')
    return float(np.mean(fracs))
