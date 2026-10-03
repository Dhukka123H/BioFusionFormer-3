"""
scripts/inspect_reactome.py
---------------------------
Load the Reactome flat-files and print human-only statistics:
  - Number of pathways (Homo sapiens)
  - Number of unique Ensembl gene IDs
  - Number of unique miRNA IDs
  - Number of gene-pathway pairs
  - Pathway hierarchy depth (longest root-to-leaf path)
"""

import sys
import gzip
import zipfile
from pathlib import Path
from collections import defaultdict, deque

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
REACTOME_DIR = ROOT / "data" / "raw" / "reactome"

ENSEMBL2REACTOME        = REACTOME_DIR / "Ensembl2Reactome.txt"
ENSEMBL2REACTOME_ALL    = REACTOME_DIR / "Ensembl2Reactome_All_Levels.txt"
MIRBASE2REACTOME        = REACTOME_DIR / "miRBase2Reactome.txt"
MIRBASE2REACTOME_ALL    = REACTOME_DIR / "miRBase2Reactome_All_Levels.txt"
REACTOME_PATHWAYS       = REACTOME_DIR / "ReactomePathways.txt"
REACTOME_RELATION       = REACTOME_DIR / "ReactomePathwaysRelation.txt"

# ── Helpers ───────────────────────────────────────────────────────────────────

def open_file(path: Path):
    """Open plain, .gz, or (single-entry) .zip files transparently."""
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner = zf.namelist()[0]
        return zf.open(inner)
    return open(path, "r", encoding="utf-8", errors="replace")


def check_file(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"  [MISSING] {label}: {path}")
        return False
    if path.stat().st_size == 0:
        print(f"  [EMPTY]   {label}: {path}")
        return False
    return True


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pathway_ids(path: Path) -> set:
    """
    ReactomePathways.txt columns:
      stable_id  <tab>  name  <tab>  species
    Return set of stable_ids for Homo sapiens.
    """
    ids = set()
    with open_file(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            stable_id, _name, species = parts[0], parts[1], parts[2]
            if "Homo sapiens" in species:
                ids.add(stable_id)
    return ids


def load_pathway_names(path: Path) -> dict:
    """Return {stable_id: name} for Homo sapiens."""
    names = {}
    with open_file(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            stable_id, name, species = parts[0], parts[1], parts[2]
            if "Homo sapiens" in species:
                names[stable_id] = name
    return names


def load_gene_pathway_pairs(path: Path, human_pathways: set) -> tuple[set, set]:
    """
    Ensembl2Reactome*.txt columns:
      gene_id  pathway_id  url  pathway_name  evidence_code  species
    Returns (set of Ensembl IDs, set of (gene_id, pathway_id) pairs) for human.
    """
    genes = set()
    pairs = set()
    with open_file(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            gene_id, pathway_id, _url, _name, _ev, species = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            )
            if "Homo sapiens" not in species:
                continue
            if pathway_id not in human_pathways:
                continue
            genes.add(gene_id)
            pairs.add((gene_id, pathway_id))
    return genes, pairs


def load_mirna_pathway_pairs(path: Path, human_pathways: set) -> tuple[set, set]:
    """
    miRBase2Reactome*.txt — same column layout as Ensembl2Reactome.
    Returns (set of miRNA IDs, set of (mirna_id, pathway_id) pairs) for human.
    """
    mirnas = set()
    pairs = set()
    with open_file(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            mirna_id, pathway_id, _url, _name, _ev, species = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            )
            if "Homo sapiens" not in species:
                continue
            if pathway_id not in human_pathways:
                continue
            mirnas.add(mirna_id)
            pairs.add((mirna_id, pathway_id))
    return mirnas, pairs


def load_hierarchy(path: Path, human_pathways: set) -> tuple[dict, dict]:
    """
    ReactomePathwaysRelation.txt columns: parent_id  child_id
    Returns (parent→children, child→parents) restricted to human pathways.
    """
    children: dict[str, set] = defaultdict(set)
    parents:  dict[str, set] = defaultdict(set)
    with open_file(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            parent, child = parts[0], parts[1]
            if parent in human_pathways and child in human_pathways:
                children[parent].add(child)
                parents[child].add(parent)
    return dict(children), dict(parents)


def compute_max_depth(
    children: dict,
    parents: dict,
    all_nodes: set,
) -> int:
    """
    BFS from all root nodes (nodes with no parent).
    Returns the maximum depth (root = 0).
    """
    roots = {n for n in all_nodes if n not in parents or not parents[n]}
    if not roots:
        # fall back: find nodes not appearing as any child
        all_children = {c for cs in children.values() for c in cs}
        roots = all_nodes - all_children

    depth = {r: 0 for r in roots}
    queue = deque(roots)
    max_d = 0

    while queue:
        node = queue.popleft()
        d = depth[node]
        for child in children.get(node, []):
            if child not in depth or depth[child] < d + 1:
                depth[child] = d + 1
                max_d = max(max_d, d + 1)
                queue.append(child)

    return max_d


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Reactome Inspection  —  Homo sapiens only")
    print("=" * 60)

    # 0. Verify files
    required = [
        (REACTOME_PATHWAYS,    "ReactomePathways.txt"),
        (REACTOME_RELATION,    "ReactomePathwaysRelation.txt"),
        (ENSEMBL2REACTOME,     "Ensembl2Reactome.txt"),
        (ENSEMBL2REACTOME_ALL, "Ensembl2Reactome_All_Levels.txt"),
        (MIRBASE2REACTOME,     "miRBase2Reactome.txt"),
        (MIRBASE2REACTOME_ALL, "miRBase2Reactome_All_Levels.txt"),
    ]
    missing = [lbl for path, lbl in required if not check_file(path, lbl)]
    if missing:
        print(f"\n[ERROR] {len(missing)} required file(s) missing. Run download_reactome.py first.")
        sys.exit(1)

    # 1. Human pathway IDs
    print("\n[1] Loading human pathway IDs from ReactomePathways.txt …")
    human_pathways = load_pathway_ids(REACTOME_PATHWAYS)
    print(f"    Human pathways : {len(human_pathways):,}")

    # 2. Gene-pathway pairs (direct mappings only)
    print("[2] Loading gene-pathway pairs (direct) …")
    genes_direct, pairs_direct = load_gene_pathway_pairs(ENSEMBL2REACTOME, human_pathways)
    print(f"    Ensembl genes (direct) : {len(genes_direct):,}")
    print(f"    Gene-pathway pairs (direct) : {len(pairs_direct):,}")

    # 3. Gene-pathway pairs (all levels, including inferred)
    print("[3] Loading gene-pathway pairs (all levels) …")
    genes_all, pairs_all = load_gene_pathway_pairs(ENSEMBL2REACTOME_ALL, human_pathways)
    print(f"    Ensembl genes (all levels) : {len(genes_all):,}")
    print(f"    Gene-pathway pairs (all levels) : {len(pairs_all):,}")

    # 4. miRNA-pathway pairs (direct)
    print("[4] Loading miRNA-pathway pairs (direct) …")
    mirnas_direct, mirna_pairs_direct = load_mirna_pathway_pairs(MIRBASE2REACTOME, human_pathways)
    print(f"    miRNA IDs (direct) : {len(mirnas_direct):,}")
    print(f"    miRNA-pathway pairs (direct) : {len(mirna_pairs_direct):,}")

    # 5. miRNA-pathway pairs (all levels)
    print("[5] Loading miRNA-pathway pairs (all levels) …")
    mirnas_all, mirna_pairs_all = load_mirna_pathway_pairs(MIRBASE2REACTOME_ALL, human_pathways)
    print(f"    miRNA IDs (all levels) : {len(mirnas_all):,}")
    print(f"    miRNA-pathway pairs (all levels) : {len(mirna_pairs_all):,}")

    # 6. Pathway hierarchy
    print("[6] Building pathway hierarchy …")
    children_map, parents_map = load_hierarchy(REACTOME_RELATION, human_pathways)
    max_depth = compute_max_depth(children_map, parents_map, human_pathways)
    n_relations = sum(len(v) for v in children_map.values())
    roots_count = sum(
        1 for n in human_pathways
        if n not in parents_map or not parents_map[n]
    )
    leaves_count = sum(
        1 for n in human_pathways
        if n not in children_map or not children_map[n]
    )

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS  (Homo sapiens only)")
    print("=" * 60)
    print(f"  Pathways total            : {len(human_pathways):>8,}")
    print(f"  Root pathways (top level) : {roots_count:>8,}")
    print(f"  Leaf pathways             : {leaves_count:>8,}")
    print(f"  Parent-child relations    : {n_relations:>8,}")
    print(f"  Hierarchy max depth       : {max_depth:>8}")
    print()
    print(f"  Ensembl genes (direct)    : {len(genes_direct):>8,}")
    print(f"  Ensembl genes (all levels): {len(genes_all):>8,}")
    print(f"  Gene-pathway pairs (direct)    : {len(pairs_direct):>8,}")
    print(f"  Gene-pathway pairs (all levels): {len(pairs_all):>8,}")
    print()
    print(f"  miRNA IDs (direct)        : {len(mirnas_direct):>8,}")
    print(f"  miRNA IDs (all levels)    : {len(mirnas_all):>8,}")
    print(f"  miRNA-pathway pairs (direct)    : {len(mirna_pairs_direct):>8,}")
    print(f"  miRNA-pathway pairs (all levels): {len(mirna_pairs_all):>8,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
