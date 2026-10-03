"""
Phase A2 - Pathway graph construction
=======================================
Builds three CSV files in data/interim/reactome/:
  pathway_nodes.csv  - id, name, depth, is_root, is_leaf
  pathway_edges.csv  - parent, child (human-only DAG)
  gene_pathway_edges.csv - gene_id, pathway_id (all-levels, ENSG only)

Also verifies DAG (no cycles) and reports orphans.
"""
import sys
import logging
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "reactome"
OUT  = ROOT / "data" / "interim" / "reactome"
OUT.mkdir(parents=True, exist_ok=True)
LOG  = ROOT / "logs" / "phase_a2.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("A2")


def load_pathways(path: Path) -> dict:
    """Returns {id: (name, species)}"""
    pathways = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = (parts[1], parts[2])
    return pathways


def load_relations(path: Path, human_ids: set) -> list[tuple[str, str]]:
    """Returns list of (parent, child) restricted to human pathways."""
    edges = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                p, c = parts[0], parts[1]
                if p in human_ids and c in human_ids:
                    edges.append((p, c))
    return edges


def compute_depths(children: dict, human_ids: set) -> dict:
    """BFS from roots; returns {id: depth}."""
    all_children_set = {c for cs in children.values() for c in cs}
    roots = {n for n in human_ids if n not in all_children_set}
    depth = {r: 0 for r in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            if child not in depth or depth[child] < depth[node] + 1:
                depth[child] = depth[node] + 1
                queue.append(child)
    return depth


def detect_cycles(children: dict, nodes: set) -> list:
    """DFS-based cycle detection. Returns list of cycle-containing node IDs."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    cycles = []

    def dfs(v):
        color[v] = GRAY
        for w in children.get(v, []):
            if color[w] == GRAY:
                cycles.append(v)
            elif color[w] == WHITE:
                dfs(w)
        color[v] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return cycles


def main():
    log.info("Loading ReactomePathways.txt ...")
    all_pathways = load_pathways(RAW / "ReactomePathways.txt")
    human_ids = {pid for pid, (name, sp) in all_pathways.items()
                 if "Homo sapiens" in sp}
    log.info("Human pathway IDs: %d", len(human_ids))

    log.info("Loading ReactomePathwaysRelation.txt ...")
    edges = load_relations(RAW / "ReactomePathwaysRelation.txt", human_ids)
    log.info("Human pathway edges: %d", len(edges))

    # Build adjacency
    children: dict[str, set] = defaultdict(set)
    parents: dict[str, set]  = defaultdict(set)
    for p, c in edges:
        children[p].add(c)
        parents[c].add(p)

    # Depths
    depth_map = compute_depths(dict(children), human_ids)

    # Fill in any nodes not reached by BFS (disconnected)
    for pid in human_ids:
        if pid not in depth_map:
            depth_map[pid] = -1  # orphan / disconnected

    is_root = {pid: (pid not in {c for _, c in edges}) for pid in human_ids}
    is_leaf  = {pid: (pid not in {p for p, _ in edges}) for pid in human_ids}

    # Cycle detection
    cycles = detect_cycles(dict(children), human_ids)
    log.info("Cycles detected: %d", len(cycles))

    # Orphans (depth == -1)
    orphans = [pid for pid, d in depth_map.items() if d == -1]
    log.info("Orphans (unreachable from roots): %d", len(orphans))

    # ── pathway_nodes.csv ────────────────────────────────────────────────────
    node_rows = []
    for pid in sorted(human_ids):
        name, _ = all_pathways[pid]
        node_rows.append({
            "id": pid,
            "name": name,
            "depth": depth_map.get(pid, -1),
            "is_root": is_root[pid],
            "is_leaf": is_leaf[pid],
        })
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(OUT / "pathway_nodes.csv", index=False)
    log.info("Saved pathway_nodes.csv (%d rows)", len(nodes_df))

    # ── pathway_edges.csv ────────────────────────────────────────────────────
    edges_df = pd.DataFrame(edges, columns=["parent", "child"])
    edges_df.to_csv(OUT / "pathway_edges.csv", index=False)
    log.info("Saved pathway_edges.csv (%d rows)", len(edges_df))

    # ── gene_pathway_edges.csv ───────────────────────────────────────────────
    gp = ROOT / "data" / "interim" / "reactome" / "gene2pathway.parquet"
    if gp.exists():
        gp_df = pd.read_parquet(gp)
        al_df = gp_df[gp_df["level"] == "all_levels"][["gene_id", "pathway_id"]].drop_duplicates()
        al_df.to_csv(OUT / "gene_pathway_edges.csv", index=False)
        log.info("Saved gene_pathway_edges.csv (%d rows)", len(al_df))
    else:
        log.warning("gene2pathway.parquet not found — run A1 first")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n=== A2 RESULTS ===")
    print(f"  Pathway nodes          : {len(nodes_df):,}")
    print(f"  Pathway edges (DAG)    : {len(edges_df):,}")
    print(f"  Root nodes             : {sum(is_root.values()):,}")
    print(f"  Leaf nodes             : {sum(is_leaf.values()):,}")
    print(f"  Max depth              : {max(d for d in depth_map.values() if d >= 0)}")
    print(f"  Cycles                 : {len(cycles)}")
    print(f"  Orphans                : {len(orphans)}")
    if orphans:
        print(f"  (first 5 orphans): {orphans[:5]}")
    print(f"  DAG valid              : {len(cycles) == 0}")
    if gp.exists():
        print(f"  Gene-pathway edges     : {len(al_df):,}")


if __name__ == "__main__":
    main()
