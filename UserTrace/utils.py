import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Any, Set, List, Tuple, Optional, Union
import logging
import igraph as ig
import leidenalg as la

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Utils")

# -------------------- Utility Functions --------------------
def save_json(path: str, obj: Any):
    """Save object to JSON file with proper directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    """Load object from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def connected_components_undirected(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    """Find connected components in an undirected graph."""
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    
    for node in graph.keys():
        if node in seen:
            continue
        
        q = deque([node])
        group: Set[str] = set([node])
        seen.add(node)
        
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    group.add(v)
                    q.append(v)
        
        comps.append(group)
    
    return comps

def detect_file_communities_leiden(
    file_graph: Dict[str, Set[str]],
    edge_details: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None,
    resolution: float = 1.0,
    iterations: Optional[int] = None,
    seed: Optional[int] = 42,
) -> List[Set[str]]:
    """
    Detect file communities using Leiden algorithm.
    Falls back to connected components if Leiden is unavailable.
    """
    try:
        import igraph as ig
        import leidenalg as la
    except Exception as e:
        logger.warning(f"Leiden unavailable ({e}); falling back to connected components.")
        comms = connected_components_undirected(file_graph)
        return sorted(comms, key=lambda g: (-len(g), sorted(list(g))[0] if g else ""))
    
    # Build vertex list and index mapping
    vertices: Set[str] = set(file_graph.keys())
    for u, nbrs in file_graph.items():
        vertices.update(nbrs)
    
    vlist = sorted(vertices)
    vidx = {v: i for i, v in enumerate(vlist)}
    
    # Build edges and weights
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    
    def get_weight(a: str, b: str) -> float:
        if not edge_details:
            return 1.0
        return float(len(edge_details.get((a, b), edge_details.get((b, a), []))))
    
    added = set()
    for a in vlist:
        for b in file_graph.get(a, set()):
            if a == b:
                continue
            
            e = (a, b) if a < b else (b, a)
            if e in added:
                continue
            
            added.add(e)
            ia, ib = vidx[e[0]], vidx[e[1]]
            edges.append((ia, ib))
            weights.append(get_weight(e[0], e[1]))
    
    # Create graph and run Leiden
    g = ig.Graph(n=len(vlist), edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights
    
    partition_cls = la.RBConfigurationVertexPartition
    kwargs = {"resolution_parameter": resolution}
    if weights:
        kwargs["weights"] = g.es["weight"]
    if seed is not None:
        kwargs["seed"] = seed
    if iterations is not None:
        kwargs["n_iterations"] = iterations
    
    try:
        part = la.find_partition(g, partition_cls, **kwargs)
    except TypeError:
        part = la.find_partition(g, partition_cls, weights=g.es["weight"] if weights else None)
    
    # Convert back to file names
    comms: List[Set[str]] = []
    for comm in part:
        files = {vlist[i] for i in comm}
        comms.append(files)
    
    comms = sorted(comms, key=lambda s: (-len(s), sorted(list(s))[0] if s else ""))
    return comms

def build_file_graph_from_components(components: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], List[Tuple[str, str]]]]:
    """Build file-level dependency graph from component dependencies."""
    comp_to_file: Dict[str, str] = {}
    for cid, comp in components.items():
        f = getattr(comp, "relative_path", None) or getattr(comp, "file_path", "")
        comp_to_file[cid] = f
    
    file_graph: Dict[str, Set[str]] = defaultdict(set)
    edge_details: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    
    for cid, comp in components.items():
        src_file = comp_to_file.get(cid, "")
        if not src_file:
            continue
        
        for dep in getattr(comp, "depends_on", set()) or []:
            if dep not in components:
                continue
            
            dst_file = comp_to_file.get(dep, "")
            if not dst_file or dst_file == src_file:
                continue
            
            a, b = sorted([src_file, dst_file])
            file_graph[a].add(b)
            file_graph[b].add(a)
            edge_details[(a, b)].append((cid, dep))
    
    # Ensure all files are in the graph
    for f in set(comp_to_file.values()):
        _ = file_graph[f]
    
    return file_graph, edge_details

def _first_line(text: str, limit: int = 200) -> str:
    """Extract first line of text with length limit."""
    if not text:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return (line[:limit]).strip()

def build_graph_from_components(components: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build dependency graph from components."""
    graph = {}
    for comp_id, comp in components.items():
        depends_on = getattr(comp, 'depends_on', set()) or set()
        graph[comp_id] = depends_on
    return graph

def dependency_first_dfs(graph: Dict[str, Set[str]]) -> List[str]:
    """Perform topological sort using DFS."""
    visited = set()
    temp_visited = set()
    result = []
    
    def dfs(node):
        if node in temp_visited:
            return  # Cycle detected, skip
        if node in visited:
            return
        
        temp_visited.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor in graph:  # Only visit nodes that exist
                dfs(neighbor)
        temp_visited.remove(node)
        visited.add(node)
        result.append(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return result