"""
Graph Traversal Engine.
Performs bounded multi-hop path traversal across NetworkX MultiDiGraph structures.
Supports outgoing, incoming, and bidirectional exploration with cycle prevention,
predicate filtering, and maximum degree bounds.
"""

from collections import deque
from typing import Any, Literal

from graphrag_bench.config import get_settings
from graphrag_bench.graph_retrieval_schemas import GraphPath, PathEdge, PathNode
from graphrag_bench.graph_schemas import GraphRelationship
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import get_logger

logger = get_logger("graph_traversal")


class GraphTraversalEngine:
    """Bounded, direction-aware multi-hop graph path explorer."""

    def __init__(self, store: GraphStore) -> None:
        """Initialize traversal engine with backed GraphStore.

        Args:
            store: Active GraphStore instance.
        """
        self.store = store
        self.settings = get_settings()

    def get_neighbors(
        self,
        entity_id: str,
        direction: Literal["both", "out", "in"] = "both",
        predicates: list[str] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve 1-hop neighbors with edge keys and directional context."""
        if entity_id not in self.store._graph:
            return []

        neighbors: list[dict[str, Any]] = []
        pred_set = {p.upper() for p in predicates} if predicates else None

        # Outgoing edges
        if direction in ("out", "both"):
            for _, target_id, key, data in self.store._graph.out_edges(
                entity_id, keys=True, data=True
            ):
                pred = data.get("predicate", "").upper()
                if pred_set and pred not in pred_set:
                    continue
                target_entity = self.store.get_entity(target_id)
                rel = self.store.get_relationship(key)
                if target_entity and rel:
                    neighbors.append(
                        {
                            "neighbor": target_entity,
                            "relationship": rel,
                            "direction": "outgoing",
                            "hop": 1,
                        }
                    )

        # Incoming edges
        if direction in ("in", "both"):
            for source_id, _, key, data in self.store._graph.in_edges(
                entity_id, keys=True, data=True
            ):
                pred = data.get("predicate", "").upper()
                if pred_set and pred not in pred_set:
                    continue
                source_entity = self.store.get_entity(source_id)
                rel = self.store.get_relationship(key)
                if source_entity and rel:
                    neighbors.append(
                        {
                            "neighbor": source_entity,
                            "relationship": rel,
                            "direction": "incoming",
                            "hop": 1,
                        }
                    )

        return neighbors

    def traverse_from_seed(
        self,
        seed_entity_id: str,
        max_hops: int = 2,
        direction: Literal["both", "out", "in"] = "both",
        predicates: list[str] | set[str] | None = None,
        doc_id_filter: str | None = None,
        max_paths: int = 50,
        max_degree: int = 50,
    ) -> list[GraphPath]:
        """Perform bounded breadth-first traversal up to max_hops from seed entity.

        Args:
            seed_entity_id: Knowledge graph identifier of the starting node.
            max_hops: Traversal depth limit (bounded between 1 and 3).
            direction: Traversal directionality ('out', 'in', 'both').
            predicates: Optional allowed relationship predicates.
            doc_id_filter: Optional source document filter.
            max_paths: Maximum number of distinct paths to extract.
            max_degree: Degree cutoff preventing combinatorial explosion at super-nodes.

        Returns:
            List of cycle-free GraphPath instances with node and edge sequences.
        """
        # Validate seed existence
        seed_entity = self.store.get_entity(seed_entity_id)
        if not seed_entity or seed_entity_id not in self.store._graph:
            logger.debug("Seed entity '%s' not present in graph.", seed_entity_id)
            return []

        # Bound max_hops safely between 1 and 3
        effective_hops = max(1, min(3, max_hops))
        pred_set = {p.upper() for p in predicates} if predicates else None

        seed_node = PathNode(
            entity_id=seed_entity.entity_id,
            canonical_name=seed_entity.canonical_name,
            entity_type=seed_entity.entity_type,
            hop_index=0,
        )

        discovered_paths: list[GraphPath] = []
        # Queue elements: (current_node_id, [PathNode], [PathEdge], set_of_visited_node_ids)
        queue: deque[tuple[str, list[PathNode], list[PathEdge], set[str]]] = deque()
        queue.append((seed_entity_id, [seed_node], [], {seed_entity_id}))

        path_idx = 1
        while queue and len(discovered_paths) < max_paths:
            curr_id, node_seq, edge_seq, visited_ids = queue.popleft()
            current_hop = len(edge_seq)

            # Check hub degree limit on current node
            in_deg = self.store._graph.in_degree(curr_id)
            out_deg = self.store._graph.out_degree(curr_id)
            total_deg = in_deg + out_deg
            if total_deg > max_degree:
                logger.debug(
                    "Skipping expansion on super-node '%s' (degree %d > threshold %d)",
                    curr_id,
                    total_deg,
                    max_degree,
                )
                continue

            # Gather candidate outgoing steps
            candidate_steps: list[tuple[str, GraphRelationship, str]] = []
            if direction in ("out", "both"):
                for _, target_id, key, data in self.store._graph.out_edges(
                    curr_id, keys=True, data=True
                ):
                    pred = data.get("predicate", "").upper()
                    if pred_set and pred not in pred_set:
                        continue
                    if target_id in visited_ids:
                        continue  # Cycle prevention: node already in this specific path
                    rel = self.store.get_relationship(key)
                    if rel:
                        if doc_id_filter and rel.source_document_id != doc_id_filter:
                            continue
                        candidate_steps.append((target_id, rel, "outgoing"))

            # Gather candidate incoming steps
            if direction in ("in", "both"):
                for source_id, _, key, data in self.store._graph.in_edges(
                    curr_id, keys=True, data=True
                ):
                    pred = data.get("predicate", "").upper()
                    if pred_set and pred not in pred_set:
                        continue
                    if source_id in visited_ids:
                        continue  # Cycle prevention: node already in this specific path
                    rel = self.store.get_relationship(key)
                    if rel:
                        if doc_id_filter and rel.source_document_id != doc_id_filter:
                            continue
                        candidate_steps.append((source_id, rel, "incoming"))

            for next_node_id, rel, step_dir in candidate_steps:
                next_entity = self.store.get_entity(next_node_id)
                if not next_entity:
                    continue

                new_node = PathNode(
                    entity_id=next_entity.entity_id,
                    canonical_name=next_entity.canonical_name,
                    entity_type=next_entity.entity_type,
                    hop_index=current_hop + 1,
                )
                new_edge = PathEdge(
                    relationship_id=rel.relationship_id,
                    subject_entity_id=rel.subject_entity_id,
                    predicate=rel.predicate,
                    object_entity_id=rel.object_entity_id,
                    direction=step_dir,
                    evidence_quote=rel.evidence_quote,
                    confidence=rel.confidence,
                    source_document_id=rel.source_document_id,
                    source_chunk_id=rel.source_chunk_id,
                    source_filename=rel.source_filename,
                )

                new_node_seq = list(node_seq) + [new_node]
                new_edge_seq = list(edge_seq) + [new_edge]
                new_visited = set(visited_ids) | {next_node_id}

                # Compile document and chunk provenance
                doc_ids = list(dict.fromkeys(e.source_document_id for e in new_edge_seq))
                chunk_ids = list(dict.fromkeys(e.source_chunk_id for e in new_edge_seq))

                path_obj = GraphPath(
                    path_id=f"path_{path_idx:04d}",
                    seed_entity_id=seed_entity.entity_id,
                    seed_entity_name=seed_entity.canonical_name,
                    nodes=new_node_seq,
                    edges=new_edge_seq,
                    hop_count=len(new_edge_seq),
                    source_document_ids=doc_ids,
                    source_chunk_ids=chunk_ids,
                )
                discovered_paths.append(path_obj)
                path_idx += 1

                if len(discovered_paths) >= max_paths:
                    break

                # Continue BFS queueing if below maximum hop depth
                if current_hop + 1 < effective_hops:
                    queue.append((next_node_id, new_node_seq, new_edge_seq, new_visited))

        return discovered_paths

    # Alias for convenience
    traverse = traverse_from_seed
