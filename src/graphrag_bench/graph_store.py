"""Knowledge Graph Storage and Querying Engine wrapping NetworkX MultiDiGraph.

Provides:
- In-memory NetworkX MultiDiGraph data structure.
- Node management with rich GraphEntity schemas.
- Edge management with GraphRelationship triples and evidence provenance.
- Graph analytics and statistics (density, components, degree, predicate distribution).
- Ego-graph / neighborhood subgraph extraction.
- Document-level deletion with orphan node purging.
- Atomic persistence to canonical JSON (`knowledge_graph.json`).
- Atomic export to sanitized GraphML (`knowledge_graph.graphml`).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship

logger = logging.getLogger(__name__)


class GraphStore:
    """Persistent Knowledge Graph store wrapping NetworkX MultiDiGraph."""

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entities: dict[str, GraphEntity] = {}
        self._relationships: dict[str, GraphRelationship] = {}

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def add_entity(self, entity: GraphEntity) -> None:
        """Add or update an entity node in the graph."""
        self._entities[entity.entity_id] = entity
        self._graph.add_node(
            entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            description=entity.description,
            mention_count=entity.mention_count,
            aliases=list(entity.aliases),
            source_document_ids=list(entity.source_document_ids),
            source_chunk_ids=list(entity.source_chunk_ids),
            metadata=dict(entity.metadata),
        )

    def add_relationship(self, rel: GraphRelationship) -> None:
        """Add a directed relationship edge between subject and object entities.

        If either the subject or object node is not yet present, a minimal placeholder
        node is created to guarantee graph referential integrity.
        """
        # Ensure subject node exists
        if rel.subject_entity_id not in self._graph:
            placeholder_sub = GraphEntity(
                entity_id=rel.subject_entity_id,
                canonical_name=rel.subject_entity_id,
                entity_type="OTHER",
                description="Implicit entity reference",
                mention_count=1,
                source_document_ids=[rel.source_document_id] if rel.source_document_id else [],
                source_chunk_ids=[rel.source_chunk_id] if rel.source_chunk_id else [],
            )
            self.add_entity(placeholder_sub)

        # Ensure object node exists
        if rel.object_entity_id not in self._graph:
            placeholder_obj = GraphEntity(
                entity_id=rel.object_entity_id,
                canonical_name=rel.object_entity_id,
                entity_type="OTHER",
                description="Implicit entity reference",
                mention_count=1,
                source_document_ids=[rel.source_document_id] if rel.source_document_id else [],
                source_chunk_ids=[rel.source_chunk_id] if rel.source_chunk_id else [],
            )
            self.add_entity(placeholder_obj)

        self._relationships[rel.relationship_id] = rel
        self._graph.add_edge(
            rel.subject_entity_id,
            rel.object_entity_id,
            key=rel.relationship_id,
            relationship_id=rel.relationship_id,
            predicate=rel.predicate,
            evidence_quote=rel.evidence_quote,
            confidence=float(rel.confidence),
            source_document_id=rel.source_document_id,
            source_chunk_id=rel.source_chunk_id,
            source_filename=rel.source_filename,
            extraction_model=rel.extraction_model,
        )

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Retrieve entity by entity ID."""
        return self._entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> GraphRelationship | None:
        """Retrieve relationship by relationship ID."""
        return self._relationships.get(relationship_id)

    def get_entities(self) -> list[GraphEntity]:
        """Return all entities in the graph."""
        return list(self._entities.values())

    def get_relationships(self) -> list[GraphRelationship]:
        """Return all relationships in the graph."""
        return list(self._relationships.values())

    def get_neighbors(
        self,
        entity_id: str,
        direction: Literal["out", "in", "both"] = "both",
    ) -> list[dict[str, Any]]:
        """Retrieve neighboring entities connected to entity_id along with edge details.

        Returns:
            List of dicts with keys:
            - 'neighbor': GraphEntity
            - 'relationship': GraphRelationship
            - 'direction': 'outgoing' | 'incoming'
        """
        if entity_id not in self._graph:
            return []

        neighbors: list[dict[str, Any]] = []

        if direction in ("out", "both"):
            for _, target_id, key, data in self._graph.out_edges(entity_id, keys=True, data=True):
                target_entity = self._entities.get(target_id)
                rel = self._relationships.get(key)
                if target_entity and rel:
                    neighbors.append(
                        {
                            "neighbor": target_entity,
                            "relationship": rel,
                            "direction": "outgoing",
                        }
                    )

        if direction in ("in", "both"):
            for source_id, _, key, data in self._graph.in_edges(entity_id, keys=True, data=True):
                source_entity = self._entities.get(source_id)
                rel = self._relationships.get(key)
                if source_entity and rel:
                    neighbors.append(
                        {
                            "neighbor": source_entity,
                            "relationship": rel,
                            "direction": "incoming",
                        }
                    )

        return neighbors

    def get_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 1,
    ) -> nx.MultiDiGraph:
        """Extract an induced ego-subgraph surrounding entity_ids up to max_hops."""
        nodes_to_include: set[str] = set()

        for eid in entity_ids:
            if eid in self._graph:
                nodes_to_include.add(eid)
                current_frontier = {eid}
                for _ in range(max_hops):
                    next_frontier: set[str] = set()
                    for node in current_frontier:
                        next_frontier.update(self._graph.successors(node))
                        next_frontier.update(self._graph.predecessors(node))
                    nodes_to_include.update(next_frontier)
                    current_frontier = next_frontier

        return self._graph.subgraph(nodes_to_include).copy()

    def remove_document(self, doc_id: str) -> tuple[int, int]:
        """Remove all relationships and provenance associated with a document.

        Purges orphan nodes (degree 0 and 0 remaining source documents).

        Returns:
            Tuple of (edges_removed_count, nodes_removed_count).
        """
        # 1. Identify relationships belonging to this document
        edges_to_remove = [
            (u, v, k)
            for u, v, k, data in self._graph.edges(keys=True, data=True)
            if data.get("source_document_id") == doc_id
        ]

        edges_removed_count = len(edges_to_remove)
        for u, v, k in edges_to_remove:
            self._graph.remove_edge(u, v, key=k)
            self._relationships.pop(k, None)

        # 2. Update entities' document provenance
        nodes_to_check: list[str] = list(self._entities.keys())
        nodes_removed_count = 0

        for nid in nodes_to_check:
            entity = self._entities.get(nid)
            if not entity:
                continue

            if doc_id in entity.source_document_ids:
                entity.source_document_ids.remove(doc_id)
                # Sync node attribute
                if nid in self._graph:
                    self._graph.nodes[nid]["source_document_ids"] = list(entity.source_document_ids)

            # Check if node has become orphan: degree 0 and no source documents left
            in_deg = self._graph.in_degree(nid) if nid in self._graph else 0
            out_deg = self._graph.out_degree(nid) if nid in self._graph else 0
            total_deg = in_deg + out_deg

            if total_deg == 0 and not entity.source_document_ids:
                if nid in self._graph:
                    self._graph.remove_node(nid)
                self._entities.pop(nid, None)
                nodes_removed_count += 1

        logger.info(
            "Removed document %s from graph: %d edges removed, %d orphan nodes purged",
            doc_id,
            edges_removed_count,
            nodes_removed_count,
        )
        return edges_removed_count, nodes_removed_count

    def get_statistics(self) -> dict[str, Any]:
        """Compute comprehensive graph topology and metadata statistics."""
        num_nodes = self._graph.number_of_nodes()
        num_edges = self._graph.number_of_edges()

        # Density: for MultiDiGraph, standard directed density n*(n-1)
        density = nx.density(self._graph) if num_nodes > 1 else 0.0

        # Connected components (weakly connected for directed graphs)
        components_count = (
            nx.number_weakly_connected_components(self._graph) if num_nodes > 0 else 0
        )

        # Entity type distribution
        type_dist: dict[str, int] = {}
        for ent in self._entities.values():
            type_dist[ent.entity_type] = type_dist.get(ent.entity_type, 0) + 1

        # Predicate distribution
        pred_dist: dict[str, int] = {}
        for rel in self._relationships.values():
            pred_dist[rel.predicate] = pred_dist.get(rel.predicate, 0) + 1

        # Degree distribution / Top degree entities
        top_entities: list[dict[str, Any]] = []
        for nid in self._graph.nodes():
            in_d = self._graph.in_degree(nid)
            out_d = self._graph.out_degree(nid)
            ent = self._entities.get(nid)
            top_entities.append(
                {
                    "entity_id": nid,
                    "canonical_name": ent.canonical_name if ent else nid,
                    "entity_type": ent.entity_type if ent else "OTHER",
                    "total_degree": in_d + out_d,
                    "in_degree": in_d,
                    "out_degree": out_d,
                }
            )
        top_entities.sort(key=lambda x: x["total_degree"], reverse=True)

        # Unique document and chunk IDs covered
        doc_ids: set[str] = set()
        chunk_ids: set[str] = set()
        for ent in self._entities.values():
            doc_ids.update(ent.source_document_ids)
            chunk_ids.update(ent.source_chunk_ids)

        return {
            "node_count": num_nodes,
            "edge_count": num_edges,
            "density": float(density),
            "weakly_connected_components": components_count,
            "is_weakly_connected": components_count == 1 if num_nodes > 0 else False,
            "entity_type_distribution": type_dist,
            "predicate_distribution": pred_dist,
            "top_entities_by_degree": top_entities[:10],
            "total_documents_represented": len(doc_ids),
            "total_chunks_represented": len(chunk_ids),
        }

    def save_json(self, path: Path | str) -> None:
        """Atomically save knowledge graph to canonical JSON format."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(".tmp")

        payload = {
            "schema_version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entities": [ent.model_dump() for ent in self._entities.values()],
            "relationships": [rel.model_dump() for rel in self._relationships.values()],
        }

        with open(temp_target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        temp_target.replace(target)
        logger.info(
            "Saved %d entities and %d relationships to %s", self.node_count, self.edge_count, target
        )

    def load_json(self, path: Path | str) -> None:
        """Load knowledge graph from canonical JSON file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Graph file does not exist: {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.reset()
        for raw_ent in data.get("entities", []):
            ent = GraphEntity.model_validate(raw_ent)
            self.add_entity(ent)

        for raw_rel in data.get("relationships", []):
            rel = GraphRelationship.model_validate(raw_rel)
            self.add_relationship(rel)

        logger.info(
            "Loaded graph from %s: %d entities, %d relationships",
            target,
            self.node_count,
            self.edge_count,
        )

    def export_graphml(self, path: Path | str) -> None:
        """Export graph to GraphML format with sanitized primitive attributes.

        GraphML requires all node and edge attribute values to be primitive types
        (str, int, float, bool). Nested lists and dictionaries are JSON-serialized strings.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(".tmp")

        # Create sanitized export copy
        export_graph = nx.MultiDiGraph()

        # Sanitize nodes
        for node_id, data in self._graph.nodes(data=True):
            clean_attrs: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_attrs[k] = v
                elif isinstance(v, (list, dict)):
                    clean_attrs[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    clean_attrs[k] = ""
                else:
                    clean_attrs[k] = str(v)
            export_graph.add_node(node_id, **clean_attrs)

        # Sanitize edges
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            clean_attrs = {}
            for k, val in data.items():
                if isinstance(val, (str, int, float, bool)):
                    clean_attrs[k] = val
                elif isinstance(val, (list, dict)):
                    clean_attrs[k] = json.dumps(val, ensure_ascii=False)
                elif val is None:
                    clean_attrs[k] = ""
                else:
                    clean_attrs[k] = str(val)
            export_graph.add_edge(u, v, key=str(key), **clean_attrs)

        nx.write_graphml(export_graph, temp_target)
        temp_target.replace(target)
        logger.info("Exported sanitized GraphML to %s", target)

    def to_networkx(self) -> nx.MultiDiGraph:
        """Return a copy of the underlying NetworkX MultiDiGraph."""
        return self._graph.copy()

    def reset(self) -> None:
        """Clear all nodes, edges, and internal lookup state."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()
