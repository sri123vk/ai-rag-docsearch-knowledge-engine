from __future__ import annotations

import re
from collections import defaultdict

from aidoc.models import Chunk, ChunkEnrichment, EntityTriple


ENTITY_PATTERN = re.compile(
    r"\b(?:control owner|auditor|reviewer|exception|remediation ticket|policy|system record|"
    r"executive acknowledgement|vendor|contract|PII|AI system|model|risk|classification)\b",
    re.IGNORECASE,
)


class KnowledgeGraphExtractor:
    def extract(self, chunks: list[Chunk]) -> list[EntityTriple]:
        triples: list[EntityTriple] = []
        for chunk in chunks:
            entities = self._entities(chunk.text)
            for entity in entities:
                triples.append(
                    EntityTriple(
                        subject=chunk.title,
                        relation="mentions",
                        object=entity,
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                    )
                )
            if "remediation" in chunk.text.lower() and "exception" in chunk.text.lower():
                triples.append(
                    EntityTriple(
                        subject="exception",
                        relation="requires",
                        object="remediation",
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                    )
                )
        return triples

    def adjacency(self, triples: list[EntityTriple]) -> dict[str, list[EntityTriple]]:
        graph: dict[str, list[EntityTriple]] = defaultdict(list)
        for triple in triples:
            graph[triple.subject].append(triple)
        return dict(graph)

    def _entities(self, text: str) -> list[str]:
        found = {match.group(0).lower() for match in ENTITY_PATTERN.finditer(text)}
        return sorted(found)


class EnrichedKnowledgeGraphExtractor:
    def extract(self, enrichments: list[ChunkEnrichment]) -> list[EntityTriple]:
        triples: list[EntityTriple] = []
        for enrichment in enrichments:
            chunk = enrichment.chunk
            for entity in enrichment.entities:
                triples.append(
                    EntityTriple(
                        subject=chunk.title,
                        relation="mentions",
                        object=f"{entity.label}:{entity.text}",
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                    )
                )
            for relation in enrichment.relations:
                triples.append(
                    EntityTriple(
                        subject=relation.subject,
                        relation=relation.relation,
                        object=relation.object,
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                    )
                )
            triples.append(
                EntityTriple(
                    subject=chunk.title,
                    relation="classified_as",
                    object=enrichment.document_label.label,
                    chunk_id=chunk.chunk_id,
                    source_uri=chunk.source_uri,
                )
            )
            triples.append(
                EntityTriple(
                    subject=chunk.title,
                    relation="has_risk",
                    object=enrichment.risk_label.label,
                    chunk_id=chunk.chunk_id,
                    source_uri=chunk.source_uri,
                )
            )
        return triples
