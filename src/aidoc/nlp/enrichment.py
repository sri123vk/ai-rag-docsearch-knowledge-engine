from __future__ import annotations

from aidoc.foundation.model_provider import FoundationModelProvider
from aidoc.models import Chunk, ChunkEnrichment
from aidoc.nlp.entity_extractor import EntityExtractor
from aidoc.nlp.relation_extractor import RelationExtractor


class NLPEnrichmentPipeline:
    def __init__(self, model_provider: FoundationModelProvider) -> None:
        self.model_provider = model_provider
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

    def enrich(self, chunks: list[Chunk]) -> list[ChunkEnrichment]:
        enrichments: list[ChunkEnrichment] = []
        for chunk in chunks:
            entities = self.entity_extractor.extract(chunk.text)
            enrichments.append(
                ChunkEnrichment(
                    chunk=chunk,
                    entities=entities,
                    keyphrases=self.model_provider.extract_keyphrases(chunk.text),
                    document_label=self.model_provider.classify_document(chunk.text),
                    risk_label=self.model_provider.classify_risk(chunk.text),
                    relations=self.relation_extractor.extract(chunk.text, entities),
                    summary=self.model_provider.summarize(chunk.text),
                )
            )
        return enrichments

