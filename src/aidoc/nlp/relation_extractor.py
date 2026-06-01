from __future__ import annotations

from aidoc.models import Entity, Relation


class RelationExtractor:
    def extract(self, text: str, entities: list[Entity]) -> list[Relation]:
        lowered = text.lower()
        entity_texts = {entity.text for entity in entities}
        relations: list[Relation] = []

        if "exception" in lowered and "remediation" in lowered:
            relations.append(Relation("exception", "requires", "remediation", 0.86))
        if "approval" in lowered and "reviewer" in entity_texts:
            relations.append(Relation("reviewer", "approves", "control result", 0.78))
        if "control owner" in entity_texts and "evidence" in lowered:
            relations.append(Relation("control owner", "maintains", "audit evidence", 0.82))
        if "policy" in entity_texts and "requires" in lowered:
            relations.append(Relation("policy", "requires", "control procedure", 0.72))
        if "ai system" in entity_texts and "human oversight" in entity_texts:
            relations.append(Relation("ai system", "requires", "human oversight", 0.88))
        if "vendor" in entity_texts and "contract" in entity_texts:
            relations.append(Relation("vendor", "governed_by", "contract", 0.8))

        return relations

