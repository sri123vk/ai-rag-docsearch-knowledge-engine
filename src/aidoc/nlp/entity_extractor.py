from __future__ import annotations

from aidoc.models import Entity
from aidoc.nlp.patterns import ENTITY_PATTERNS


class EntityExtractor:
    def extract(self, text: str) -> list[Entity]:
        lowered = text.lower()
        entities: list[Entity] = []
        for label, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in lowered:
                    entities.append(Entity(text=pattern, label=label, confidence=0.82))
        return sorted(entities, key=lambda entity: (entity.label, entity.text))

