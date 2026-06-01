from __future__ import annotations

import re

from aidoc.models import Chunk, RawDocument


class SemanticChunker:
    def __init__(self, max_words: int = 140, overlap_words: int = 25) -> None:
        if max_words <= overlap_words:
            raise ValueError("max_words must be larger than overlap_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, document: RawDocument) -> list[Chunk]:
        sections = self._split_sections(document.text)
        chunks: list[Chunk] = []
        chunk_number = 0

        for page, section in sections:
            words = section.split()
            start = 0
            while start < len(words):
                end = min(start + self.max_words, len(words))
                text = " ".join(words[start:end]).strip()
                if text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}:chunk:{chunk_number}",
                            doc_id=document.doc_id,
                            title=document.title,
                            text=text,
                            source_uri=document.source_uri,
                            page=page,
                            metadata=document.metadata,
                        )
                    )
                    chunk_number += 1
                if end == len(words):
                    break
                start = max(0, end - self.overlap_words)

        return chunks

    def _split_sections(self, text: str) -> list[tuple[int | None, str]]:
        parts = re.split(r"(?=Page\s+\d+\s+of\s+\d+)", text)
        sections: list[tuple[int | None, str]] = []
        for part in parts:
            clean = part.strip()
            if not clean:
                continue
            match = re.search(r"Page\s+(\d+)\s+of\s+\d+", clean)
            page = int(match.group(1)) if match else None
            sections.append((page, clean))
        return sections or [(None, text)]

