from __future__ import annotations

from pathlib import Path

from aidoc.models import RawDocument


def load_text_documents(root: Path) -> list[RawDocument]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"{root} is not a readable directory")

    documents: list[RawDocument] = []
    for path in sorted(root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            RawDocument(
                doc_id=str(path.resolve()),
                title=path.stem.replace("-", " ").title(),
                text=text,
                source_uri=path.resolve().as_uri(),
                metadata={"file_name": path.name},
            )
        )
    return documents

