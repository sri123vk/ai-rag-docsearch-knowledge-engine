# System Design

## Phase Boundary

This repository is the AI intelligence layer. It assumes raw text is available from a traditional ingestion system and focuses on chunk level retrieval, knowledge extraction, and citation grounded responses.

## Components

| Component | Responsibility |
| :-- | :-- |
| Loader | Reads source documents |
| Chunker | Preserves page level context and creates retrievable units |
| Embedding model | Converts chunk text into vectors |
| Hybrid retriever | Combines lexical and semantic evidence |
| Knowledge graph extractor | Converts unstructured text into relationship triples |
| Answer builder | Produces responses with citations |

## Production Upgrade Path

1. Replace local text loading with S3 or Lucene metadata integration
2. Replace hash embeddings with transformer embeddings
3. Add a persistent vector index
4. Add reranking
5. Add LLM answer generation
6. Store knowledge graph triples in Neo4j or Amazon Neptune
7. Add evaluation metrics for retrieval quality and citation faithfulness

## GPU Plan

GPU workers should be isolated behind model interfaces:

```text
chunk batch -> GPU embedding worker -> vector store
retrieved chunks -> GPU reranker -> final evidence
document pages -> GPU OCR worker -> extracted text
```

This lets the CPU based indexing and metadata layer scale independently from model inference workloads.

