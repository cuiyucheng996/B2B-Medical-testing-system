"""标准名向量库：对齐老师 ChromaRepository（口语 embedding 检索入口节点）。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from configs.paths import B2B_ROOT, ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["ChromaAlignRepository", "get_chroma_align_repository"]

CHROMA_DIR = B2B_ROOT / "data" / "chroma" / "entity_align"


class ProjectEmbeddingFunction(EmbeddingFunction):
    """用主项目 get_embeddings()，与老师 SentenceTransformer + normalize 同类。"""

    def __call__(self, input: Documents) -> Embeddings:
        ensure_uvgraph_root()
        from src.embeddings import get_embeddings

        texts = list(input)
        if not texts:
            return []
        return get_embeddings().embed_documents(texts)


class ChromaAlignRepository:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedding_function = ProjectEmbeddingFunction()

    def get_collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_standard_word(self, collection_name: str, documents: list[str], ids: list[Any]) -> None:
        if not documents or not ids or len(documents) != len(ids):
            return
        collection = self.get_collection(collection_name)
        collection.upsert(
            documents=list(documents),
            ids=[str(item) for item in ids],
        )

    def ensure_standard_words(
        self,
        collection_name: str,
        documents: list[str],
        ids: list[Any],
    ) -> None:
        if not documents:
            return
        collection = self.get_collection(collection_name)
        if collection.count() == len(documents):
            return
        logger.info("重建实体向量库 %s: %s 条", collection_name, len(documents))
        self.add_standard_word(collection_name, documents, ids)

    def get_standard_word_by_synonym(self, entity_type: str, synonym: str) -> tuple[str | None, float | None, str | None]:
        """返回 (标准名, distance, id)。distance 越小越近（cosine）。"""
        text = (synonym or "").strip()
        if not text:
            return None, None, None
        collection = self.get_collection(entity_type)
        if collection.count() == 0:
            return None, None, None
        result = collection.query(query_texts=[text], n_results=1)
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        if not documents:
            return None, None, None
        distance = float(distances[0]) if distances else None
        node_id = str(ids[0]) if ids else None
        return str(documents[0]), distance, node_id


@lru_cache(maxsize=1)
def get_chroma_align_repository() -> ChromaAlignRepository:
    return ChromaAlignRepository()
