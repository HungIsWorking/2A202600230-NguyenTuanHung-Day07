from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # Initialize chromadb client + collection
            self._client = chromadb.EphemeralClient()  # In-memory ephemeral client
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        return {
            'id': str(self._next_index),
            'doc_id': doc.id,
            'content': doc.content,
            'embedding': embedding,
            'metadata': doc.metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Run in-memory similarity search over provided records
        query_embedding = self._embedding_fn(query)
        
        # Compute similarity score for each record
        scored_records = []
        for record in records:
            similarity = _dot(query_embedding, record['embedding'])
            scored_record = dict(record)
            scored_record['score'] = similarity
            scored_records.append(scored_record)
        
        # Sort by score (descending) and return top_k
        scored_records.sort(key=lambda r: r['score'], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            self._next_index += 1
            
            if self._use_chroma and self._collection is not None:
                # Add to ChromaDB
                self._collection.add(
                    ids=[record['id']],
                    documents=[record['content']],
                    embeddings=[record['embedding']],
                    metadatas=[{**record['metadata'], 'doc_id': record['doc_id']}],
                )
            else:
                # Add to in-memory store
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            # Use ChromaDB search
            query_embedding = self._embedding_fn(query)
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                )
                # Convert ChromaDB results to our format
                output = []
                if results and results['documents'] and len(results['documents']) > 0:
                    for i, doc_content in enumerate(results['documents'][0]):
                        output.append({
                            'content': doc_content,
                            'score': results.get('distances', [[]])[0][i] if results.get('distances') else 0.5,
                            'metadata': results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {},
                        })
                return output
            except Exception:
                # Fall back to in-memory search
                pass
        
        # In-memory search
        if not self._store:
            return []
        
        scored_results = self._search_records(query, self._store, top_k)
        return [{'content': r['content'], 'score': r['score'], 'metadata': r['metadata']} for r in scored_results]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            # No filter, just return regular search
            return self.search(query, top_k)
        
        if self._use_chroma and self._collection is not None:
            # Use ChromaDB with where filter
            query_embedding = self._embedding_fn(query)
            try:
                # Build where clause from metadata_filter
                where_clause = {}
                for key, value in metadata_filter.items():
                    where_clause[key] = {"$eq": value}
                
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    where=where_clause if where_clause else None,
                    n_results=top_k,
                )
                # Convert ChromaDB results to our format
                output = []
                if results and results['documents'] and len(results['documents']) > 0:
                    for i, doc_content in enumerate(results['documents'][0]):
                        output.append({
                            'content': doc_content,
                            'score': results.get('distances', [[]])[0][i] if results.get('distances') else 0.5,
                            'metadata': results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {},
                        })
                return output
            except Exception:
                # Fall back to in-memory filtering
                pass
        
        # In-memory filtering then search
        if not self._store:
            return []
        
        # Filter records by metadata
        filtered_records = []
        for record in self._store:
            match = True
            for key, value in metadata_filter.items():
                if record['metadata'].get(key) != value:
                    match = False
                    break
            if match:
                filtered_records.append(record)
        
        # Search among filtered records
        if not filtered_records:
            return []
        
        scored_results = self._search_records(query, filtered_records, top_k)
        return [{'content': r['content'], 'score': r['score'], 'metadata': r['metadata']} for r in scored_results]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                # Get all records with this doc_id first
                results = self._collection.get(
                    where={"doc_id": {"$eq": doc_id}}
                )
                if results and results['ids']:
                    self._collection.delete(ids=results['ids'])
                    return len(results['ids']) > 0
                return False
            except Exception:
                # Fall back to in-memory deletion
                pass
        
        # In-memory deletion
        original_size = len(self._store)
        self._store = [r for r in self._store if r['doc_id'] != doc_id]
        return len(self._store) < original_size
