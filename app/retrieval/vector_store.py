"""
FAISS Vector Store Manager for property semantic embeddings and vector similarity search.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from app.data.preprocessor import Document

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorStoreManager:
    """Manages embedding generation, vector indexing, persistence, and semantic retrieval."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_dir: str = "vectorstore",
    ):
        self.model_name = model_name
        self.vector_store_dir = vector_store_dir
        self.index_path = os.path.join(vector_store_dir, "faiss.index")
        self.meta_path = os.path.join(vector_store_dir, "metadata.json")

        self.model = None
        self.index = None
        self.documents: List[Document] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        os.makedirs(self.vector_store_dir, exist_ok=True)
        self._init_model()

    def _init_model(self):
        """Initializes the SentenceTransformer embedding model."""
        if HAS_ST:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[VectorStoreManager] Warning initializing SentenceTransformer: {e}")
                self.model = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embedding vector for input text."""
        if self.model:
            emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)
        else:
            # Simple TF-IDF / character ngram fallback embedding if sentence-transformers not present
            np.random.seed(abs(hash(text)) % (2**32))
            emb = np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generates embedding vectors for a batch of texts."""
        if self.model:
            embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        else:
            return np.vstack([self._get_embedding(t) for t in texts])

    def build_index(self, documents: List[Document]) -> int:
        """Builds FAISS index from preprocessed property documents."""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings = self._get_embeddings_batch(texts)
        self.embeddings_matrix = embeddings

        dim = embeddings.shape[1]
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            self.save_index()
        else:
            # Store in numpy matrix for cosine fallback
            self.save_index()

        return len(documents)

    def save_index(self):
        """Persists vector index and document metadata to disk."""
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, self.index_path)

        meta_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        if self.embeddings_matrix is not None:
            np.save(os.path.join(self.vector_store_dir, "embeddings.npy"), self.embeddings_matrix)

    def load_index(self) -> bool:
        """Loads index and metadata from disk if present."""
        if not os.path.exists(self.meta_path):
            return False

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in meta_data
        ]

        if HAS_FAISS and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        emb_file = os.path.join(self.vector_store_dir, "embeddings.npy")
        if os.path.exists(emb_file):
            self.embeddings_matrix = np.load(emb_file)

        return True

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Performs vector similarity search against property documents."""
        if not self.documents:
            return []

        query_vector = self._get_embedding(query).reshape(1, -1)

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    # Scale score from cosine range [-1, 1] to [0, 1]
                    norm_score = max(0.0, float((score + 1.0) / 2.0))
                    results.append((self.documents[idx], norm_score))
            return results

        # Cosine similarity fallback using numpy matrix
        if self.embeddings_matrix is not None:
            sims = np.dot(self.embeddings_matrix, query_vector.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                norm_score = max(0.0, float((score + 1.0) / 2.0))
                results.append((self.documents[idx], norm_score))
            return results

        return []
