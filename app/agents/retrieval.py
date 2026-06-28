import logging
from typing import Dict, Any, List
logger = logging.getLogger(__name__)
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
from ..utils.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL
class RetrievalAgent:
    def __init__(self):
        if DEPS_AVAILABLE:
            from pathlib import Path
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Retrieval agent initialized ({self.collection.count()} docs)")
        else:
            self.client = None
            self.collection = None
            self.embedder = None
            self._fallback_docs = []
            logger.warning("ChromaDB/SentenceTransformers not installed - using fallback")
    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        logger.info(f"Retrieval: searching for '{query[:60]}...'")
        if self.collection and self.collection.count() > 0:
            query_emb = self.embedder.encode([query]).tolist()
            # Fetch more documents initially (e.g. 20) for reranking
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=min(20, self.collection.count())
            )
            chunks = []
            for i in range(len(results["documents"][0])):
                chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0
                })
            
            try:
                from FlagEmbedding import FlagReranker
                reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
                pairs = [[query, c["text"]] for c in chunks]
                scores = reranker.compute_score(pairs)
                for c, s in zip(chunks, scores):
                    c["score"] = s
            except Exception as e:
                logger.warning(f"BGE Reranker failed or not installed, using base scores: {e}")
            
            chunks.sort(key=lambda x: x["score"], reverse=True)
            chunks = chunks[:10] # Return top 10 after reranking
        else:
            chunks = []
            logger.info("No documents in ChromaDB - run ingestion first")
        logger.info(f"Retrieved {len(chunks)} chunks")
        return {
            "retrieved_chunks": chunks,
            "status": f"Retrieved {len(chunks)} chunks"
        }
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None,
                      ids: List[str] = None) -> int:
        if not self.collection or not texts:
            return 0
        if ids is None:
            import hashlib
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return len(texts)
