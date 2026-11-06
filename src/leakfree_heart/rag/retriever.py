"""Hybrid retriever: TF-IDF (always available, deterministic) fused with dense embeddings + FAISS when
sentence-transformers is installed and the model can be loaded. Fusion = reciprocal rank fusion."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..clinical_text import retrieval_query
from .kb import Chunk

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, chunks: list[Chunk], backend: str = "auto",
                 model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 seed_boost: float = 1.6):
        self.seed_boost = seed_boost
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self._docs = [f"{c.title}. {c.text}" for c in chunks]
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.M = self.tfidf.fit_transform(self._docs)
        self.dense = None
        self.backend = "tfidf"
        if backend in ("auto", "dense"):
            try:
                self._init_dense(model_name, device)
                self.backend = "hybrid(dense+tfidf)"
            except Exception as e:  # noqa: BLE001
                if backend == "dense":
                    raise
                print(f"[retriever] dense backend unavailable ({type(e).__name__}); using TF-IDF only")

    # ------------------------------------------------------------------ dense
    def _init_dense(self, model_name: str, device: str | None):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name, device=device)
        emb = self.model.encode(self._docs, normalize_embeddings=True, show_progress_bar=False)
        self.emb = np.asarray(emb, dtype="float32")
        try:
            import faiss  # type: ignore
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except Exception:  # noqa: BLE001
            self.index = None
        self.dense = True
        self._prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([self._prefix + query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        return (self.emb @ q.T).ravel()

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([query])
        return (self.M @ q.T).toarray().ravel()

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 4, feature: str | None = None) -> list[tuple[Chunk, float]]:
        mask = np.ones(len(self.chunks), bool)
        if feature:
            m = np.array([feature in c.features for c in self.chunks])
            if m.any():
                mask = m
        scorers = [self._tfidf_scores(query)]
        if self.dense:
            scorers.append(self._dense_scores(query))
        fused = np.zeros(len(self.chunks))
        for s in scorers:
            s = np.where(mask, s, -np.inf)
            order = np.argsort(-s)
            for rank, i in enumerate(order[: max(k * 4, 20)]):
                if np.isfinite(s[i]):
                    fused[i] += 1.0 / (60 + rank + 1)
        boost = np.array([self.seed_boost if c.id.startswith("KB-") else 1.0 for c in self.chunks])
        fused = np.where(mask, fused * boost, -np.inf)
        top = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in top if np.isfinite(fused[i]) and fused[i] > 0]

    def retrieve_for_explanation(self, patient: dict, top_features: list[str], per_feature_k: int = 2,
                                 prevention_k: int = 2) -> list[Chunk]:
        """Feature-aware retrieval: for each influential feature query the evidence tagged with that feature,
        then add prevention passages for the recommendations. Order and de-duplication are deterministic."""
        out: list[Chunk] = []
        seen = set()

        def add(chs):
            for c in chs:
                if c.id not in seen:
                    seen.add(c.id)
                    out.append(c)

        for f in top_features:
            q = retrieval_query(f, patient.get(f))
            add([c for c, _ in self.search(q, k=per_feature_k, feature=f)])
        drivers = " ".join(retrieval_query(f, patient.get(f)) for f in top_features[:2])
        add([c for c, _ in self.search("lifestyle prevention " + drivers, k=prevention_k, feature="prevention")])
        add([c for c, _ in self.search("risk estimates support clinical judgement limits of the data", k=1, feature="general")])
        return out
