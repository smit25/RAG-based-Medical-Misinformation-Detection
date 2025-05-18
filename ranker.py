# langchain_hybrid_ranker.py
from __future__ import annotations
from typing import List, Tuple, Dict, Sequence
from collections import defaultdict

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Ranker:
    """
    Hybrid (vector + lexical) retriever with optional CE or IRC re‑ranking.
    Mirrors the logic of your original Ranker, but built on LangChain components.
    """

    def __init__(self, faiss_store: FAISS, bm25_retriever: BM25Retriever, inverter, rank_type: str = "irc", ):
        self.inverter = inverter
        self.rank_type = rank_type.lower()

        self.faiss_retriever = faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def _docs_to_pairs(docs: Sequence[Document]) -> List[Tuple[str, float]]:
        """
        Convert LangChain `Document` objects to `(id, score)` pairs.
        Assumes each document was stored with `metadata={'id': ...}`.
        """
        return [(d.metadata["id"], float(getattr(d, "score", 0.0))) for d in docs]


    @staticmethod
    def _normalize(ranking: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not ranking:
            return ranking
        scores = [s for _, s in ranking]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return ranking
        return [(d, (s - lo) / (hi - lo)) for d, s in ranking]
    

    def faiss_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        docs = self.faiss_retriever.get_relevant_documents(query, k=k)
        return self._docs_to_pairs(docs)


    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        docs = self.bm25_retriever.get_relevant_documents(query, k=k)
        return self._docs_to_pairs(docs)


    def reciprocal_rank_fusion(lists: List[List[Tuple[str, float]]], k: int = 40) -> List[Tuple[str, float]]:
        scores = defaultdict(float)
        for result in lists:
            for rank, (doc_id, _) in enumerate(result):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


    def hybrid_search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        faiss = self.faiss_search(query, k)
        bm25 = self.bm25_search(query, k)
        return self.reciprocal_rank_fusion([bm25, faiss], k=k)


    def _inverse_consistency(self, orig: List[Tuple[str, float]], inv: List[Tuple[str, float]], lam: float = 0.5) -> List[Tuple[str, float]]:
        orig = self._normalize(orig)
        inv = self._normalize(inv)
        norm = max(len(orig), len(inv))

        o_rank = {d: r for r, (d, _) in enumerate(orig, 1)}
        i_rank = {d: r for r, (d, _) in enumerate(inv, 1)}

        cons = {
            d: 1 - abs(o_rank.get(d, norm + 1) - i_rank.get(d, norm + 1)) / norm
            for d in set(o_rank) | set(i_rank)
        }

        final = defaultdict(float)
        for d, s in orig + inv:
            final[d] += s * ((1 - lam) + lam * cons[d])
        return sorted(final.items(), key=lambda x: x[1], reverse=True)


    def rerank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        original = self.hybrid_search(query, top_k)
        inverted = self.hybrid_search(self.inverter.invert_sentence(query), top_k)

        combined = self._inverse_consistency(original, inverted)

        return combined[:top_k]
