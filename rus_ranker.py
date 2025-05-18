# langchain_hybrid_ranker.py
from __future__ import annotations
from typing import List, Tuple, Dict, Sequence
from collections import defaultdict
import numpy as np
import nltk
from scipy.stats import spearmanr

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document


# from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RUSRanker:
    """
    Hybrid (vector + lexical) retriever with optional CE or IRC re‑ranking.
    Mirrors the logic of your original Ranker, but built on LangChain components.
    """

    def __init__(self, faiss_store: FAISS, bm25_retriever: BM25Retriever, rank_type: str = "irc", ):
        self.rank_type = rank_type.lower()

        self.faiss_retriever = faiss_store #faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def _docs_to_pairs(docs: Sequence[Document]) -> List[Tuple[str, float]]:
        """
        Convert LangChain `Document` objects to `(id, score)` pairs.
        Assumes each document was stored with `metadata={'id': ...}`.
        """
        return [(d.metadata["name"], float(getattr(d, "score", 0.0))) for d in docs]


    @staticmethod
    def _normalize(ranking: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not ranking:
            return ranking
        scores = [s for _, s in ranking]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return ranking
        return [(d, (s - lo) / (hi - lo)) for d, s in ranking]
    

    # Normalized Discounted Cumulative Relevance (DCR)
    def dcg(self,scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))


    def calculate_rus(self,similarity_scores, relevance_scores, alpha=0.5, beta=0.4, gamma=0.1):
        
        ideal_relevance = sorted(relevance_scores, reverse=True)
        actual_dcr = self.dcg(relevance_scores)
        ideal_dcr = self.dcg(ideal_relevance)
        normalized_dcr = actual_dcr / ideal_dcr if ideal_dcr > 0 else 0

        # Spearman rank correlation between similarity and relevance
        corr, _ = spearmanr(similarity_scores, relevance_scores)
        scaled_corr = (corr + 1) / 2  # Scale to [0, 1]

        # Wasted Similarity Penalty
        total_similarity = sum(similarity_scores)
        wasted_similarity = sum(sim for sim, rel in zip(similarity_scores, relevance_scores) if rel == 0)
        waste_penalty = wasted_similarity / total_similarity if total_similarity > 0 else 0

        rus = alpha * normalized_dcr + beta * scaled_corr - gamma * waste_penalty

        return {
            "RUS": rus,
            "Normalized_DCR": normalized_dcr,
            "Scaled_Correlation": scaled_corr,
            "Wasted_Similarity_Penalty": waste_penalty
        }


    def rus_search(self, docs, similarity_scores, relevance_scores, k: int = 5, rus_threshold: float = 0.5) -> List[Tuple[str, float]]:
        # Calculate baseline RUS for all documents
        sims = list(similarity_scores)
        rels = list(relevance_scores)
        baseline_stats = self.calculate_rus(sims, rels)
        baseline_rus = baseline_stats["RUS"]
        
        docs_with_rus = []
        for i in range(len(docs)):
            sim_without_i = sims[:i] + sims[i+1:]
            rel_without_i = rels[:i] + rels[i+1:]
            
            # Calculate RUS without this document
            stats_without_i = self.calculate_rus(sim_without_i, rel_without_i)
            rus_without_i = stats_without_i["RUS"]
            
            rus_impact = baseline_rus - rus_without_i
            
            doc = docs[i]
            doc.metadata["rus_impact"] = rus_impact
            doc.metadata["Normalized_DCR"] = stats_without_i["Normalized_DCR"]
            doc.metadata["Scaled_Correlation"] = stats_without_i["Scaled_Correlation"]
            doc.metadata["Wasted_Similarity_Penalty"] = stats_without_i["Wasted_Similarity_Penalty"]
            docs_with_rus.append((doc, rus_impact))
        
        docs_with_rus = sorted(docs_with_rus, key=lambda x: x[1], reverse=True)
        # for d in docs_with_rus:
            # print(f"Document: {d[0].page_content}, RUS Impact: {d[0].metadata['rus_impact']}")
            # print("         =============             ")
        
        filtered_docs = [doc for doc, impact in docs_with_rus if impact >= 0]
        
        # If no documents meet the threshold, return the top document by RUS impact
        if not filtered_docs and docs_with_rus:
            docs_with_rus.sort(key=lambda x: x[1], reverse=True)
            filtered_docs = [docs_with_rus[0][0]]
        
        return filtered_docs[:min(len(filtered_docs),3)]


    def get_documents_by_similarity(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        similarity_docs = self.faiss_retriever.similarity_search_with_relevance_scores(query, k=k)
        # similarity_docs = self._normalize(similarity_docs)
        similairty_scores = [s for _, s in similarity_docs]
        docs = [Document(page_content=d.page_content, metadata={"name": d.metadata["name"]}, score=s)
                for d, s in similarity_docs]
        return docs, similairty_scores


        # sentences = nltk.sent_tokenize(query)

        # seen_contents = set()
        # docs = []
        # scores = []

        # for sentence in sentences:
        #     similarity_docs = self.faiss_retriever.similarity_search_with_relevance_scores(sentence, k=5)
        #     for doc, score in similarity_docs:
        #         if doc.page_content not in seen_contents:
        #             seen_contents.add(doc.page_content)
        #             docs.append(Document(
        #                 page_content=doc.page_content,
        #                 metadata={"name": doc.metadata.get("name", "")},
        #                 score=score
        #             ))
        #             scores.append(score)

        # return docs, scores



    def reciprocal_rank_fusion(lists: List[List[Tuple[str, float]]], k: int = 40) -> List[Tuple[str, float]]:
        scores = defaultdict(float)
        for result in lists:
            for rank, (doc_id, _) in enumerate(result):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

