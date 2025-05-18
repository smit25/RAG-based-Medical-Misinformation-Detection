import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
from utils import device



class Ranker:
    def __init__(self, _bm25, _faiss_index, _corpus_ids, _inverter, _rank_type = "irc"): #options = ["ce", "irc"]
        self.inverter = _inverter
        self.rank_type = _rank_type

        self.faiss_indexer = _faiss_index
        self.bm25 = _bm25
        self.corpus_ids = _corpus_ids

        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2").to(device)
        self.model.eval()


    def preprocess(self, query):
        return word_tokenize(query.lower())


    def faiss_search(self,query, top_k=10):
        query_embedding = self.faiss_indexer.embedding_model.encode(query, convert_to_tensor=False)
        distances, indices = self.faiss_indexer.search_faiss_index(query_embedding, top_k)
        return [(self.corpus_ids[int(doc_idx)], float(dist)) for doc_idx, dist in zip(indices[0], distances[0])]


    def bm25_search(self,query, top_k=10):
        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus_ids[idx], float(scores[idx])) for idx in top_indices]


    def reciprocal_rank_fusion(self,results_list, k=20):
        fused_scores = defaultdict(float)

        for results in results_list:
            for rank, (doc_id, score) in enumerate(results):
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


    def hybrid_search(self,query, top_k = 20):
        faiss_results = self.faiss_search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)

        hybrid_results = self.reciprocal_rank_fusion([bm25_results, faiss_results])
        return hybrid_results


    def normalize_scores(self, ranking):
        if not ranking:
            return ranking
        scores = [score for doc, score in ranking]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return ranking
        return [(doc, (score - min_score) / (max_score - min_score)) for doc, score in ranking]


    def borda_fusion(self, results_list, top_k=20):
        fused_scores = defaultdict(float)
        for ranked_list in results_list:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                fused_scores[doc_id] += (top_k - rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


    def inverse_consistency_rerank(self, original_ranking, reversed_ranking, lambda_=0.5, alpha = 0.8):
        original_ranking = self.normalize_scores(original_ranking)
        reversed_ranking = self.normalize_scores(reversed_ranking)

        norm = max(len(original_ranking), len(reversed_ranking))
        original_rank_dict = {doc: rank for rank, (doc, _) in enumerate(original_ranking, start=1)}
        reversed_rank_dict = {doc: rank for rank, (doc, _) in enumerate(reversed_ranking, start=1)}

        all_docs = set(original_rank_dict.keys()).union(set(reversed_rank_dict.keys()))

        consistency_scores = {}
        for doc in all_docs:
            r_orig = original_rank_dict.get(doc, norm + 1)
            r_inv = reversed_rank_dict.get(doc, norm + 1)
            rank_diff = abs(r_orig - r_inv)
            consistency_scores[doc] = 1 - (rank_diff / norm)
            # consistency_scores[doc] = np.exp(-alpha * rank_diff)

        final_scores = defaultdict(float)
        for doc, score in original_ranking:
            stability_weight = (1 - lambda_) + lambda_ * consistency_scores.get(doc, 0)
            final_scores[doc] += score * stability_weight
        for doc, score in reversed_ranking:
            stability_weight = (1 - lambda_) + lambda_ * consistency_scores.get(doc, 0)
            final_scores[doc] += score * stability_weight

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


    def cross_encoder_rerank(self, query, candidate_docs, max_length=512):
        pairs = [(query, doc_text) for doc_id, doc_text in candidate_docs]

        inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits.squeeze(-1).tolist()
        ranked_results = sorted(zip([doc_id for doc_id, _ in candidate_docs], scores),
                                  key=lambda x: x[1], reverse=True)
        return ranked_results


    def rerank(self, query, top_k = 10):
        final_results = {}
        original_ranking = self.hybrid_search(query, top_k = top_k)
        combined_ranking = original_ranking
        if self.rank_type == "ce":
            pass
            # candidate_docs = [(doc_id, corpus[doc_id]["text"]) for doc_id, _ in original_ranking]
            # combined_ranking = cross_encoder_rerank(query, candidate_docs)
        else:
            inverted_query = self.inverter.invert_sentence(query)
            inverted_ranking = self.hybrid_search(inverted_query, top_k = top_k)
            combined_ranking = self.inverse_consistency_rerank(original_ranking, inverted_ranking)

        return combined_ranking
    



