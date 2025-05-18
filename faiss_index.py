from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss_index

class FAISSIndex:
    def __init__(self,nlist=100, m=8, n_bits=8):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.m = m
        self.n_bits = n_bits

    def encode_documents(self, documents):
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss_index.normalize_L2(embeddings)
        return embeddings

    def build_faiss_index(self, embeddings):
        d = embeddings.shape[1]
        self.index = faiss_index.IndexFlatL2(d)
        self.index.add(embeddings)

    def search_faiss_index(self, query_embedding, top_k=10):
        query_embeddings = np.expand_dims(query_embedding, axis=0).astype("float32")
        faiss_index.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices