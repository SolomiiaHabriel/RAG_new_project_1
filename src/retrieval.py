import os
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


INDEX_DIR = "indexes"
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(INDEX_DIR, "chunks_metadata.pkl")

MODEL_NAME = "intfloat/multilingual-e5-base"
def load_index_and_chunks():
    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(METADATA_FILE, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks

def tokenize(text):
    return text.lower().split()

def bm25_search(query, chunks, top_k=10):
    corpus = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [tokenize(text) for text in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices):
        results.append({
            "chunk": chunks[idx],
            "score": float(scores[idx]),
            "rank": rank + 1,
            "retriever": "bm25"
        })

    return results

def dense_search(query, index, chunks, model, top_k=10):
    query_text = "query: " + query

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(scores[0][rank]),
            "rank": rank + 1,
            "retriever": "dense"
        })

    return results

def reciprocal_rank_fusion(result_lists, top_k=5, k=60):
    fused_scores = {}

    for results in result_lists:
        for result in results:
            chunk_id = result["chunk"]["id"]
            rank = result["rank"]

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    "chunk": result["chunk"],
                    "score": 0.0
                }

            fused_scores[chunk_id]["score"] += 1 / (k + rank)

    ranked_results = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_results[:top_k]

def hybrid_search(query, top_k=5):
    index, chunks = load_index_and_chunks()

    print("Завантажую embedding-модель...")
    model = SentenceTransformer(MODEL_NAME)

    bm25_results = bm25_search(query, chunks, top_k=10)
    dense_results = dense_search(query, index, chunks, model, top_k=10)

    final_results = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        top_k=top_k
    )

    return final_results

if __name__ == "__main__":
    question = "Що таке штучний інтелект?"

    results = hybrid_search(question, top_k=5)

    print("\nПитання:")
    print(question)

    print("\nЗнайдені чанки:")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Результат {i} ---")
        print(f"RRF score: {result['score']}")
        print(f"Source: {result['chunk']['source']}")
        print(result["chunk"]["text"][:500])