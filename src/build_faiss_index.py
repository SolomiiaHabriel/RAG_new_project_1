import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = "data/processed/chunks.json"
INDEX_DIR = "indexes"
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(INDEX_DIR, "chunks_metadata.pkl")

MODEL_NAME = "intfloat/multilingual-e5-base"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def create_embeddings(chunks):
    print("Завантажую embedding-модель...")
    model = SentenceTransformer(MODEL_NAME)

    texts = []

    for chunk in chunks:
        texts.append("passage: " + chunk["text"])

    print("Створюю embeddings для чанків...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    return embeddings


def build_faiss_index(embeddings):
    vector_dimension = embeddings.shape[1]

    print(f"Розмірність векторів: {vector_dimension}")

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    print(f"Кількість векторів в індексі: {index.ntotal}")

    return index


def save_index_and_metadata(index, chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(METADATA_FILE, "wb") as file:
        pickle.dump(chunks, file)

    print("\nГотово!")
    print(f"FAISS-індекс збережено: {FAISS_INDEX_FILE}")
    print(f"Метадані чанків збережено: {METADATA_FILE}")


def main():
    chunks = load_chunks()

    print(f"Завантажено чанків: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    save_index_and_metadata(index, chunks)


if __name__ == "__main__":
    main()