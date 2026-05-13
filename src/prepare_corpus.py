import os
import json
from pypdf import PdfReader
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "chunks.json")


WIKI_ARTICLES = [
    "Штучний інтелект",
    "Машинне навчання",
    "Нейронна мережа",
]


def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_pdfs():
    documents = []

    if not os.path.exists(RAW_DATA_DIR):
        print("Папка data/raw не знайдена.")
        return documents

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            print(f"Читаю PDF: {filename}")

            text = load_pdf_text(file_path)

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def load_wikipedia_articles():
    documents = []

    wikipedia.set_lang("uk")

    for article_title in WIKI_ARTICLES:
        try:
            print(f"Завантажую Wikipedia: {article_title}")

            page = wikipedia.page(article_title)

            documents.append({
                "source": f"Wikipedia: {article_title}",
                "text": page.content
            })

        except Exception as e:
            print(f"Не вдалося завантажити статтю: {article_title}")
            print(e)

    return documents


def split_documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])

        for chunk_text in text_chunks:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "source": doc["source"]
            })
            chunk_id += 1

    return chunks


def save_chunks(chunks):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"\nГотово!")
    print(f"Кількість чанків: {len(chunks)}")
    print(f"Файл збережено: {OUTPUT_FILE}")


def main():
    pdf_documents = load_pdfs()
    wiki_documents = load_wikipedia_articles()

    all_documents = pdf_documents + wiki_documents

    print(f"\nУсього документів: {len(all_documents)}")

    chunks = split_documents_into_chunks(all_documents)

    save_chunks(chunks)


if __name__ == "__main__":
    main()