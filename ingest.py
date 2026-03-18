import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir):
    docs = []
    for p in data_dir.glob("**/*"):
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = p.name
            docs.extend(loaded)

        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = p.name
                d.metadata["page"] = 1
            docs.extend(loaded)

    return docs


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} raw documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for d in documents:
        for c in splitter.split_text(d.page_content):
            chunks.append({
                "content": c,
                "meta": {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", "NA")
                }
            })

    print(f"Total chunks after split: {len(chunks)}")

    emb = SentenceTransformer(EMBED_MODEL_NAME)

    client = PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["meta"] for c in chunks]

    print("Computing embeddings...")
    embeddings = emb.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    ids = [f"doc-{i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist()
    )

    print(f"Ingest complete. Chunks stored: {len(texts)}")
    print(f"Chroma DB location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
