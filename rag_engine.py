import os
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("docs")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def retrieve(query, k=4):
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def call_hf_inference(prompt):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 400
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    data = r.json()
    return data["choices"][0]["message"]["content"]
