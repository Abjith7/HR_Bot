import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use any HF chat model

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("docs")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def retrieve(query, k=4):
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents"]
    )
    return results["documents"][0]


def call_hf_inference(prompt, model=HF_MODEL):
    if not HUGGINGFACE_API_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN missing")

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.2
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def answer_query(user_question):
    docs = retrieve(user_question, k=4)
    context = "\n\n---\n\n".join(docs)

    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer. "
        "If the answer does not exist in the context, reply: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\nAnswer:"
    )

    reply = call_hf_inference(prompt)
    return reply
