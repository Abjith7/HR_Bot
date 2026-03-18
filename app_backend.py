import time
from rag_engine import retrieve, call_hf_inference
from memory_manager import ConversationMemory

# ---------- COST CONFIG ----------
COST_PER_1K_INPUT_TOKENS = 0.002   # USD estimate
COST_PER_1K_OUTPUT_TOKENS = 0.002 # USD estimate
USD_TO_INR = 83


# ---------- TOKEN ESTIMATOR ----------
def estimate_tokens(text: str):
    return max(1, int(len(text) / 4))


# ---------- GREETING LOGIC ----------
GREETINGS = {
    "hi", "hello", "hey", "hai", "hii",
    "good morning", "good evening", "good afternoon",
    "gm", "ge"
}

def is_greeting(text: str) -> bool:
    return text.lower().strip() in GREETINGS


# ---------- CONVERSATION MEMORY ----------
memory = ConversationMemory(max_turns=8)


# ---------- MAIN CHAT FUNCTION ----------
def answer_query(user_question):
    try:
        user_question_clean = user_question.lower().strip()

        # ----- ✅ GREETING HANDLER -----
        if is_greeting(user_question_clean):
            greeting_reply = (
                "Hello, I'm Genie 🧞 — your personal document assistant.\n\n"
                "You can ask me things like:\n"
                "• What is this document about?\n"
                "• Which document does this come from?\n"
                "• Explain this simply\n"
            )
            return {
                "answer": greeting_reply,
                "sources": [],
                "latency": {},
                "usage": {},
                "cost_inr": 0
            }

        t0 = time.time()

        # ---- Conversation context ----
        chat_context = memory.get_context()

        results = retrieve(user_question, k=4)
        t1 = time.time()

        if not results or not results.get("documents"):
            return {
                "answer": "No documents are available for retrieval.",
                "sources": [],
                "latency": {},
                "usage": {},
                "cost_inr": 0
            }

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        scores = results["distances"][0]

        # ---- Hallucination guard ----
        if min(scores) > 0.75:
            answer = "I don't have enough reliable information to answer that."
            memory.add_turn(user_question, answer, [])
            return {
                "answer": answer,
                "sources": [],
                "latency": {},
                "usage": {},
                "cost_inr": 0
            }

        context_blocks = []
        sources = []

        for d, m, s in zip(docs, metas, scores):
            context_blocks.append(d)
            src = m.get("source", "unknown")
            page = m.get("page", "NA")
            similarity = round(1 - s, 3)
            sources.append(f"{src} | page {page} | similarity {similarity}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are Genie, a professional knowledge assistant.

Conversation so far:
{chat_context}

Use ONLY the following retrieved context to answer.
Always cite sources at the end.
If not found, say: I don't know.

Context:
{context}

User Question:
{user_question}

Answer:
""".strip()

        # ----- TOKEN ESTIMATION (INPUT) -----
        input_tokens = estimate_tokens(prompt)

        reply = call_hf_inference(prompt)

        # ----- TOKEN ESTIMATION (OUTPUT) -----
        output_tokens = estimate_tokens(reply)

        t2 = time.time()

        latency = {
            "retrieval_time": round(t1 - t0, 3),
            "generation_time": round(t2 - t1, 3),
            "total_time": round(t2 - t0, 3)
        }

        # ----- COST CALCULATION -----
        cost_usd = (
            (input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS +
            (output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
        )
        cost_inr = round(cost_usd * USD_TO_INR, 4)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        memory.add_turn(user_question, reply, sources)

        return {
            "answer": reply,
            "sources": sources,
            "latency": latency,
            "usage": usage,
            "cost_inr": cost_inr
        }

    except Exception as e:
        # ----- FAIL-SAFE RETURN -----
        return {
            "answer": f"Backend error: {str(e)}",
            "sources": [],
            "latency": {},
            "usage": {},
            "cost_inr": 0
        }
