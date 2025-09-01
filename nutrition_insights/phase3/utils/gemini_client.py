# phase3/utils/gemini_client.py

from typing import Optional
import os
import requests
import pandas as pd

# Import robust RAG logic
from nutrition_insights.phase3.services.query_router import is_in_scope, build_context_snippets
from nutrition_insights.phase3.utils.data import load_data

# --- robust .env loading ---
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional; if missing, rely on real environment vars
    pass


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

# Load the main filtered corpus as DataFrame (cached at module level)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
try:
    CORPUS_DF = load_data(DATA_DIR)
except Exception:
    CORPUS_DF = pd.DataFrame()

def chat_completion(prompt: str, system: Optional[str] = None, timeout: int = 60, model: Optional[str] = None) -> str:
    """
    Gemini LLM client using robust RAG context and strict out-of-scope handling.
    """
    if not GEMINI_API_KEY:
        return "⚠️ (Gemini) GEMINI_API_KEY is not set"

    # Strict out-of-scope guard
    if not is_in_scope(prompt):
        return "Out of scope — this question is not related to protein or the available data."

    # Use robust context retrieval (same as chatbot)
    df = CORPUS_DF.copy() if CORPUS_DF is not None else pd.DataFrame()
    snippets = build_context_snippets(df, prompt, topn=8)

    # Format context as in chatbot
    def _format_context(snippets):
        blocks = []
        for i, s in enumerate(snippets, 1):
            src = getattr(s, "source", "unknown")
            dt = getattr(s, "date", "")
            title = getattr(s, "title", "") or ""
            text = getattr(s, "excerpt", "") or ""
            blocks.append(f"[{i}] ({src} | {dt}) {title}\n{text}")
        return "\n\n".join(blocks)

    context = _format_context(snippets)

    OUT_OF_SCOPE = "Out of scope — this question is not related to protein or the available data."

    user_prompt = (
        f"User question: {prompt}\n\n"
        f"Below are context snippets from research journals, blogs, and Reddit. "
        f"Base your answer strictly on the provided context. Do not repeat generic advice. "
        f"If the context is not relevant, say: {OUT_OF_SCOPE}. "
        f"If there are different opinions, summarize them. Use evidence from all sources (journals, blogs, reddit).\n\n"
        f"{context}\n\n"
        f"Give a clear, practical answer in 6–10 sentences, then list 3–6 bullet citations with URLs from the snippets.\n"
        f"If there is not enough information, reply exactly with:\n{OUT_OF_SCOPE}"
    )

    # Compose system prompt as in chatbot
    system_prompt = system or (
        "You are ProteinScope Assistant. Answer strictly about dietary protein: "
        "types (whey/casein/plant), dosage, timing, safety, performance, consumer trends. "
        "Ground answers in the provided snippets. "
        "If you lack support or the query is off-topic, say: "
        f'"{OUT_OF_SCOPE}"'
    )

    # If context is too short, fallback to a user-friendly message
    if not snippets or len(context.strip()) < 40:
        return OUT_OF_SCOPE

    body = {"contents": [{"role": "user", "parts": [{"text": user_prompt}]}]}

    use_model = model if model else GEMINI_MODEL
    api_url = f"https://generativelanguage.googleapis.com/v1/models/{use_model}:generateContent"
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Strict out-of-scope filter
            if OUT_OF_SCOPE.lower() in result.lower() or "not related" in result.lower():
                return OUT_OF_SCOPE
            return result
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 503 and attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"⚠️ (Gemini API error) Service Unavailable (503). Please try again later."
        except Exception as e:
            return f"⚠️ (Gemini API error) {e}"