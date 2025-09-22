from typing import Optional
def nvidia_oss_chat_completion(prompt: str, system: Optional[str] = None, timeout: int = 60) -> str:
    """
    Use NVIDIA OSS endpoint for chatbot completions.
    """
    from openai import OpenAI
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv("NVIDIA_API_KEY")
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=False
        )
        # Only return the answer content, ignore reasoning_content
        result = ""
        for choice in completion.choices:
            if choice.message.content:
                result += choice.message.content
        return result.strip()
    except Exception as e:
        return f"⚠️ (NVIDIA OSS API error) {e}"
# phase3/utils/openai_client.py


from typing import Optional
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv


# Import robust RAG logic only inside functions to avoid circular import
from nutrition_insights.phase3.utils.data import load_data


# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_OSS_MODEL = "openai-oss"
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the main filtered corpus as DataFrame (cached at module level)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
try:
    CORPUS_DF = load_data(DATA_DIR)
except Exception:
    CORPUS_DF = pd.DataFrame()


def chat_completion(prompt: str, system: Optional[str] = None, timeout: int = 60, model: Optional[str] = None) -> str:
    """
    OpenAI LLM client using robust RAG context and strict out-of-scope handling.
    """
    # Import here to avoid circular import
    from nutrition_insights.phase3.utils.common import is_in_scope
    from nutrition_insights.rag.retrieval import build_context_snippets
    # Strict out-of-scope guard
    if not is_in_scope(prompt):
        return "Out of scope — this question is not related to protein or the available data."

    # Use robust context retrieval (same as chatbot)
    df = CORPUS_DF.copy() if CORPUS_DF is not None else pd.DataFrame()
    snippets = build_context_snippets(df, prompt, topn=8)

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

    # If context is too short, fallback to a user-friendly message
    if not snippets or len(context.strip()) < 40:
        return OUT_OF_SCOPE

    # Compose system prompt
    system_prompt = system or (
        "You are ProteinScope Assistant. Answer strictly about dietary protein: "
        "types (whey/casein/plant), dosage, timing, safety, performance, consumer trends. "
        "Ground answers in the provided snippets. "
        "If you lack support or the query is off-topic, say: "
        f'\"{OUT_OF_SCOPE}\"'
    )

    # OpenAI API call
    try:
        response = client.chat.completions.create(
            model=model or OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt}\n\n{context}"}
            ],
            temperature=1,
            max_tokens=1024,
            timeout=timeout,
        )
        result = response.choices[0].message.content.strip()
        if OUT_OF_SCOPE.lower() in result.lower() or "not related" in result.lower():
            return OUT_OF_SCOPE
        return result
    except Exception as e:
        return f"⚠️ (OpenAI API error) {e}"