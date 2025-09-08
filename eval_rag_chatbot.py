
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nutrition_insights"))

import json
from sentence_transformers import SentenceTransformer, util

# Path to your test set (JSONL: {"question": ..., "answer": ...})
TEST_SET = Path("test_questions.jsonl")


# Import the chatbot pipeline
from nutrition_insights.phase3.components import chatbot
import pandas as pd

# Load your data (simulate a DataFrame as in the app)
data_path = Path("nutrition_insights/data/combined.json")
if data_path.exists():
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame([])

def get_chatbot_response(question):
    # Use the same logic as the Streamlit app, but minimal for batch eval
    # You may want to adjust source_filter/window_days as needed
    # This assumes chatbot.render() is not directly callable for batch, so we mimic the core logic
    from nutrition_insights.phase3.utils.common import is_in_scope
    from nutrition_insights.phase3.utils.retrieval import build_context_snippets
    from nutrition_insights.phase3.langchain_agent import select_source_agent
    from nutrition_insights.phase3.utils.openai_client import nvidia_oss_chat_completion
    
    if not is_in_scope(question):
        return chatbot.OUT_OF_SCOPE
    agentic_source = select_source_agent(question)
    df_filtered = df.copy()
    if agentic_source == 'reddit_blogs' and 'source' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['source'].str.lower().isin(['reddit', 'blogs'])]
    elif agentic_source != 'all' and 'source' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['source'].str.lower() == agentic_source]
    # Use both 'text' and 'combined_text' columns for context if available
    df_for_context = df_filtered.copy()
    if "combined_text" in df_for_context.columns and "text" in df_for_context.columns:
        def merge_texts(row):
            t1 = str(row.get("text", "")).strip()
            t2 = str(row.get("combined_text", "")).strip()
            if t1 and t2:
                return t1 + "\n" + t2 if t1 != t2 else t1
            return t1 or t2
        df_for_context["text"] = df_for_context.apply(merge_texts, axis=1)
    elif "combined_text" in df_for_context.columns:
        df_for_context["text"] = df_for_context["combined_text"]
    snippets = build_context_snippets(df_for_context, question, topn=500, agentic_source=agentic_source)
    context = chatbot._format_context(snippets)
    user_prompt = (
        f"User question: {question}\n\n"
        f"Below are context snippets from research journals, blogs, and Reddit. "
        f"Base your answer strictly on the provided context. Do not repeat generic advice. "
        f"When using information from the context snippets, cite them in your answer using [n] where n is the snippet number. "
        f"At the end of your answer, include all [n] references you used. "
        f"If the context is not relevant, say: {chatbot.OUT_OF_SCOPE}. "
        f"If there are different opinions, summarize them. Use evidence from all sources (journals, blogs, reddit).\n\n"
        f"{context}\n\n"
        f"Give a clear, practical answer in 6–10 sentences.\n"
        f"If there is not enough information, reply exactly with:\n{chatbot.OUT_OF_SCOPE}"
    )
    reply = nvidia_oss_chat_completion(
        user_prompt,
        system=chatbot._system_prompt(),
        timeout=60
    )
    return reply or chatbot.OUT_OF_SCOPE

# Load test set
with TEST_SET.open("r", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f if line.strip()]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

scores = []
for case in test_cases:
    question = case["question"]
    gt_answer = case["answer"]
    pred_answer = get_chatbot_response(question)
    
    # Compute embedding similarity
    emb_gt = model.encode(gt_answer, convert_to_tensor=True)
    emb_pred = model.encode(pred_answer, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_gt, emb_pred).item()
    scores.append(sim)
    print(f"Q: {question}\nGT: {gt_answer}\nPred: {pred_answer}\nSimilarity: {sim:.3f}\n---")

print(f"\nAverage embedding similarity: {sum(scores)/len(scores):.3f}")
