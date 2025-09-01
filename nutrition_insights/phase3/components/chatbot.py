from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# phase3/components/chatbot.py

import streamlit as st
import pandas as pd


# Use Gemini for LLM, but RAG for retrieval/context
from nutrition_insights.phase3.utils.openai_client import chat_completion

# Import RAG retrieval and context logic
from nutrition_insights.phase3.services.query_router import is_in_scope, build_context_snippets


def get_openai_response(prompt: str, system: str = None, timeout: int = 60) -> str:
    return chat_completion(prompt, system=system, timeout=timeout)


OUT_OF_SCOPE = (
    "Out of scope — this question is not related to protein or the available data."
)


def _init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[dict(role, content)]


def _render_history():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


def _system_prompt() -> str:
    return (
        "You are ProteinScope Assistant. Answer strictly about dietary protein: "
        "types (whey/casein/plant etc), dosage, timing, safety, performance, consumer trends. "
        "Ground answers in the provided snippets. "
        "If you lack support or the query is off-topic, say: "
        f"\"{OUT_OF_SCOPE}\""
    )


def _format_context(snippets: list[dict]) -> str:
    # Each snippet: Snippet object with attributes
    blocks = []
    for i, s in enumerate(snippets, 1):
        src = getattr(s, "source", "unknown")
        dt = getattr(s, "date", "")
        title = getattr(s, "title", "") or ""
        text = getattr(s, "excerpt", "") or ""
        blocks.append(f"[{i}] ({src} | {dt}) {title}\n{text}")
    return "\n\n".join(blocks)


def render(df: pd.DataFrame, source_filter: str, window_days: int) -> None:

    st.subheader("Chatbot")
    _init_state()
    _render_history()

    q = st.chat_input("Ask about protein (timing, dose, safety, trends)...")
    if not q:
        st.caption("Tip: try “Best time for whey?”, “Collagen trending complaints?”, “BCAA vs EAA?”.")
        return

    # Guardrail
    if not is_in_scope(q):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": OUT_OF_SCOPE})
        with st.chat_message("assistant"):
            st.markdown(OUT_OF_SCOPE)
        st.info(":mag: [DEBUG] Out of scope — not a protein-related question.")
        return

    # Filter dataframe by source_filter and window_days before building context snippets
    df_filtered = df.copy()
    if source_filter and source_filter != "All" and "source" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["source"].str.lower() == source_filter.lower()]
    if window_days and "date" in df_filtered.columns:
        try:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
            # Try parsing with multiple formats, fallback to default
            def parse_date(val):
                for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y %b", "%Y %b %d"):
                    try:
                        return pd.to_datetime(val, format=fmt, errors="raise")
                    except Exception:
                        continue
                return pd.to_datetime(val, errors="coerce")
            parsed_dates = df_filtered["date"].apply(parse_date)
            df_filtered = df_filtered[parsed_dates >= cutoff]
        except Exception:
            pass
    snippets = build_context_snippets(df_filtered, q, topn=8)
    context = _format_context(snippets)
    # Debug: Show why out of scope
    if not snippets:
        st.warning(f"[DEBUG] No context snippets found for: '{q}' (df_filtered rows: {len(df_filtered)})")
    elif len(context.strip()) < 40:
        st.warning(f"[DEBUG] Context too short for: '{q}' (context: '{context.strip()}')")


    # Compose prompt for OpenAI (encourage complete, practical answers)
    user_prompt = (
        f"User question: {q}\n\n"
        f"Below are context snippets from research journals, blogs, and Reddit. "
        f"Base your answer strictly on the provided context. Do not repeat generic advice. "
        f"If the context is not relevant, say: {OUT_OF_SCOPE}. "
        f"If there are different opinions, summarize them. Use evidence from all sources (journals, blogs, reddit).\n\n"
        f"{context}\n\n"
        f"Give a clear, practical answer in 6–10 sentences.\n"
        f"If there is not enough information, reply exactly with:\n{OUT_OF_SCOPE}"
    )

    with st.spinner("Thinking..."):
        reply = chat_completion(
            prompt=user_prompt,
            system=_system_prompt(),
            timeout=180,
        )

    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.chat_history.append({"role": "assistant", "content": reply or OUT_OF_SCOPE})

    # Show the user's question above the assistant's answer in a visually distinct way

    # --- Insert Reddit citation phrase in answer if Reddit reference is cited ---
    answer = reply or OUT_OF_SCOPE
    import re
    cited_nums = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        try:
            cited_nums.add(int(m.group(1)))
        except Exception:
            pass
    # Check if any cited reference is from Reddit
    reddit_cited = False
    for i, s in enumerate(snippets, 1):
        if i in cited_nums and getattr(s, 'source', '').lower() == 'reddit':
            reddit_cited = True
            break
    # If Reddit is cited, prepend phrase
    if reddit_cited and 'according to a reddit post' not in answer.lower():
        answer = 'According to a Reddit post, ' + answer[0].lower() + answer[1:] if answer else answer

    with st.chat_message("assistant"):
        st.info(f"**Question:** {q}")
        st.markdown(answer)

    # --- Show only references actually cited in the answer ---
    import re
    cited_nums = set()
    for m in re.finditer(r"\[(\d+)\]", reply or ""):
        try:
            cited_nums.add(int(m.group(1)))
        except Exception:
            pass
    references = []
    for i, s in enumerate(snippets, 1):
        if i in cited_nums:
            title = getattr(s, 'title', '(no title)')
            url = getattr(s, 'url', None) or getattr(s, 'permalink', None) or ''
            src = getattr(s, 'source', '?')
            date = getattr(s, 'date', '')
            if url:
                references.append(f"[{title}]({url}) — {src} | {date}")
            else:
                references.append(f"{title} — {src} | {date}")
    if references:
        with st.expander("References cited in answer"):
            for ref in references:
                st.markdown(f"- {ref}")