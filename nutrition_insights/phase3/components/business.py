# phase3/components/business.py
from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd
from textwrap import shorten
import json
import re
from collections import Counter, defaultdict

from utils.openai_client import chat_completion
from utils.common import looks_protein_related


# ----------------- Helper Functions -----------------

def looks_booming(text: str) -> bool:
    """Detect 'booming' signals in text."""
    if not isinstance(text, str):
        return False
    booming_keywords = [
        "growth", "trend", "surge", "rising", "increasing", "hot", "booming",
        "popular", "demand", "momentum", "expanding", "spike", "uptick",
        "interest", "buzz", "adoption", "exploding", "taking off", "mainstream",
        "skyrocketing", "rapid", "accelerating"
    ]
    return any(word in text.lower() for word in booming_keywords)


def _slice_for_business(df: pd.DataFrame, source_filter: str, max_items: int = 120) -> pd.DataFrame:
    """Filter for recent, long-text entries."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if source_filter and source_filter != "All":
        sf = source_filter.lower()
        d = d[d["source"].str.lower() == sf]

    if "text" in d.columns:
        textlen = d["text"].fillna("").str.len()
    elif "abstract" in d.columns:
        textlen = d["abstract"].fillna("").str.len()
    else:
        textlen = pd.Series([0] * len(d))

    d = d.assign(_q=(textlen > 160).astype(int))
    if "date" in d.columns:
        d = d.sort_values(["_q", "date"], ascending=[False, False])
    else:
        d = d.sort_values("_q", ascending=False)

    return d.head(max_items).reset_index(drop=True)


def _compose_market_context(rows: pd.DataFrame, max_chars: int = 9000) -> str:
    """Compact context builder for Gemini."""
    lines = []
    for _, r in rows.iterrows():
        title = str(r.get("title") or r.get("kind") or r.get("subreddit") or "Item").strip()
        src = str(r.get("source", "") or "").strip().capitalize()
        body = str(r.get("abstract") or r.get("summary") or r.get("text") or "").replace("\n", " ").strip()
        body = shorten(body, width=420, placeholder=" …")

        meta = []
        if pd.notna(r.get("date")):
            try:
                meta.append(pd.to_datetime(r["date"]).date().isoformat())
            except Exception:
                pass
        if src:
            meta.append(src)

        meta_str = " • ".join(meta)
        head = f"- {title}" + (f" ({meta_str})" if meta_str else "")
        lines.append(f"{head}\n  {body}")

        if sum(len(x) for x in lines) > max_chars:
            break

    return "MARKET CONTEXT (mixed sources):\n" + "\n".join(lines)


# ---------- TEXT & KEYWORD HELPERS ----------

_WORD_BOUNDARY = r"(?i)(?<![A-Za-z0-9]){kw}(?![A-Za-z0-9])"

def _safe_iter_text(series):
    for x in series.fillna(""):
        if isinstance(x, str):
            yield x
        else:
            yield str(x)

def _count_keywords(text_series: pd.Series, keywords: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(keywords, 0)
    for txt in _safe_iter_text(text_series):
        for kw in keywords:
            if re.search(_WORD_BOUNDARY.format(kw=re.escape(kw)), txt):
                counts[kw] += 1
    return counts

def _count_keyword_groups(text_series: pd.Series, groups: dict[str, list[str]]) -> dict[str, int]:
    """Counts at group level (a hit for any word in the group counts once per document)."""
    group_counts = dict.fromkeys(groups.keys(), 0)
    for txt in _safe_iter_text(text_series):
        for gname, kws in groups.items():
            for kw in kws:
                if re.search(_WORD_BOUNDARY.format(kw=re.escape(kw)), txt):
                    group_counts[gname] += 1
                    break  # avoid double counting a group within the same doc
    return group_counts


# --- Horizontal bar chart using Plotly for premium look ---
from components.plot_utils_business import horizontal_bar_chart, horizontal_bar_chart_series

def _series_bar(series: pd.Series, title: str, sort_desc: bool = True, top_n: int | None = None):
    s = series.dropna()
    if sort_desc:
        s = s.sort_values(ascending=True)
    if top_n:
        s = s.tail(top_n)
    horizontal_bar_chart_series(s, title=title)


# ----------------- Main Render Function -----------------


@st.cache_data(show_spinner=False)
def cached_analysis(refresh_key: int = 0, source_filter: str = "All", model: str | None = None):
    """
    Cached business analysis. Invalidate only when refresh_key changes.
    """
    data_path = "../data/corpus_filtered.jsonl"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            combined_data = [json.loads(line) for line in f]
    except Exception as e:
        return None, f"❌ Failed to load filtered corpus: {e}"
    return combined_data, None


def clean_llm_output(text: str) -> str:
    """Convert messy LLM text into clean bullet points."""
    import re

    # 1. Remove markdown/code fences & headings
    text = re.sub(r"^```[a-zA-Z]*", "", text.strip(), flags=re.MULTILINE)
    text = text.replace("```", "")
    text = re.sub(r"^\s*(```|~~~).*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+ .*", "", text, flags=re.MULTILINE)

    # 2. Normalize line breaks
    text = re.sub(r"\n+", "\n", text)

    # 3. Split into candidate lines (handles -, *, •, numbered, emoji bullets)
    raw_lines = re.split(r"(?:^|\n)\s*(?:[-*•]|\d+\.|👉|➡️|✔️|✅|⚡|🔹|▪️)\s*", text)

    # 4. Clean each line
    bullets = []
    for l in raw_lines:
        l = l.strip(" .-•\t")
        if not l:
            continue
        bullets.append(f"- {l}")

    # 5. Deduplicate
    seen = set()
    unique_bullets = []
    for b in bullets:
        if b not in seen:
            unique_bullets.append(b)
            seen.add(b)

    return "\n".join(unique_bullets)


def render(source_filter: str = None, model: str | None = None, refresh_key: int = 0) -> None:
    # ...existing code...
    combined_data, err = cached_analysis(refresh_key, source_filter or "All", model)
    if err:
        st.error(err)
        return

    df = pd.DataFrame(combined_data)

    # Optional source filter
    if source_filter and source_filter != "All" and "source_type" in df.columns:
        df = df[df["source_type"].str.lower() == source_filter.lower()]

    # --- Visualization: Preferred Forms of Protein Consumption ---
    st.markdown("## 🥤 Preferred Forms of Protein Consumption")
    form_keywords = {
        "Shake/Drink": ["shake", "drink", "rtd", "ready to drink", "bottled"],
        "Bar/Snack": ["bar", "snack", "protein bar", "energy bar", "snack bar"],
        "Powder": ["powder", "scoop", "mix", "whey powder", "casein powder", "soy powder"],
        "Capsule/Tablets": ["capsule", "tablet", "pill", "softgel"],
        "Food/Meal": ["meal", "food", "breakfast", "lunch", "dinner", "recipe"],
        "Yogurt/Dairy": ["yogurt", "curd", "dahi", "dairy", "milk", "paneer", "cheese"],
        "Other": ["cookie", "biscuit", "ice cream", "dessert", "cereal", "granola"]
    }
    form_counts = {k: 0 for k in form_keywords}
    for text in df.get("combined_text", []):
        if isinstance(text, str):
            for form, kws in form_keywords.items():
                for kw in kws:
                    if re.search(rf"\\b{re.escape(kw)}\\b", text, re.IGNORECASE):
                        form_counts[form] += 1
                        break
    form_counts = {k: v for k, v in form_counts.items() if v > 0}
    if form_counts:
        st.bar_chart(pd.Series(form_counts).sort_values(), use_container_width=True)
    else:
        st.info("No clear preference for protein forms detected in this dataset.")
    """
    Business Dashboard: Reads combined.json, analyzes with OpenAI,
    shows attractive insights + visualizations.
    """
    combined_data, err = cached_analysis(refresh_key, source_filter or "All", model)
    if err:
        st.error(err)
        return

    df = pd.DataFrame(combined_data)

    # Optional source filter
    if source_filter and source_filter != "All" and "source_type" in df.columns:
        df = df[df["source_type"].str.lower() == source_filter.lower()]

    # --- Visualization 1: Top brands ---
    st.markdown("### 🏷️ Top Mentioned Brands")
    brand_keywords = [
        "MyProtein", "Optimum Nutrition", "Dymatize", "MuscleBlaze", "MuscleTech",
        "GNC", "Ultimate Nutrition", "Labrada", "BigMuscles", "BSN",
        "Cellucor", "Isopure", "Evlution Nutrition", "Kaged Muscle",
        "Rule One", "Ronnie Coleman Signature Series",
        "Transparent Labs", "Legion Athletics", "Bulk Powders", "Naked Nutrition",
        "Orgain", "Vega", "Garden of Life", "PlantFusion", "Sunwarrior",
        "Vital Proteins", "PEScience", "Ghost Lifestyle", "Alani Nu",
        "Redcon1", "JYM Supplement Science", "Xtend", "RSP Nutrition",
        "HealthKart", "Ultimate Nutrition India", "Proburst", "Sinew Nutrition",
        "MuscleXP", "Fast&Up", "Wellcore", "BigFlex", "AS-IT-IS Nutrition",
        "Bodybuilding.com", "Muscle & Strength", "BarBend", "True Nutrition",
        "Promix Nutrition", "BPI Sports", "Met-Rx", "Six Star Pro Nutrition"
    ]
    brand_counts = {b: 0 for b in brand_keywords}
    for text in df.get("combined_text", []):
        if isinstance(text, str):
            for brand in brand_keywords:
                if re.search(re.escape(brand), text, re.IGNORECASE):
                    brand_counts[brand] += 1
    brand_counts = {b: c for b, c in brand_counts.items() if c > 0}
    if brand_counts:
        horizontal_bar_chart(brand_counts, title="Top Mentioned Brands", x_label="Mentions", y_label="Brand")
    else:
        st.info("No major brand mentions found in this dataset.")

    st.divider()

    # =========================
    # 2) CONSUMER PREFERENCE TRENDS
    # =========================
    st.markdown("## 🌱 Consumer Preference Trends")
    attr_groups = {
        "Protein Type • Whey/Isolate": ["whey", "isolate", "wpi", "concentrate", "hydrolysate"],
        "Protein Type • Casein": ["casein", "micellar"],
        "Protein Type • Plant": ["plant-based", "plant based", "vegan", "pea protein", "soy protein", "rice protein"],
        "Protein Type • Collagen": ["collagen", "collagen peptides"],
        "Format • RTD / Drinks": ["rtd", "ready to drink", "protein shake", "bottled shake"],
        "Format • Bars/Snacks": ["protein bar", "snack bar", "energy bar"],
        "Attributes • Low Sugar/Carb": ["low sugar", "no sugar", "sugar-free", "low carb", "keto"],
        "Attributes • Lactose/Gluten": ["lactose-free", "lactose free", "gluten-free", "gluten free"],
        "Attributes • Clean/Organic": ["clean label", "organic", "non-gmo", "grass-fed", "natural"],
        "Taste • Flavors": ["chocolate", "vanilla", "strawberry", "coffee", "cookie", "unflavored", "salted caramel"],
        "Convenience • Mix/Sachets": ["mixability", "scoop", "single-serve", "sachet", "sticks", "on-the-go"]
    }
    attr_counts = _count_keyword_groups(df.get("combined_text", pd.Series([], dtype=object)), attr_groups)
    attr_series = pd.Series(attr_counts, dtype="int64")
    if attr_series.sum() > 0:
        horizontal_bar_chart_series(attr_series, title="Mentions by Consumer Preference Group", x_label="Mentions", y_label="Group")
    else:
        st.info("No clear consumer preference keywords detected in this dataset.")

    st.divider()

    # --- Prepare OpenAI context ---
    entries, refs = [], []
    for _, row in df.iterrows():
        stype = row.get("source_type", "")
        title = row.get("title", "")
        url = row.get("url", "")
        text = row.get("combined_text", "")
        if isinstance(text, str) and len(text) > 50:
            entries.append(f"[{stype}] {title}\n{text}")
            refs.append(f"- **{stype}** | [{title}]({url})")

    max_entries = 120
    context = "\n\n".join(entries[:max_entries])
    references = "\n".join(refs[:max_entries])

    openai_prompt = (
        "Analyze the following recent protein market news, research, and discussions. "
        "Summarize into **3 categories**: (1) 📈 Trends, (2) 💡 Business Opportunities, (3) ⚠ Risks & Challenges. "
        "For each category, return ONLY short bullet points. "
        "Each point must start with a '-' and be on a separate line. "
        "Focus on practical, actionable insights for business and product teams—avoid deep scientific jargon.\n\n"
        + context
    )

    openai_model = "gpt-4o-mini"

    @st.cache_data(show_spinner=False)
    def cached_openai_analysis(prompt, model):
        try:
            return chat_completion(prompt, model=model, timeout=180)
        except Exception as e:
            return f"❌ OpenAI API error: {e}"

    openai_analysis = cached_openai_analysis(openai_prompt, openai_model)

    if openai_analysis.startswith("❌ OpenAI API error"):
        st.error(openai_analysis)
    else:
        trends = openai_analysis.split('💡 Business Opportunities')[0].strip() if '💡 Business Opportunities' in openai_analysis else openai_analysis
        bizop, risks = "", ""
        if '💡 Business Opportunities' in openai_analysis:
            after_biz = openai_analysis.split('💡 Business Opportunities')[1]
            if '⚠ Risks & Challenges' in after_biz:
                bizop = after_biz.split('⚠ Risks & Challenges')[0].strip()
                risks = after_biz.split('⚠ Risks & Challenges')[1].strip()
            else:
                bizop = after_biz.strip()

        trends = clean_llm_output(trends)
        bizop = clean_llm_output(bizop)
        risks = clean_llm_output(risks)

        # --- Premium Matte Black UI ---
        st.markdown(f"""
        <div style='display: flex; flex-wrap: wrap; gap: 32px; justify-content: center; margin-bottom: 32px;'>
            <div style='flex:1 1 320px; min-width:320px; max-width:420px; background: linear-gradient(135deg,#232526 0%,#414345 100%); border-radius:32px; box-shadow:0 8px 32px rgba(60,60,120,0.18); border:2px solid #2d2d2d; padding:32px 28px 28px 28px;'>
                <div style='font-size:2.2rem; margin-bottom:10px;'>📈</div>
                <h3 style='margin-bottom:12px;font-size:1.35rem;font-weight:900;color:#7ee787;'>Key Insights</h3>
                <div style='font-size:1.11rem;color:#e5e5e5;line-height:1.7;'>
                    {trends}
                </div>
            </div>
            <div style='flex:1 1 320px; min-width:320px; max-width:420px; background: linear-gradient(135deg,#232526 0%,#2b5876 100%); border-radius:32px; box-shadow:0 8px 32px rgba(60,120,60,0.13); border:2px solid #2d2d2d; padding:32px 28px 28px 28px;'>
                <div style='font-size:2.2rem; margin-bottom:10px;'>💡</div>
                <h3 style='margin-bottom:12px;font-size:1.35rem;font-weight:900;color:#ffd700;'>Business Opportunities</h3>
                <div style='font-size:1.11rem;color:#f3f3f3;line-height:1.7;'>
                    {bizop}
                </div>
            </div>
            <div style='flex:1 1 320px; min-width:320px; max-width:420px; background: linear-gradient(135deg,#232526 0%,#ff512f 100%); border-radius:32px; box-shadow:0 8px 32px rgba(120,60,60,0.13); border:2px solid #2d2d2d; padding:32px 28px 28px 28px;'>
                <div style='font-size:2.2rem; margin-bottom:10px;'>⚠️</div>
                <h3 style='margin-bottom:12px;font-size:1.35rem;font-weight:900;color:#ffb4a2;'>Risks & Challenges</h3>
                <div style='font-size:1.11rem;color:#f3f3f3;line-height:1.7;'>
                    {risks}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if references.strip():
        with st.expander("📚 Sources"):
            st.markdown(references, unsafe_allow_html=True)
