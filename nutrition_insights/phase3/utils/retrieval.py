from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from nutrition_insights.phase3.utils.faiss_utils import faiss_topk

@dataclass(frozen=True)
class Snippet:
    title: str
    source: str
    date: Optional[str]
    url: Optional[str]
    excerpt: str
    score: float

def build_context_snippets(
    df: pd.DataFrame,
    query: str,
    *,
    topn: int = None,
    per_source_cap: int = None,
    min_chars: int = 120,
    max_chars: int = 900,
    recency_boost: float = 0.10,  # small boost for newer items
    prefer_journals: bool = False,
    agentic_source: str = None,
) -> List[Snippet]:
    """
    Rank and format context snippets for the LLM.
    Uses FAISS vector search for retrieval.
    Optionally filters by agentic_source: 'journals', 'reddit_blogs', or 'all'.
    """
    hits = faiss_topk(query, k=topn or 8)
    if not hits:
        return []
    # Filter hits by agentic_source if provided
    if agentic_source == 'journals':
        hits = [item for item in hits if (item.get('source') or '').lower() == 'journals']
    elif agentic_source == 'reddit_blogs':
        hits = [item for item in hits if (item.get('source') or '').lower() in ['reddit', 'blogs']]
    # else: use all
    snippets = []
    for item in hits:
        title = item.get("title") or "(untitled)"
        source = item.get("source") or "unknown"
        date = item.get("date")
        url = item.get("url")
        text = item.get("text") or ""
        excerpt = text[:max_chars]
        if len(excerpt) < min_chars:
            excerpt = text[:min_chars]
        snippets.append(
            Snippet(
                title=title,
                source=source,
                date=date,
                url=url,
                excerpt=excerpt,
                score=float(item.get("_score", 0.0)),
            )
        )
    return snippets
