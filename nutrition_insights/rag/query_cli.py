# nutrition_insights/rag/query_cli.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# ---------------- Path setup so imports work no matter how we run this ----------------
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]          # .../nutrition_insights
REPO_ROOT = PKG_ROOT.parent         # repo root

# Ensure packages exist for imports like nutrition_insights.*
for d in (PKG_ROOT, PKG_ROOT / "rag"):
    initf = d / "__init__.py"
    if not initf.exists():
        try:
            initf.write_text("", encoding="utf-8")
        except Exception:
            pass

# Make repo root & package root importable
for p in (str(REPO_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------- Robust import of llm_connection ----------------
class LLMError(Exception):
    pass

def _import_llm():
    # 1) preferred: nutrition_insights.rag.llm_connection
    try:
        from nutrition_insights.rag.llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass
    # 2) common: nutrition_insights/llm_connection.py
    try:
        from nutrition_insights.llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass
    # 3) same-directory fallback
    try:
        from llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass

    def _stub(*_, **__):
        raise LLMError(
            "Could not import llm_connection. Place it at either:\n"
            " - nutrition_insights/llm_connection.py, or\n"
            " - nutrition_insights/rag/llm_connection.py\n"
            "and ensure __init__.py files exist."
        )
    return _stub, LLMError

get_chat_fn, LLMError = _import_llm()

# ---------------- Paths & data ----------------
DATA = PKG_ROOT / "data"
FAISS_INDEX = DATA / "faiss.index"
INDEX_META  = DATA / "index_meta.jsonl"
INDEX_INFO  = DATA / "index_info.json"
FILTERED    = DATA / "corpus_filtered.jsonl"

# ---------------- Retrieval (FAISS) ----------------
import numpy as np
try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("FAISS is required. Install with: pip install faiss-cpu") from e

def load_index():
    if not FAISS_INDEX.exists() or not INDEX_META.exists():
        raise SystemExit("Missing FAISS files; run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX))
    import json
    def _make_hashable(val):
        if isinstance(val, (list, dict, set)):
            return json.dumps(val, ensure_ascii=False)
        return val
    meta = []
    for line in INDEX_META.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            # Ensure all values in dict are hashable
            item_hashable = {k: _make_hashable(v) for k, v in item.items()}
            meta.append(item_hashable)
    return index, meta

# NOTE: build_index uses an embedding function; at query time we mirror dim with a stable hash.
from sentence_transformers import SentenceTransformer
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def embed_query(q: str, dim: int = 768) -> np.ndarray:
    model = get_model()
    emb = model.encode([q], normalize_embeddings=True)
    return np.asarray(emb).astype("float32")

def topk(index, meta, qvec, k: int):
    D, I = index.search(qvec.astype("float32"), k)
    hits = []
    for rank, (d, i) in enumerate(zip(D[0], I[0])):
        if i < 0 or i >= len(meta):
            continue
        item = meta[i].copy()
        item["_rank"] = rank
        item["_score"] = float(d)
        hits.append(item)
    return hits

# ---------------- RAG assembly & guardrails ----------------
SYSTEM = (
    "You are ProteinScope, a nutrition domain assistant. Answer only about dietary protein, "
    "protein supplements, amino acids, protein timing, dose, quality (DIAAS/PDCAAS), safety, "
    "performance,timings,comparision of proteins, recovery, satiety, body composition, clinical use (e.g., sarcopenia), and related topics. "
    "If a question is out of scope (e.g., fashion, stocks), say it's out of scope."
)

def format_context(snippets: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(snippets, 1):
        src = s.get("source", "unknown")
        dt = s.get("date", "")
        title = s.get("title", "") or ""
        text = s.get("excerpt", "") or s.get("text", "") or s.get("content", "") or ""
        blocks.append(f"[{i}] ({src} | {dt}) {title}\n{text}")
    return "\n\n".join(blocks)


# Use robust scope/context logic from phase3
from nutrition_insights.phase3.services.query_router import is_in_scope, build_context_snippets

def main():
    ap = argparse.ArgumentParser(description="Query the FAISS index with guardrails + Ollama")
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=10)
    ap.add_argument("--verified-boost", type=float, default=0.10)
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    ap.add_argument("--temp", type=float, default=0.2)

    # Scope options (opt‑in)
    ap.add_argument("--strict", action="store_true",
                    help="If set, reject queries outside protein scope.")
    ap.add_argument("--scope-thresh", type=float, default=0.35,
                    help="Similarity/score threshold used only when --strict is on.")
    ap.add_argument("--debug-scope", action="store_true",
                    help="Print scope diagnostics when --strict is on.")

    # Minimal evidence gates
    ap.add_argument("--min-support", type=int, default=1,
                    help="Minimum retrieved hits required to attempt an answer.")
    ap.add_argument("--min-topscore", type=float, default=0.0,
                    help="If >0, require the top hit score to exceed this to answer.")

    args = ap.parse_args()

    # Normalize question early and use consistently
    question = " ".join(args.question.strip().split())

    # Load index & retrieve first (scope can use hits as a signal)
    index, meta = load_index()
    qvec = embed_query(question, dim=index.d)
    hits = topk(index, meta, qvec, args.k)

    # Optional strict scope guard
    if args.strict:
        if not is_in_scope(question):
            print("This question looks outside the protein/nutrition scope. "
                  "Please ask about dietary protein, supplements, amino acids, "
                  "timing, dose, quality, safety, or performance.")
            sys.exit(0)

    # Basic support checks
    if len(hits) < args.min_support:
        print("Not enough supporting documents in the index to answer confidently.")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h.get('title','')[:80]} | {h.get('url','')}")
        sys.exit(0)
    if args.min_topscore > 0 and (hits[0].get("_score", 0.0) or 0.0) < args.min_topscore:
        print(f"Top hit score {hits[0].get('_score', 0.0):.3f} is below --min-topscore={args.min_topscore}.")
        sys.exit(0)

    # Prepare context for LLM
    # Use only top N hits for context, and format as in the working chatbot
    chosen = hits[:8]
    ctx_text = format_context(chosen)

    # Compose prompt as in the working chatbot
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Context snippets (use these; cite inline like [1], [2] where helpful):\n"
        f"{ctx_text}\n\n"
        "Answer concisely for a business/product audience. If insufficient context or off-topic, reply exactly with:\n"
        "Out of scope — this question is not related to protein or the available data."
    )

    chat = get_chat_fn(model=args.model, temperature=args.temp)
    try:
        # FIX: call signature is (system, prompt)
        answer = chat(SYSTEM, user_prompt).strip()
        print(answer)
    except LLMError as e:
        print("[No LLM configured]", str(e))
        # Even without an LLM, print the sources to help debugging
        verified = [c for c in chosen if c.get("is_verified")]
        community = [c for c in chosen if not c.get("is_verified")]
        if verified:
            print("\nVerified Sources:")
            for i, c in enumerate(verified, 1):
                print(f"[{i}] {c.get('title','')[:80]} | {c.get('published','')} | {c.get('url','')}")
        if community:
            print("\nCommunity Sources:")
            for i, c in enumerate(community, 1):
                print(f"[{i}] {c.get('title','')[:80]} | {c.get('published','')} | {c.get('url','')}")
        print("\n\nCitations:")
        for c in verified + community:
            u = c.get("url") or c.get("link") or ""
            if u:
                print(u)
        sys.exit(0)

if __name__ == "__main__":
    main()