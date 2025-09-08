import faiss
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FAISS_INDEX = DATA_DIR / "faiss.index"
INDEX_META = DATA_DIR / "index_meta.jsonl"
INDEX_INFO = DATA_DIR / "index_info.json"

def load_faiss_index():
    if not FAISS_INDEX.exists() or not INDEX_META.exists():
        raise RuntimeError("Missing FAISS files; run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX))
    meta = []
    for line in INDEX_META.read_text(encoding="utf-8").splitlines():
        if line.strip():
            meta.append(json.loads(line))
    return index, meta

from sentence_transformers import SentenceTransformer

# Cache the model to avoid reloading on every call
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def embed_query(q: str, dim: int = None) -> np.ndarray:
    model = get_model()
    emb = model.encode([q], normalize_embeddings=True)
    return np.asarray(emb).astype("float32")

def faiss_topk(query: str, k: int = 8):
    index, meta = load_faiss_index()
    qvec = embed_query(query, dim=index.d)
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
