# nutrition_insights/scripts/build_corpus_jsonl.py
"""
Script to build corpus_filtered.jsonl from combined.json (blogs, reddit, journals merged).
Each line in corpus_filtered.jsonl is a JSON object with required fields for RAG.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COMBINED_FILE = DATA_DIR / "combined.json"
CORPUS_FILE = DATA_DIR / "corpus_filtered.jsonl"

# Load combined.json
with COMBINED_FILE.open("r", encoding="utf-8") as f:
    items = json.load(f)

# Helper: minimal field mapping/normalization

def normalize_entry(entry):
    # Ensure required fields for RAG
    out = {}
    out["id"] = entry.get("id") or entry.get("url") or entry.get("permalink")
    out["title"] = entry.get("title", "")
    out["url"] = entry.get("url") or entry.get("permalink")
    out["source"] = entry.get("source", "unknown")
    out["source_type"] = entry.get("source_type", "unknown")
    out["is_verified"] = entry.get("is_verified", False)
    out["date"] = entry.get("date") or entry.get("published_at") or entry.get("created_utc")
    # Use 'text' if present, else 'combined_text', else ''
    out["text"] = entry.get("text") or entry.get("combined_text") or entry.get("summary") or ""
    # Add any other fields as needed
    return out

with CORPUS_FILE.open("w", encoding="utf-8") as f:
    for entry in items:
        norm = normalize_entry(entry)
        f.write(json.dumps(norm, ensure_ascii=False) + "\n")

print(f"✅ Built {CORPUS_FILE} with {len(items)} entries from {COMBINED_FILE}")
