import os
import re
import json
import argparse
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import html
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparse

"""
Blog Scraper (RSS/Atom) — outputs aligned with reddit_scraper format.

Features:
- Incremental mode with --since ISO-8601 or --days fallback
- Appends to data/blogs.json with dedupe
- Stores last_run_iso in data/state_blogs.json
- Protein/gym/diet focused keywords
- JSON schema matches reddit_scraper style
"""

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "blogs.json"
STATE_FILE = DATA_DIR / "state_blogs.json"

# ---------- Defaults ----------
SITES = [
    "https://barbend.com/feed/",
    "https://www.precisionnutrition.com/blog/feed",
    "https://www.eatthis.com/feed/",
    "https://blog.myfitnesspal.com/feed/",
    "https://nutrition.org/feed/",
    "https://www.menshealth.com/rss/all.xml/",
    "https://www.bodybuilding.com/rss/articles",
    "https://www.muscleandfitness.com/feed/",
]

DEFAULT_DAYS = int(os.getenv("BLOG_DAYS_BACK", "90"))
DEFAULT_TOTAL_CAP = int(os.getenv("BLOG_TOTAL_CAP", "300"))
DEFAULT_PER_DOMAIN_CAP = int(os.getenv("BLOG_PER_DOMAIN_CAP", "50"))

KEYWORDS = [
    "protein","whey","casein","pea protein","soy protein","plant protein",
    "collagen","amino acid","amino acids","bcaa","eaa","leucine",
    "high-protein","protein powder","protein shake","protein bar",
    "muscle","gym","fitness","workout","training","diet","nutrition","supplement"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NutritionBot/1.3; +https://example.com/bot)"}

# ---------- Utilities ----------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_json(path: Path, arr: list):
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(obj: dict):
    save_json(STATE_FILE, obj)

def parse_since(since: str | None, days: int) -> datetime:
    if since:
        try:
            s = since.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc) - timedelta(days=days)

def parse_dt(dt_str):
    if not dt_str:
        return None
    try:
        dt = dateparse.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def contains_keyword(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)

def dedupe_merge(existing: list, new_items: list) -> list:
    seen = set()
    out = []
    for it in existing + new_items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
    return out

# ---------- Feed Parsing ----------
def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag

def _find_first_text(elem: ET.Element, names: list[str]) -> str | None:
    for n in names:
        for child in elem.iter():
            if _strip_ns(child.tag) == n:
                if child.text and child.text.strip():
                    return child.text.strip()
    return None

def parse_feed_bytes(content: bytes) -> dict:
    entries = []
    try:
        root = ET.fromstring(content)
    except Exception:
        return {"entries": []}

    root_name = _strip_ns(root.tag).lower()

    if root_name == "rss":
        channel = next((c for c in root if _strip_ns(c.tag) == "channel"), None)
        candidates = [it for it in channel if _strip_ns(it.tag) == "item"] if channel else []
        for item in candidates:
            entries.append({
                "title": _find_first_text(item, ["title"]) or "",
                "link": _find_first_text(item, ["link"]) or "",
                "summary": _find_first_text(item, ["description"]) or "",
                "published": _find_first_text(item, ["pubDate"]),
            })
    elif root_name == "feed":
        for entry in [e for e in root if _strip_ns(e.tag) == "entry"]:
            link_url = ""
            for ln in entry.iter():
                if _strip_ns(ln.tag) == "link":
                    href = ln.attrib.get("href", "")
                    if href:
                        link_url = href
                        break
            entries.append({
                "title": _find_first_text(entry, ["title"]) or "",
                "link": link_url,
                "summary": _find_first_text(entry, ["summary", "content"]) or "",
                "published": _find_first_text(entry, ["published", "updated"]),
            })
    return {"entries": entries}

def fetch_feed(url: str) -> dict:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return parse_feed_bytes(r.content)
    except Exception as e:
        print(f"[FEEDERR] {url} → {e}")
        return {"entries": []}

# ---------- Article Extraction ----------
def extract_article_content(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            tag.extract()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        if len(text) > 300:
            return text
        all_text = soup.get_text(" ", strip=True)
        if len(all_text) > 300:
            return all_text
    except Exception:
        pass
    return None

# ---------- Core Processing ----------
def process_feed(feed_url: str, bucket: list, cutoff_dt: datetime, total_cap: int, per_domain_cap: int, per_domain_counts: dict):
    fp = fetch_feed(feed_url)
    entries = fp.get("entries", [])
    print(f"[FEED] {feed_url} → {len(entries)} entries")

    for e in entries:
        if len(bucket) >= total_cap:
            break
        raw_link = (e.get("link") or "").strip()
        if not raw_link:
            continue
        domain = urlparse(raw_link).netloc
        title = (e.get("title") or "").strip()
        summary = (e.get("summary") or "").strip()
        pub_dt = parse_dt(e.get("published"))

        if pub_dt and pub_dt < cutoff_dt:
            continue

        if not (contains_keyword(title) or contains_keyword(summary)):
            continue

        per_domain_counts.setdefault(domain, 0)
        if per_domain_counts[domain] >= per_domain_cap:
            continue

        article_content = extract_article_content(raw_link)
        combined_text = f"{title}\n\n{summary}"
        if article_content and contains_keyword(article_content):
            combined_text += f"\n\n{article_content}"

        item = {
            "title": title,
            "url": raw_link,
            "domain": domain,
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "date_status": "known" if pub_dt else "unknown",
            "combined_text": combined_text.strip(),
            "source": "blogs",
            "source_type": "blog_article",
            "is_verified": False,
        }

        bucket.append(item)
        per_domain_counts[domain] += 1

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="ISO-8601 UTC cutoff")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Window (days)")
    ap.add_argument("--total-cap", type=int, default=DEFAULT_TOTAL_CAP, help="Global cap")
    ap.add_argument("--per-domain-cap", type=int, default=DEFAULT_PER_DOMAIN_CAP, help="Per-domain cap")
    ap.add_argument("--sites", nargs="*", default=SITES, help="Override sites")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    since_iso = args.since or state.get("last_run_iso")
    cutoff_dt = parse_since(since_iso, args.days)

    bucket = []
    per_domain_counts = {}
    for feed_url in args.sites:
        if len(bucket) >= args.total_cap:
            break
        process_feed(feed_url, bucket, cutoff_dt, args.total_cap, args.per_domain_cap, per_domain_counts)

    existing = load_json(OUTPUT_FILE)
    merged = dedupe_merge(existing, bucket)
    save_json(OUTPUT_FILE, merged)
    print(f"✅ Blogs: +{len(bucket)} new | total={len(merged)} → {OUTPUT_FILE}")

    save_state({"last_run_iso": iso_now(), "added": len(bucket), "total": len(merged)})

if __name__ == "__main__":
    main()
