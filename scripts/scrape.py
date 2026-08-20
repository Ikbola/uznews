"""Scrape kun.uz articles into JSONL for the uznews dataset.

Usage:
    python scripts/scrape.py --year 2026 --limit 200 --sample   
    python scripts/scrape.py --year 2026                      
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "uznews-research-bot/0.1 (+https://github.com/<your-username>/uznews)"
DELAY_SECONDS = 2.0
SITEMAP_DELAY = 1.0

OUT_DIR = Path("data/raw")
SITEMAP_CACHE = Path("data/sitemaps")
SITEMAP_URL = "https://kun.uz/sitemap/site-map-news_year_{year}.xml"
ARTICLE_RE = re.compile(r"/news/\d{4}/\d{2}/\d{2}/")
LANG_PREFIX_RE = re.compile(r"https?://[^/]+/(?:([a-z]{2})/)?news/")

# Keep everything topical; the final label set is chosen after scraping,
# from the observed distribution rather than assumption.
KEEP_CATEGORIES = None  # None = keep all categories

BODY_SELECTORS = [
    "div[itemprop='articleBody']",
    "article .content",
    "div.article-content",
    "div.news-content",
    "main article",
]

BOILERPLATE_MARKERS = ("KUN.UZ", "Reklama", "tahririyat yozma roziligi")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
)))

# sitemap traversal

def _fetch_xml(url: str) -> bytes:
    """Fetch a sitemap, caching it on disk — these rarely change."""
    SITEMAP_CACHE.mkdir(parents=True, exist_ok=True)
    cached = SITEMAP_CACHE / url.rstrip("/").split("/")[-1]
    if cached.exists():
        return cached.read_bytes()

    resp = SESSION.get(url, timeout=(10, 30))
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    time.sleep(SITEMAP_DELAY)
    return resp.content


def _locs(xml_bytes: bytes) -> tuple[str, list[str]]:
    """Return (root tag name, all <loc> values), ignoring XML namespaces."""
    root = ElementTree.fromstring(xml_bytes)
    return (
        root.tag.split("}")[-1],
        [
            el.text.strip()
            for el in root.iter()
            if el.tag.split("}")[-1] == "loc" and el.text
        ],
    )


def _walk(url: str) -> Iterator[str]:
    """Yield article URLs, descending through nested sitemap indexes."""
    root_tag, locs = _locs(_fetch_xml(url))

    if root_tag == "sitemapindex":
        for child in locs:
            yield from _walk(child)
    else:
        for loc in locs:
            if ARTICLE_RE.search(loc):
                yield loc


def iter_article_urls(year: int) -> Iterator[str]:
    yield from _walk(SITEMAP_URL.format(year=year))


def url_lang(url: str) -> str:
    """Language prefix from the URL; the default site is Latin Uzbek."""
    m = LANG_PREFIX_RE.search(url)
    if not m:
        return "?"
    return m.group(1) or "uz"


# article parsing 

def find_category(soup: BeautifulSoup) -> str | None:
    """Read the category slug from the schema.org BreadcrumbList."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for block in data if isinstance(data, list) else [data]:
            if not isinstance(block, dict):
                continue
            if block.get("@type") != "BreadcrumbList":
                continue
            for item in block.get("itemListElement", []):
                target = item.get("item")
                url = target.get("@id") if isinstance(target, dict) else target
                if url and "/news/category/" in url:
                    return url.rstrip("/").split("/")[-1]
    return None


def clean_paragraphs(node) -> str:
    parts = []
    for p in node.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) < 30:
            continue
        if any(marker in text for marker in BOILERPLATE_MARKERS):
            continue
        parts.append(text)
    return "\n\n".join(parts)


def find_body(soup: BeautifulSoup) -> tuple[str, str]:
    """Return (text, which_selector_worked)."""
    for selector in BODY_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = clean_paragraphs(node)
            if len(text) > 200:
                return text, selector
    body = soup.find("body")
    return (clean_paragraphs(body) if body else ""), "fallback:all-<p>"


def scrape_article(url: str) -> dict:
    resp = SESSION.get(url, timeout=(10, 30))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    title_tag = soup.find("h1")
    text, selector = find_body(soup)

    return {
        "url": url,
        "lang": url_lang(url),
        "title": title_tag.get_text(strip=True) if title_tag else None,
        "category": find_category(soup),
        "text": text,
        "chars": len(text),
        "selector": selector,
    }


# driver 

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", default="uz")
    parser.add_argument("--sample", action="store_true",
                        help="shuffle URLs to sample across the whole year")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"kun_{args.year}.jsonl"
    seen_path = OUT_DIR / f"seen_{args.year}.txt"

    done = set()
    if seen_path.exists():
        with seen_path.open(encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}
        print(f"resuming: {len(done)} already visited")

    all_urls = list(iter_article_urls(args.year))
    print("languages in sitemap:", dict(Counter(url_lang(u) for u in all_urls)))

    pool = [u for u in all_urls if url_lang(u) == args.lang and u not in done]
    print(f"{len(pool)} {args.lang} URLs available")

    if args.sample:
        random.seed(42)
        random.shuffle(pool)

    urls = pool[:args.limit] if args.limit else pool
    print(f"{len(urls)} URLs to fetch")

    kept = 0
    stats: dict = {"selectors": {}, "categories": {}, "failed": 0}

    with out_path.open("a", encoding="utf-8") as fh, \
            seen_path.open("a", encoding="utf-8") as seen_fh:
        for i, url in enumerate(urls, 1):
            try:
                record = scrape_article(url)
            except requests.RequestException as exc:
                print(f"  [{i}] FAILED {url}: {exc}")
                stats["failed"] += 1
                time.sleep(DELAY_SECONDS)
                continue

            seen_fh.write(url + "\n")

            sel = record["selector"]
            cat = record["category"] or "(none)"
            stats["selectors"][sel] = stats["selectors"].get(sel, 0) + 1
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

            if args.limit:
                print(f"  [{i}] {cat:14} {record['chars']:6} chars  via {sel}")

            if record["chars"] > 200 and (
                KEEP_CATEGORIES is None or record["category"] in KEEP_CATEGORIES
            ):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if i % 100 == 0:
                print(f"  {i}/{len(urls)} fetched, {kept} kept")

            time.sleep(DELAY_SECONDS)

    print(f"\nkept {kept} articles -> {out_path}")
    print(f"failed: {stats['failed']}")
    print("selectors:", stats["selectors"])
    print("categories:", dict(sorted(
        stats["categories"].items(), key=lambda kv: -kv[1]
    )))


if __name__ == "__main__":
    main()