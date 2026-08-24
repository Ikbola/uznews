"""Turn scraped articles into balanced train/val/test splits.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/kun_2026.jsonl")
OUT_DIR = Path("data/processed")

LABELS = ["jahon", "uzbekiston", "jamiyat", "iqtisodiyot", "sport"]

MIN_CHARS = 400
SEED = 42
TEST_FRAC = 0.10
VAL_FRAC = 0.10

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def text_fingerprint(text: str) -> str:
    """Hash of collapsed lowercase text, for near-duplicate detection."""
    collapsed = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / len(text)


def main() -> None:
    rows = [json.loads(line) for line in RAW_PATH.open(encoding="utf-8")]
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} articles")

    def drop(mask, reason):
        nonlocal df
        n = mask.sum()
        df = df[~mask].copy()
        print(f"  dropped {n:5} — {reason}")

    drop(~df["category"].isin(LABELS), "category not in label set")
    drop(df["selector"].eq("fallback:all-<p>"), "body selector fell back")
    drop(df["chars"] < MIN_CHARS, f"shorter than {MIN_CHARS} chars")
    drop(df["title"].isna(), "no title")
    drop(df["text"].map(cyrillic_ratio) > 0.10, "mostly Cyrillic")

    df["fp"] = df["text"].map(text_fingerprint)
    drop(df.duplicated("fp"), "duplicate body text")
    drop(df.duplicated("title"), "duplicate title")

    print(f"\n{len(df)} articles remain")
    print("by class:", dict(Counter(df["category"])))

    n_per_class = df["category"].value_counts().min()
    print(f"\nbalancing to {n_per_class} per class")

    df = (
        pd.concat([
            group.sample(n=n_per_class, random_state=SEED)
            for _, group in df.groupby("category")
        ])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    train, held = train_test_split(
        df, test_size=TEST_FRAC + VAL_FRAC,
        stratify=df["category"], random_state=SEED,
    )
    val, test = train_test_split(
        held, test_size=TEST_FRAC / (TEST_FRAC + VAL_FRAC),
        stratify=held["category"], random_state=SEED,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["url", "title", "text", "category"]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        part[cols].to_json(path, orient="records", lines=True, force_ascii=False)
        counts = dict(sorted(Counter(part["category"]).items()))
        print(f"{name:6} {len(part):5}  {counts}")

    print(f"\nwritten to {OUT_DIR}/")


if __name__ == "__main__":
    main()