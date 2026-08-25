# uznews — Uzbek news topic classification

A five-way topic classifier for Uzbek-language news articles, built on a
dataset scraped and labelled from scratch. The headline result is that a
TF-IDF baseline outperforms a fine-tuned XLM-RoBERTa, and that both models
fail on the same articles for the same reason.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + logistic regression | **0.786** | **0.781** |
| XLM-RoBERTa, raw text | 0.768 | 0.762 |
| XLM-RoBERTa, normalized text | 0.782 | 0.779 |

Test set: 271 articles, balanced across five classes.

## The interesting part: both models fail identically

Per-class accuracy on the test set:

| Class | Baseline | XLM-R (norm) |
|---|---|---|
| sport | 54/55 | 54/55 |
| jahon | 50/54 | 50/54 |
| iqtisodiyot | 42/54 | 43/54 |
| jamiyat | 38/54 | 37/54 |
| uzbekiston | 29/54 | 28/54 |

`sport` is effectively solved by both. `uzbekiston` fails for both, and the
errors concentrate in the same place: 74% of baseline errors and 83% of
transformer errors fall inside the `uzbekiston` / `jamiyat` / `iqtisodiyot`
triangle.

This is a label problem, not a model problem. Kun.uz mixes two axes in its
taxonomy — geography (`uzbekiston`, `jahon`) and topic (`sport`,
`iqtisodiyot`). A domestic economics story legitimately belongs to both
`uzbekiston` and `iqtisodiyot`, and the choice between them is an editorial
one. `jamiyat` ("society") acts as a catch-all that absorbs the remainder.

Two architecturally unrelated models converging on the same accuracy and the
same error structure is evidence that ~78% is close to the ceiling this
taxonomy allows, not the ceiling these models impose.

## Why the baseline wins

Topic classification from full articles is close to a keyword problem —
sports vocabulary is unmistakable, and TF-IDF with bigrams captures it
directly. With 2,164 training examples, a 278M-parameter model has little
room to demonstrate any advantage in contextual understanding, while also
having to learn a classification head from random initialization.

Truncation was tested as an alternative explanation: raising `MAX_LEN` from
256 to 512 and adding a fourth epoch changed accuracy by less than half a
point. The result is not an artifact of the transformer seeing less text.

The API therefore serves the baseline. It is more accurate, loads in
milliseconds, and needs no GPU.

## Apostrophe normalization ablation

Uzbek Latin script uses `oʻ` (U+02BB) and `ʼ` (U+02BC), but kun.uz writes
both with typographic quotes (U+2018 / U+2019). Normalization affects
200/200 sampled training documents.

Effect on accuracy: **+1.5 points** (four articles out of 271). A run at
256 tokens gave +0.7 points — same direction, similar magnitude.

Both deltas are within noise for a test set this size. The honest conclusion
is that this experiment cannot distinguish a small positive effect from zero;
establishing the difference would require multiple random seeds.

Building this exposed a gap in [uznorm](https://github.com/Ikbola/uznorm):
it handled U+2019 but not U+2018, so it silently normalized only the rarer
of the two characters. Found by testing against real corpus text rather than
hand-written fixtures.

## Dataset

Scraped from kun.uz: sitemap → article URLs → per-article fetch. Labels come
from each page's schema.org `BreadcrumbList`, which names the section the
publisher filed the article under. Language is filtered by URL prefix (the
site publishes Latin Uzbek, Cyrillic Uzbek, Russian, and English versions of
the same articles).

- 11,220 Latin Uzbek articles from 2026
- Filtered: extraction failures, articles under 400 characters, duplicates
  by body fingerprint and by title, residual Cyrillic
- Balanced to 541 per class, split 80/10/10, stratified
- **2,164 train / 270 val / 271 test**

Crawl policy: 2s delay, identifying user-agent, robots.txt checked. See
[docs/sources.md](docs/sources.md).

**Article text is not redistributed.** Kun.uz requires written permission to
reuse its material. `data/` is gitignored; the scraper is published so the
corpus can be rebuilt.

## Limitations

- Single source. The model partly learns kun.uz house style; accuracy will
  not transfer cleanly to other Uzbek publications.
- Label set follows the publisher's taxonomy, including its geography/topic
  overlap.
- Balanced training set; real-world class distribution is heavily skewed
  (`jahon` alone is 37% of published articles).
- Normalization is context-based (apostrophe after `o`/`g` → okina) and
  misfires on foreign possessives: `Bloomberg's` → `Bloombergʻs`.
- Single seed per configuration. Differences under ~1.5 points are not
  resolvable at this test set size.

## Usage

```bash
pip install -e ".[dev]"
python scripts/scrape.py --year 2026       
python scripts/build_dataset.py
python scripts/baseline.py
uvicorn uznews.api:app --reload
```

`POST /classify` with `{"text": "..."}` returns a predicted label, its
confidence, and scores across all five classes. Full probabilities are
returned deliberately: given the label overlap above, a split between
`uzbekiston` and `iqtisodiyot` is information a bare label would hide.

Fine-tuning runs in [notebooks/](notebooks/) on a free Colab GPU.

## Repository

```
scripts/     scrape.py, build_dataset.py, baseline.py
src/uznews/  api.py
tests/       API contract tests
notebooks/   XLM-RoBERTa fine-tuning and ablation
results/     metrics, confusion matrices, saved baseline model
docs/        sources.md — crawl policy and site structure
```