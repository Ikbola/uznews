# Data sources

## Selection

Three Uzbek news sites were surveyed: kun.uz, daryo.uz, gazeta.uz.
kun.uz was selected as the sole source.

**Rationale:** cleanest crawling policy (blanket allow, no wildcard rules
requiring manual interpretation), published sitemap, clear topic
sections. A single source keeps the scraper simple; the tradeoff is
noted under Limitations in the README.

**Not used:**
- daryo.uz — permissive robots.txt, viable, dropped to limit scope.
- gazeta.uz — robots.txt relies on wildcard rules (`Disallow: *?*` with
  `?page=` / `?r=` / `?v=` carve-outs) that Python's
  `urllib.robotparser` silently ignores. Would require hand-implementing
  the query-string rules. Dropped for the same reason.

## kun.uz

**robots.txt** (checked 2026-08-19)

- Blanket `Allow: /` for all user agents; no `Disallow` rules
- No `Crawl-delay` specified
- Sitemap: https://kun.uz/sitemap.xml
- `Clean-param` and `Host` present; Yandex-specific, not applicable

**Terms of use**

The site footer states that copying, distributing, or otherwise using
published materials requires written consent from the editorial board.
No article text is redistributed by this project (see Redistribution
below).

**Crawl policy adopted**

- 2 seconds between requests (no crawl-delay specified; chosen
  conservatively rather than treating silence as permission)
- User agent: `uznews-research-bot/0.1 (+<repo url>)`
- Sample HTML cached locally during parser development so selector
  iteration does not repeatedly hit the live site

**Site structure**

- Latin Uzbek is served from the root domain. `https://kun.uz/uz`
  redirects to `https://kun.uz`. Other language versions are reachable
  via the site's language switcher and are not used here.
- Category pages: `https://kun.uz/news/category/<slug>`
- Article pages: `https://kun.uz/news/YYYY/MM/DD/<article-slug>`
- **The category does not appear in the article URL.** Labels are
  therefore derived from the category listing page an article was
  collected from, not parsed from the URL.
- Confirmed category slugs: `sport`, `jahon`, `jamiyat`, `talim`,
  `moliya`, `avto`, `soglom-hayot`, `kuchmas-mulk`, `ayollar-dunyosi`,
  `turizm`, `biznes`
- Probable but unverified: `uzbekiston`, `iqtisodiyot` (seen as card
  labels, not as links)
- The site is a Next.js application; some page regions render
  client-side. Category listings render server-side and are reachable
  with plain HTTP requests.
- Pagination: TODO — unresolved, see below.

**Label taxonomy decision**

Kun.uz categories mix two axes: topic (Sport, Iqtisodiyot, Ta'lim) and
geography (O'zbekiston, Jahon). Geographic categories are excluded from
the label set, since an article can belong to both a geographic and a
topical category and the overlap would introduce label noise unrelated
to topic.

## Redistribution

Article text is **not** committed to this repository and is not
redistributed. `data/` is excluded via `.gitignore`. The scraper,
parsing code, trained model, and evaluation results are the
deliverables. If a shareable dataset is needed, article URLs and labels
will be released so the corpus can be rebuilt locally.

## Script handling

Uzbek is written in both Latin and Cyrillic. Only Latin-script articles
are collected, since the normalization ablation concerns Latin
apostrophe characters (U+02BB / U+02BC). A script-detection filter is
applied during cleaning to catch Cyrillic or Russian text that reaches
the corpus.