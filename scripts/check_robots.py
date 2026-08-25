"""Checking robots.txt rules for candidate news sites before any scraping."""

import urllib.robotparser as robotparser
from urllib.parse import urljoin

import requests

USER_AGENT = "uznews-research-bot/0.1 (+https://github.com/<Ikbola>/uznews)"

SITES = [
    "https://kun.uz",
    "https://daryo.uz",
    "https://www.gazeta.uz",
]

SAMPLE_PATHS = ["/", "/uz/", "/uz/sport"]


def check(site: str) -> None:
    robots_url = urljoin(site, "/robots.txt")
    print(f"\n{'=' * 60}\n{site}\n{'=' * 60}")

    try:
        resp = requests.get(
            robots_url, headers={"User-Agent": USER_AGENT}, timeout=10
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  could not fetch robots.txt: {exc}")
        return

    print("--- raw robots.txt ---")
    print(resp.text.strip()[:2000])
    print("--- parsed verdicts ---")

    rp = robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())

    delay = rp.crawl_delay(USER_AGENT)
    print(f"  crawl-delay: {delay if delay is not None else 'not specified'}")

    for path in SAMPLE_PATHS:
        url = urljoin(site, path)
        verdict = "ALLOW" if rp.can_fetch(USER_AGENT, url) else "BLOCK"
        print(f"  {verdict}  {url}")


if __name__ == "__main__":
    for site in SITES:
        check(site)