"""
Low-level Federal Reserve wrapper: monetary-policy press releases and FOMC
statement text. No caching, no DB writes.

Source is the Fed's public RSS feed, which only holds the most recent ~15
releases (about six months). Older windows are reported as uncovered by the
caller rather than silently returning nothing.
"""

import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

import config
from resilience import retry_call

FEED_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
HEADERS = {"User-Agent": "economic-pulse-agent/0.1"}


def _get(url: str) -> bytes:
    def attempt(timeout: float):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            raise ConnectionError(f"Federal Reserve request failed ({url}): {exc}") from exc
        if response.status_code != 200:
            raise ConnectionError(f"Federal Reserve returned HTTP {response.status_code} for {url}")
        return response

    response = retry_call(attempt, source="fed")
    # Raw bytes: the feed starts with a UTF-8 byte-order mark that
    # requests' guessed text encoding turns into garbage.
    return response.content


def fetch_fomc_press_releases() -> list[dict]:
    """
    Recent monetary-policy press releases, oldest first.

    Returns [{"date": "2026-09-16", "title": "...", "url": "https://..."}, ...].
    Raises ConnectionError on network/HTTP failure, ValueError if the feed
    can't be parsed.
    """
    try:
        root = ET.fromstring(_get(FEED_URL))
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse Fed RSS feed: {exc}") from exc

    releases = []
    for item in root.iter("item"):
        pub = item.findtext("pubDate", "").strip()
        releases.append({
            "date": parsedate_to_datetime(pub).date().isoformat(),
            "title": item.findtext("title", "").strip(),
            "url": item.findtext("link", "").strip(),
        })
    return sorted(releases, key=lambda r: r["date"])


def fetch_statement_text(url: str, max_chars: int = 1500) -> str:
    """
    Opening paragraphs of an FOMC statement (the policy decision and the
    Committee's read of the economy), stopping before the voting record.
    """
    page = _get(url).decode("utf-8", errors="replace")
    start = page.find('id="article"')
    body = page[start:] if start != -1 else page
    paragraphs = []
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())
        if len(text) < 40 or text.startswith("For release"):
            continue
        if text.startswith("Voting for") or text.startswith("For media inquiries"):
            break
        paragraphs.append(text)
    if not paragraphs:
        raise ValueError(f"No statement text found at {url}")
    excerpt = "\n".join(paragraphs)
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rsplit(" ", 1)[0] + " ..."
