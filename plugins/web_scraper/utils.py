import os
import random
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=int(os.getenv("SCRAPER_RETRIES", "3")),
    backoff_factor=float(os.getenv("SCRAPER_BACKOFF_FACTOR", "0.3")),
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# A small pool of User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
]


def scrape_url(url: str) -> dict:
    """
    Scrape the given URL and return a dict with the page title and first paragraph text.
    """
    # Optional random delay to mimic human behavior
    try:
        delay_min = float(os.getenv("SCRAPER_DELAY_MIN", "1"))
        delay_max = float(os.getenv("SCRAPER_DELAY_MAX", "3"))
        time.sleep(random.uniform(delay_min, delay_max))
    except Exception:
        pass

    # Prepare headers with rotating User-Agent
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    # If SCRAPERAPI_KEY is set, route request through ScraperAPI
    api_key = os.getenv("SCRAPERAPI_KEY")
    target_url = f"http://api.scraperapi.com?api_key={api_key}&url={url}" if api_key else url

    timeout = int(os.getenv("SCRAPER_TIMEOUT", "10"))
    response = session.get(target_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    first_p_tag = soup.find("p")
    first_paragraph = first_p_tag.get_text(strip=True) if first_p_tag else ""
    return {"title": title, "first_paragraph": first_paragraph}
