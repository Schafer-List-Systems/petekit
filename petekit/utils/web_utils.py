"""WebUtils - shared helpers for web scraping and rendering."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from bs4 import BeautifulSoup


def _find_chrome() -> str:
    for binary in ("google-chrome", "chromium-browser", "chromium"):
        try:
            result = subprocess.run(
                ("which", binary),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split()[0]
        except Exception:
            continue
    return "google-chrome"


_CHROME_BIN = _find_chrome()


def _fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch raw HTML from a URL using httpx."""
    import httpx
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"Error fetching URL: {type(e).__name__}: {e}"


def _render_chrome(url: str, timeout: int = 30) -> str:
    """Render a URL via headless Chrome and return the DOM output."""
    try:
        proc = subprocess.run(
            (
                _CHROME_BIN,
                "--headless=new",
                "--no-sandbox",
                "--dump-dom",
                "--disable-gpu",
                f"--timeout-ms={timeout * 1000}",
                url,
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            error_detail = (proc.stderr or "non-zero exit code").strip()
            return f"Error rendering page: Chrome exited with code {proc.returncode}. {error_detail}"
        return proc.stdout
    except subprocess.TimeoutExpired:
        return f"Error: Page render timed out after {timeout} seconds."
    except Exception as e:
        return f"Error rendering page: {type(e).__name__}: {e}"


def _format_for_buffer(content: str) -> str:
    """Format content for storage in a buffer.

    - Tries to reformat JSON via json.loads + dumps(indent=2).
    - Tries to reformat HTML/XML via BeautifulSoup prettify.
    - Falls back to the original string if neither applies.
    """
    stripped = content.strip()
    if not stripped:
        return content

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, indent=2)
        except Exception:
            pass

    try:
        soup = BeautifulSoup(content, "lxml")
        return soup.prettify()
    except Exception:
        pass

    return content


def _format_scrape_output(records: list[dict]) -> str:
    """Format a list of dicts as a line-by-line JSON array."""
    if not records:
        return "[]"
    json_lines = [json.dumps(r) for r in records]
    return "[\n" + ",\n".join(json_lines) + "\n]"


def _web_scrape(html: str, tag: str, attr: str | list[str] | None = None) -> list[dict]:
    """Extract structured data from HTML using a tag name and optional attribute specifier.

    Args:
        html: raw HTML string
        tag: HTML tag name to match (e.g. "a", "img", "div")
        attr: None -> extract text content of each element as {"text": ...}
              str -> extract that attribute as {attr: value}
              list[str] -> extract multiple attributes as {a: v1, b: v2, ...}

    Returns:
        List of dicts, one per matched tag.
    """
    soup = BeautifulSoup(html, "lxml")
    results = []
    for elem in soup.find_all(tag):
        if attr is None:
            results.append({"text": elem.get_text(strip=True)})
        elif isinstance(attr, list):
            results.append({a: elem.get(a, "") for a in attr})
        else:
            results.append({attr: elem.get(attr, "")})
    return results
