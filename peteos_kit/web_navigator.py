"""WebNavigator - agentic object for web fetching, rendering, and screenshots."""

from __future__ import annotations

import subprocess
import tempfile
import time as time_mod
from pathlib import Path

from bs4 import BeautifulSoup

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .buffer_manager import Buffer, BufferManager


def _find_chrome() -> str:
    for binary in ("google-chrome", "chromium-browser", "chromium"):
        try:
            result = subprocess.run(
                ("which", binary),
                capture_output=True, text=True, timeout=5,
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
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"},
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
            import json
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


def _extract_links(html: str) -> list[dict]:
    """Extract all links from HTML. Returns list of dicts with title and href."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        title = (a.get_text(strip=True) or a.get("title", "") or a["href"])[:100]
        links.append({"title": title, "href": a["href"]})
    return links


def _extract_images(html: str) -> list[str]:
    """Extract all image src URLs from HTML."""
    soup = BeautifulSoup(html, "lxml")
    return [img.get("src", "") or img.get("data-src", "") or "" for img in soup.find_all("img", src=True)]


def _extract_by_selector(html: str, selector: str) -> list[str]:
    """Extract text content from HTML elements matching the CSS selector."""
    soup = BeautifulSoup(html, "lxml")
    results = []
    try:
        for elem in soup.select(selector):
            text = elem.get_text(separator="\n", strip=True)
            if text:
                results.append(text)
    except Exception:
        return []
    return results


class WebNavigator(BufferManager, AgenticObject):
    """You are a web navigator. You fetch and render web pages into memory buffers.

    Workflow:
      1. load_url(url) — fetch a page via HTTP into a buffer (keyed by URL)
      2. render_url(url) — fetch via headless Chrome into a buffer (keyed by URL)
      3. extract_links(url) / extract_images(url) / extract_by_selector(url, selector) — carve derived buffers
      4. grep_buffer(url, "regex") + read_buffer(url, start=N, end=M) — search and read

    Use list_buffers to see loaded URLs and their derived buffers.
    Use (?i) at the start of a grep pattern for case-insensitive matching.
    """

    @property
    def temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="web_navigator_"))
        return self._temp_dir

    @tool(description="Fetch a URL and store its extracted text content in a buffer keyed by the URL.")
    def load_url(self, url: str, timeout: int = 15) -> str:
        """Fetch a URL via HTTP, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        formatted = _format_for_buffer(html)
        lines = formatted.splitlines()
        now = time_mod.time()
        self._buffers[url] = Buffer(lines=lines, created_at=now, modified_at=now)
        return f"Loaded buffer '{url}' ({len(lines)} lines)."

    @tool(description="Render a URL via headless Chrome and store extracted text in a buffer keyed by the URL.")
    def render_url(self, url: str, timeout: int = 30) -> str:
        """Render a URL via headless Chrome, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _render_chrome(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        formatted = _format_for_buffer(html)
        lines = formatted.splitlines()
        now = time_mod.time()
        self._buffers[url] = Buffer(lines=lines, created_at=now, modified_at=now)
        return f"Rendered buffer '{url}' ({len(lines)} lines)."

    @tool(description="Extract all links from a URL and store them as lines in a derived buffer (keyed by URL+:links).")
    def extract_links(self, url: str, timeout: int = 15) -> str:
        """Fetch a URL, extract links, store in a derived buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        links = _extract_links(html)
        if not links:
            return f"No links found in '{url}'."
        lines = [f"{link['title']}\t{link['href']}" for link in links]
        key = f"{url}:links"
        now = time_mod.time()
        self._buffers[key] = Buffer(lines=lines, created_at=now, modified_at=now)
        return f"Extracted {len(links)} link(s) into buffer '{key}'."

    @tool(description="Extract all image src URLs from a URL and store them as lines in a derived buffer (keyed by URL+:images).")
    def extract_images(self, url: str, timeout: int = 15) -> str:
        """Fetch a URL, extract image sources, store in a derived buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        images = _extract_images(html)
        if not images:
            return f"No images found in '{url}'."
        key = f"{url}:images"
        now = time_mod.time()
        self._buffers[key] = Buffer(lines=images, created_at=now, modified_at=now)
        return f"Extracted {len(images)} image(s) into buffer '{key}'."

    @tool(description="Extract text content of HTML elements matching the CSS selector. Stores each match as a line in a derived buffer (keyed by URL+:selector:{selector}).")
    def extract_by_selector(self, url: str, selector: str, timeout: int = 15) -> str:
        """Fetch a URL, extract elements matching CSS selector, store in a derived buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        if not selector.strip():
            return "Error: selector must not be empty."
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        results = _extract_by_selector(html, selector)
        if not results:
            return f"No elements matching selector '{selector}' found in '{url}'."
        key = f"{url}:selector:{selector}"
        now = time_mod.time()
        self._buffers[key] = Buffer(lines=results, created_at=now, modified_at=now)
        return f"Extracted {len(results)} element(s) into buffer '{key}'."

    @tool
    def take_screenshot(self, url: str, output_file: str | None = None, timeout: int = 30) -> str:
        """Capture a screenshot of a web page using headless Chrome."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        if output_file is None:
            output_path = self.temp_dir / f"Screenshot_{Path(url).name}.png"
        else:
            output_path = Path(output_file).resolve()
            try:
                output_path.relative_to(self.temp_dir.resolve() if output_file.startswith("..") else output_path.parent)
            except ValueError:
                return f"Error: Output path '{output_file}' is outside the allowed workspace."

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                (
                    _CHROME_BIN,
                    "--headless=new",
                    "--no-sandbox",
                    f"--screenshot={output_path}",
                    "--window-size=1280,800",
                    "--disable-gpu",
                    f"--timeout-ms={timeout * 1000}",
                    url,
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0 or not output_path.is_file():
                error_detail = (proc.stderr or "Chrome exited unexpectedly").strip()
                return f"Error capturing screenshot: {error_detail}"
            return f"Screenshot saved to: {output_path}"
        except subprocess.TimeoutExpired:
            return f"Error: Screenshot timed out after {timeout} seconds."
        except Exception as e:
            return f"Error capturing screenshot: {type(e).__name__}: {e}"


