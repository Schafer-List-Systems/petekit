"""WebNavigator - agentic object for web fetching, rendering, and screenshots."""

from __future__ import annotations

import cv2
import json
import subprocess
import tempfile
import time as time_mod
from pathlib import Path

from bs4 import BeautifulSoup

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .buffer_manager import BufferManager
from petekit.image.image_buffer_manager import ImageBufferManager


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


class WebNavigator(BufferManager, ImageBufferManager, AgenticObject):
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

    def __init__(self) -> None:
        super().__init__()
        self._temp_dir: Path | None = None

    ################################################################################
    # Retrievers
    ################################################################################

    @tool(description="Fetch a URL and store its extracted text content in a buffer keyed by the URL.")
    def web_load_raw(self, url: str, timeout: int = 15) -> str:
        """Fetch a URL via HTTP, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        formatted = _format_for_buffer(html)
        result = self._create_buffer(url, text=formatted)
        return f"Loaded URL into buffer '{url}' ({result} lines). Read it via `read_buffer`."

    @tool(description="Render a URL via headless Chrome and store extracted text in a buffer keyed by the URL.")
    def web_render(self, url: str, timeout: int = 30) -> str:
        """Web render a URL via headless Chrome, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        html = _render_chrome(url, timeout=timeout)
        if html.startswith("Error"):
            return html
        formatted = _format_for_buffer(html)
        result = self._create_buffer(url, text=formatted)
        return f"Rendered buffer '{url}' ({result} lines)."

    @tool(description="Capture a screenshot of a web page using headless Chrome and store it in an image buffer. Pass width and height to override the default 1280x800 viewport size (useful for testing responsive layouts).")
    def web_snapshot(self, url: str, width: int = 1280, height: int = 800, timeout: int = 30) -> str:
        """Capture a screenshot of a web page and store it in an image buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        output_path = self.temp_dir / f"Snapshot_{Path(url).name}_{time_mod.time_ns()}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                (
                    _CHROME_BIN,
                    "--headless=new",
                    "--no-sandbox",
                    f"--screenshot={output_path}",
                    f"--window-size={width},{height}",
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
            image_array = cv2.imread(str(output_path))
            output_path.unlink()
            if image_array is None:
                return f"Error: failed to read screenshot for '{url}'."
            buffer_name = f"image:snapshot:{url}"
            self._store_np_buffer(buffer_name, image_array)
            return f"Stored screenshot for '{url}' in image buffer '{buffer_name}'. Use read_np_buffer to view it."
        except subprocess.TimeoutExpired:
            return f"Error: Screenshot timed out after {timeout} seconds."
        except Exception as e:
            return f"Error capturing screenshot: {type(e).__name__}: {e}"

    ################################################################################
    # Parsers
    ################################################################################

    @tool(description="Web scrape structured data from a buffered HTML page. Pass a tag name (e.g. a, img, div) and optionally an attribute name or list of attribute names to extract. Stores results as line-by-line JSON in a derived buffer.")
    def web_scrape(self, url: str, tag: str, attr: str | list[str] | None = None) -> str:
        """Extract structured data from an HTML buffer using a tag name and optional attribute specifier."""
        if url not in self._buffers:
            return f"Error: no buffer named '{url}'. Use web_load_raw to load it first."
        if not tag.strip():
            return "Error: tag must not be empty."
        buffer = self._buffers[url]
        html = "\n".join(entry.data for entry in buffer.lines)
        records = _web_scrape(html, tag, attr)
        if not records:
            return f"No <{tag}> elements found in buffer '{url}'."
        key = f"{url}:scrape:{tag}"
        formatted = _format_scrape_output(records)
        self._create_buffer(key, text=formatted)
        return f"Extracted {len(records)} <{tag}> element(s) into buffer '{key}'. Read it via `read_buffer`."

    @tool(description="Extract structured data from HTML elements matching a CSS selector from a buffered page. Pass a CSS selector (e.g. a[href], div.contact) and optionally an attribute name or list of attribute names to extract from each matched element. Stores results as line-by-line JSON in a derived buffer (keyed by buffer_name+:selector:{selector}).")
    def web_select_css(self, url: str, selector: str, attr: str | list[str] | None = None) -> str:
        """Extract elements by CSS selector from an HTML buffer already loaded via web_load_raw."""
        if url not in self._buffers:
            return f"Error: no buffer named '{url}'. Use web_load_raw to load it first."
        if not selector.strip():
            return "Error: selector must not be empty."
        buf = self._buffers[url]
        html = "\n".join(entry.data for entry in buf.lines)
        soup = BeautifulSoup(html, "lxml")
        try:
            elements = soup.select(selector)
        except Exception:
            return f"Error: invalid CSS selector '{selector}'."
        if not elements:
            return f"No elements matching selector '{selector}' found in buffer '{url}'."
        records = []
        for elem in elements:
            if attr is None:
                records.append({"text": elem.get_text(strip=True)})
            elif isinstance(attr, list):
                records.append({a: elem.get(a, "") for a in attr})
            else:
                records.append({attr: elem.get(attr, "")})
        key = f"{url}:selector:{selector}"
        formatted = _format_scrape_output(records)
        self._create_buffer(key, text=formatted)
        return f"Extracted {len(records)} element(s) into buffer '{key}'. Read it via `read_buffer`."
