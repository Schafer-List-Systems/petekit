"""WebScraper - text-based web fetching, rendering, and scraping."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from petekit.utils.web_utils import (
    _fetch_html,
    _format_for_buffer,
    _format_scrape_output,
    _render_chrome,
    _web_scrape,
)

from .buffer_manager import BufferManager


class WebScraper(BufferManager, AgenticObject):
    """You are a web scraper. You fetch and render web pages into memory buffers.

    Workflow:
      1. web_load_raw(url) — fetch a page via HTTP into a buffer (keyed by URL)
      2. web_render(url) — fetch via headless Chrome into a buffer (keyed by URL)
      3. web_scrape(url, tag, attr) / web_select_css(url, selector, attr) — carve structured derived buffers
      4. grep_buffer(url, "regex") + read_buffer(url, start=N, end=M) — search and read

    Use list_buffers to see loaded URLs and their derived buffers.
    Use (?i) at the start of a grep pattern for case-insensitive matching.
    """

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
