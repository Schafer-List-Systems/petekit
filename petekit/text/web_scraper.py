"""WebScraper - text-based web fetching, rendering, and scraping."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from typing import Any

from peteos.oap.agentic_object import AgenticObject
from peteos import tool

from petekit.utils.web_utils import (
    _fetch_html,
    _format_for_buffer,
    _render_chrome,
    _web_scrape,
)
from petekit.utils.text_formatters import format_dict_list_for_buffer

from .buffer_manager import BufferManager


class WebScraper(BufferManager, AgenticObject):
    """You are a web scraper. You fetch and render web pages into memory buffers. Workflow:
      1. web_load_raw(url, ...) — fetch a page via HTTP into a buffer
      2. web_render(buffer, ...) — fetch via headless Chrome into a buffer
      3. web_scrape(buffer, ...) / web_select_css(url, selector, attr) — carve structured derived buffers
      4. grep_buffer(buffer, ...) + read_buffer(url, ...) — search and read
    - Use list_buffers to see loaded URLs and their derived buffers.
    - Use (?i) at the start of a grep pattern for case-insensitive matching.
    """

    ################################################################################
    # Retrievers
    ################################################################################

    @tool(description="Fetch a URL and store its extracted text content in a buffer keyed by the URL.")
    def web_load_raw(self, url: str, timeout: int = 15) -> dict[str, Any]:
        """Fetch a URL via HTTP, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "URL must start with http:// or https://."}
        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return {"ok": False, "error": html}
        formatted = _format_for_buffer(html)
        key = f"web:raw#{url}"
        result = self.create_buffer(key, text=formatted)
        return {"ok": True, "buffer": key, "lines": result}

    @tool(description="Render a URL via headless Chrome and store extracted text in a buffer keyed by the URL.")
    def web_render(self, url: str, timeout: int = 30) -> dict[str, Any]:
        """Web render a URL via headless Chrome, format for buffer, store in a buffer."""
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "URL must start with http:// or https://."}
        html = _render_chrome(url, timeout=timeout)
        if html.startswith("Error"):
            return {"ok": False, "error": html}
        formatted = _format_for_buffer(html)
        key = f"web:render#{url}"
        result = self.create_buffer(key, text=formatted)
        return {"ok": True, "buffer": key, "lines": result}

    ################################################################################
    # Parsers
    ################################################################################

    @tool(description="Web scrape structured data from a buffered HTML page. Pass a tag name (e.g. a, img, div) and optionally an attribute name or list of attribute names to extract. Stores results as line-by-line JSON in a derived buffer.")
    def web_scrape(self, buffer: str, tag: str, attr: str | list[str] | None = None) -> dict[str, Any]:
        """Extract structured data from an HTML buffer using a tag name and optional attribute specifier."""
        if buffer not in self._buffers:
            return {"ok": False, "error": f"no buffer named '{buffer}'. Use web_load_raw to load it first."}
        if not tag.strip():
            return {"ok": False, "error": "tag must not be empty."}
        buf = self._buffers[buffer]
        html = "\n".join(entry.data for entry in buf.lines)
        records = _web_scrape(html, tag, attr)
        if not records:
            return {"ok": True, "found": False, "buffer": buffer, "tag": tag}
        key = f"web:scrape:{tag}#{buffer}"
        formatted = format_dict_list_for_buffer(records)
        self.create_buffer(key, text=formatted)
        return {"ok": True, "buffer": key, "count": len(records), "tag": tag}

    @tool(description="Extract structured data from HTML elements matching a CSS selector from a buffered page. Pass a CSS selector (e.g. a[href], div.contact) and optionally an attribute name or list of attribute names to extract from each matched element. Stores results as line-by-line JSON in a derived buffer (keyed by buffer_name+:selector:{selector}).")
    def web_select_css(self, buffer: str, selector: str, attr: str | list[str] | None = None) -> dict[str, Any]:
        """Extract elements by CSS selector from an HTML buffer already loaded via web_load_raw."""
        if buffer not in self._buffers:
            return {"ok": False, "error": f"no buffer named '{buffer}'. Use web_load_raw to load it first."}
        if not selector.strip():
            return {"ok": False, "error": "selector must not be empty."}
        buf = self._buffers[buffer]
        html = "\n".join(entry.data for entry in buf.lines)
        soup = BeautifulSoup(html, "lxml")
        try:
            elements = soup.select(selector)
        except Exception:
            return {"ok": False, "error": f"invalid CSS selector '{selector}'."}
        if not elements:
            return {"ok": True, "found": False, "buffer": buffer, "selector": selector}
        records = []
        for elem in elements:
            if attr is None:
                records.append({"text": elem.get_text(strip=True)})
            elif isinstance(attr, list):
                records.append({a: elem.get(a, "") for a in attr})
            else:
                records.append({attr: elem.get(attr, "")})
        key = f"web:selector:{selector}#{buffer}"
        formatted = format_dict_list_for_buffer(records)
        self.create_buffer(key, text=formatted)
        return {"ok": True, "buffer": key, "count": len(records), "selector": selector}
