"""RaccoonLM v2 — Internet Plugin

Provides web search (DuckDuckGo) and page-fetch capabilities
for the AI to use as tools during chat.
Inherits from Plugin base class for consistent architecture.
"""

import json
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from raccoonlm.plugins.base import Plugin


class InternetPlugin(Plugin):
    """Plugin that gives the model web access via tools."""

    @property
    def name(self) -> str:
        return "internet"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    async def shutdown(self):
        await self.client.aclose()

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract readable text content from a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to fetch",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "web_search":
            return await self._web_search(args.get("query", ""))
        elif name == "web_fetch":
            return await self._web_fetch(args.get("url", ""))
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _web_search(self, query: str) -> str:
        """Web search — DuckDuckGo HTML with session cookies."""
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            # Step 1: Visit duckduckgo.com to get session cookies
            try:
                await self.client.get("https://duckduckgo.com/", headers=headers)
            except:
                pass  # Cookies may already be set
            
            # Step 2: Search via HTML endpoint with cookies
            resp = await self.client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={**headers, "Origin": "https://duckduckgo.com", "Referer": "https://duckduckgo.com/"},
            )
            
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for result in soup.select(".result")[:8]:
                title_el = result.select_one(".result__title a")
                snippet_el = result.select_one(".result__snippet")
                if title_el:
                    title = title_el.get_text(strip=True)
                    href = title_el.get("href", "")
                    match = re.search(r"uddg=(https?://[^&]+)", str(href))
                    url = match.group(1) if match else href
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    results.append({"title": title, "url": url, "snippet": snippet})
            
            if results:
                return json.dumps({"results": results}, ensure_ascii=False)
            
            # Step 3: Fallback — try DuckDuckGo API instant answer
            api_resp = await self.client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
            )
            if api_resp.status_code == 200:
                api_data = api_resp.json()
                related = api_data.get("RelatedTopics", [])
                api_results = []
                for t in related:
                    if "Text" in t:
                        api_results.append({
                            "title": t["Text"][:80],
                            "url": t.get("FirstURL", ""),
                            "snippet": t["Text"][:200],
                        })
                    elif "Topics" in t:
                        for sub in t["Topics"][:3]:
                            api_results.append({
                                "title": sub.get("Text", "")[:80],
                                "url": sub.get("FirstURL", ""),
                                "snippet": sub.get("Text", "")[:200],
                            })
                if api_results:
                    return json.dumps({"results": api_results}, ensure_ascii=False)
            
            return json.dumps({"results": [], "note": "No results found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL and extract clean text content."""
        try:
            resp = await self.client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)

            # Truncate to avoid context overflow
            if len(content) > 8000:
                content = content[:8000] + "\n\n[...truncated...]"

            return json.dumps(
                {"url": url, "content": content, "length": len(content)},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch {url}: {str(e)}"})
