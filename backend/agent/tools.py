"""
Tools the agent can call.

One of them (`search_my_documents`) is built per user by a factory, so the
closure captures the authenticated user id. The rest hit free, key-free public
APIs:

  * Open-Meteo          weather + geocoding      https://open-meteo.com
  * Wikipedia REST      encyclopedia lookup      https://en.wikipedia.org/api/rest_v1
  * open.er-api.com     currency exchange rates
  * exec-free calculator using Python's ast module

Every tool returns a plain string, catches its own exceptions and never raises,
because a tool crash inside the graph would abort the user's whole turn.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import List

import requests
from langchain_core.tools import BaseTool, tool

from backend.rag import vectorstore

HTTP_TIMEOUT = 12  # seconds; keeps a slow API from stalling the stream


# ---------------------------------------------------------------------------
# 1. RAG retrieval (per user)
# ---------------------------------------------------------------------------
def make_document_tool(user_id: int) -> BaseTool:
    """
    Build a retrieval tool bound to one user's FAISS index.

    The user id is captured in the closure and is never taken from the model's
    arguments, so the LLM cannot be tricked into reading someone else's files.
    """

    @tool("search_my_documents")
    def search_my_documents(query: str) -> str:
        """Search the user's uploaded PDF documents for passages relevant to a
        question. Use this whenever the user asks about "my document", "the PDF",
        "the report", or anything that sounds like it comes from a file they
        uploaded. Input should be a focused search query."""
        try:
            hits = vectorstore.search(user_id, query)
        except Exception as exc:
            return f"Document search failed: {exc}"

        if not hits:
            return (
                "No uploaded documents matched that query. The user may not have "
                "uploaded any PDFs yet. Answer from general knowledge and say so."
            )

        parts: List[str] = []
        for i, hit in enumerate(hits, start=1):
            parts.append(
                f"[Excerpt {i} | {hit.source} p.{hit.page} | similarity {hit.score:.2f}]\n"
                f"{hit.text.strip()}"
            )
        return (
            "Relevant excerpts from the user's documents. Cite the source file "
            "and page when you use them.\n\n" + "\n\n---\n\n".join(parts)
        )

    return search_my_documents


# ---------------------------------------------------------------------------
# 2. Weather (Open-Meteo, no API key)
# ---------------------------------------------------------------------------
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a city or town anywhere in the world.
    Input should be a place name, optionally with a country, such as
    "Patna" or "Lisbon, Portugal"."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"Could not find a place called '{city}'."

        place = results[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=HTTP_TIMEOUT,
        ).json()

        cur = forecast.get("current", {})
        code = cur.get("weather_code")
        label = _WEATHER_CODES.get(code, "unknown conditions")
        location = ", ".join(
            filter(None, [place.get("name"), place.get("admin1"), place.get("country")])
        )
        return (
            f"Weather in {location}: {label}. "
            f"Temperature {cur.get('temperature_2m')}degC "
            f"(feels like {cur.get('apparent_temperature')}degC), "
            f"humidity {cur.get('relative_humidity_2m')}%, "
            f"wind {cur.get('wind_speed_10m')} km/h."
        )
    except Exception as exc:
        return f"Weather lookup failed: {exc}"


# ---------------------------------------------------------------------------
# 3. Wikipedia lookup
# ---------------------------------------------------------------------------
@tool("search_wikipedia")
def search_wikipedia(query: str) -> str:
    """Look up factual background information on Wikipedia. Useful for people,
    places, organisations, historical events and scientific concepts. Input
    should be a short search phrase, not a full sentence."""
    try:
        hits = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": 2, "format": "json",
            },
            headers={"User-Agent": "ConversationalAIPlatform/1.0"},
            timeout=HTTP_TIMEOUT,
        ).json()

        titles = [h["title"] for h in hits.get("query", {}).get("search", [])]
        if not titles:
            return f"Wikipedia has no article matching '{query}'."

        summaries: List[str] = []
        for title in titles:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + requests.utils.quote(title.replace(" ", "_")),
                headers={"User-Agent": "ConversationalAIPlatform/1.0"},
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            extract = (data.get("extract") or "").strip()
            if extract:
                url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                summaries.append(f"{data.get('title')}: {extract}\nSource: {url}")

        return "\n\n".join(summaries) if summaries else "No usable summary found."
    except Exception as exc:
        return f"Wikipedia lookup failed: {exc}"


# ---------------------------------------------------------------------------
# 4. Currency conversion
# ---------------------------------------------------------------------------
@tool("convert_currency")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money between two currencies using live mid-market
    rates. Currencies must be three-letter ISO codes such as USD, INR, EUR, JPY."""
    try:
        base = from_currency.strip().upper()
        target = to_currency.strip().upper()
        data = requests.get(
            f"https://open.er-api.com/v6/latest/{base}", timeout=HTTP_TIMEOUT
        ).json()

        if data.get("result") != "success":
            return f"Unknown or unsupported currency code '{base}'."

        rate = data.get("rates", {}).get(target)
        if rate is None:
            return f"No exchange rate available for {base} to {target}."

        converted = float(amount) * float(rate)
        return (
            f"{amount:,.2f} {base} = {converted:,.2f} {target} "
            f"(rate 1 {base} = {rate:.4f} {target}, updated {data.get('time_last_update_utc')})"
        )
    except Exception as exc:
        return f"Currency conversion failed: {exc}"


# ---------------------------------------------------------------------------
# 5. Web search (DuckDuckGo, no API key)
# ---------------------------------------------------------------------------
# The library formerly published as `duckduckgo-search` is now `ddgs`. It is
# imported lazily inside the tool so that a missing or broken install degrades
# to a clear message instead of breaking the whole agent at startup.

def _ddgs_search(category: str, query: str, max_results: int) -> List[dict]:
    """
    Run a ddgs query, preferring the DuckDuckGo backend and falling back to the
    library's automatic backend selection if DuckDuckGo returns nothing.

    ddgs is a scraper, not an official API, so any single backend can go quiet
    without warning. The fallback is what keeps the tool useful when that
    happens. Raises on failure; the caller turns exceptions into text.
    """
    from ddgs import DDGS

    with DDGS() as client:
        method = getattr(client, category)
        results = method(
            query,
            region="wt-wt",       # worldwide, no regional weighting
            safesearch="moderate",
            max_results=max_results,
            backend="duckduckgo",
        )
        if not results:
            results = method(
                query,
                region="wt-wt",
                safesearch="moderate",
                max_results=max_results,
                backend="auto",   # bing, brave and others as a safety net
            )
    return results or []


def _format_results(results: List[dict], limit: int) -> str:
    """Render raw ddgs dicts into something an LLM can cite.

    Different backends label the link differently, `href` on some and `url` on
    others, so both are read.
    """
    lines: List[str] = []
    for i, item in enumerate(results[:limit], start=1):
        title = (item.get("title") or "Untitled").strip()
        link = (item.get("href") or item.get("url") or "").strip()
        body = (item.get("body") or item.get("excerpt") or "").strip()
        date = (item.get("date") or "").strip()

        header = f"[{i}] {title}"
        if date:
            header += f"  ({date})"
        lines.append(f"{header}\n{body}\nSource: {link}")

    return "\n\n".join(lines)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web via DuckDuckGo for current information. Use this for
    anything recent, changing, or outside your training data: news, prices,
    releases, current office holders, company details, sports results, or when
    the user asks what is happening now. Prefer search_wikipedia for stable
    encyclopedic background, and this tool for anything time-sensitive. Input
    should be a focused search query, not a full sentence."""
    try:
        results = _ddgs_search("text", query, max_results=6)
    except ImportError:
        return (
            "Web search is unavailable because the 'ddgs' package is not "
            "installed. Run: pip install ddgs"
        )
    except Exception as exc:
        name = type(exc).__name__
        if "Ratelimit" in name:
            return (
                "DuckDuckGo is rate limiting this client right now. Tell the user "
                "to try again shortly, and answer from what you already know."
            )
        if "Timeout" in name:
            return "The web search timed out. Answer from existing knowledge and say so."
        return f"Web search failed ({name}): {exc}"

    if not results:
        return (
            f"No web results for '{query}'. Try a shorter or differently worded "
            "query, or answer from existing knowledge and say the search came back empty."
        )

    return (
        f"Live DuckDuckGo results for '{query}'. Cite the source URLs you use, "
        "and note that snippets may be incomplete.\n\n"
        + _format_results(results, limit=6)
    )


@tool("news_search")
def news_search(query: str) -> str:
    """Search recent news articles via DuckDuckGo, returning headlines with
    publication dates. Use this instead of web_search when the user explicitly
    asks about news, current events, or what has happened recently on a topic.
    Input should be a short topic or entity name."""
    try:
        results = _ddgs_search("news", query, max_results=6)
    except ImportError:
        return (
            "News search is unavailable because the 'ddgs' package is not "
            "installed. Run: pip install ddgs"
        )
    except Exception as exc:
        name = type(exc).__name__
        if "Ratelimit" in name:
            return "DuckDuckGo is rate limiting this client right now. Try again shortly."
        return f"News search failed ({name}): {exc}"

    if not results:
        return f"No recent news found for '{query}'."

    return (
        f"Recent news matching '{query}', newest first where available. Cite the "
        "source URLs and mention publication dates.\n\n"
        + _format_results(results, limit=6)
    )


# ---------------------------------------------------------------------------
# 6. Safe calculator
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Walk the AST and evaluate only arithmetic. No names, calls or attributes."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only plain arithmetic is allowed")


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression precisely. Use this instead of doing
    mental maths for anything non-trivial. Supports + - * / // % and **,
    for example "1580 * 1.18" or "(45000 - 12000) / 12"."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
        return f"{expression} = {result:,}"
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


# ---------------------------------------------------------------------------
# 7. Current date and time
# ---------------------------------------------------------------------------
@tool("get_current_datetime")
def get_current_datetime() -> str:
    """Return the current UTC date, time and weekday. Call this before doing any
    reasoning that depends on today's date, since the model's own sense of
    'today' is unreliable."""
    now = datetime.now(timezone.utc)
    return now.strftime("Current UTC datetime: %A, %d %B %Y, %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PUBLIC_TOOLS: List[BaseTool] = [
    web_search,
    news_search,
    get_weather,
    search_wikipedia,
    convert_currency,
    calculator,
    get_current_datetime,
]


def build_toolset(user_id: int) -> List[BaseTool]:
    """All tools available to one user: their private RAG tool plus the public ones."""
    return [make_document_tool(user_id)] + PUBLIC_TOOLS
