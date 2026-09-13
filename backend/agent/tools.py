"""
Tools the agent can call.

One of them (`search_my_documents`) is built per user by a factory, so the
closure captures the authenticated user id. The other two hit free, key-free
public APIs:

  * DuckDuckGo Instant Answer API   web lookups           https://api.duckduckgo.com
  * Open-Meteo                      weather + geocoding   https://open-meteo.com

Plus a basic four-operation calculator with no external dependency.

The toolset is deliberately small: one tool per capability, no overlapping
alternatives. Earlier revisions also included Wikipedia lookup, currency
conversion, and a clock tool; they were cut to keep exactly these four.

Every tool returns a plain string, catches its own exceptions and never raises,
because a tool crash inside the graph would abort the user's whole turn.
"""

from __future__ import annotations

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
# 3. Web search (DuckDuckGo's official Instant Answer API, no API key)
# ---------------------------------------------------------------------------
# This is DuckDuckGo's own free, keyless JSON endpoint at api.duckduckgo.com,
# not a scraper. Worth being upfront about what it actually is: DuckDuckGo's
# own documentation is explicit that this is NOT a full search-results API.
# It powers their "instant answer" knowledge panels, definitions, abstracts
# and related topics, sourced from Wikipedia and similar. It answers "what is
# FAISS" or "who is the CEO of X" well, and returns nothing for narrow,
# highly specific, or purely news-style queries that would need a genuine
# ranked list of articles. That tradeoff is accepted here in exchange for
# using an endpoint DuckDuckGo actually supports, instead of scraping their
# results page.

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo's Instant Answer API for definitions,
    factual lookups, and background on known people, places, organisations
    and concepts. Best for "what is" / "who is" style questions about
    established topics. It does not return a ranked list of articles, so it
    may return nothing for very narrow, obscure, or breaking-news queries;
    if that happens, say so and answer from existing knowledge. Input should
    be a short, focused search phrase."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
                "no_redirect": 1,
            },
            headers={"User-Agent": "ConversationalAIPlatform/1.0"},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        return "The web search timed out. Answer from existing knowledge and say so."
    except Exception as exc:
        return f"Web search failed ({type(exc).__name__}): {exc}"

    sections: List[str] = []

    # A direct instant answer, e.g. simple calculations or quick facts.
    if data.get("Answer"):
        sections.append(f"Direct answer: {data['Answer']}")

    # A dictionary-style definition.
    if data.get("Definition"):
        source = data.get("DefinitionSource", "")
        sections.append(f"Definition ({source}): {data['Definition']}")

    # The main encyclopedic abstract, usually sourced from Wikipedia.
    if data.get("AbstractText"):
        heading = data.get("Heading") or query
        source = data.get("AbstractSource", "")
        url = data.get("AbstractURL", "")
        sections.append(
            f"{heading} ({source}): {data['AbstractText']}\nSource: {url}"
        )

    # Related topics as a supplementary list, when present.
    related_lines: List[str] = []
    for item in data.get("RelatedTopics", [])[:5]:
        text = item.get("Text")
        url = item.get("FirstURL")
        if text and url:
            related_lines.append(f"- {text}\n  Source: {url}")
    if related_lines:
        sections.append("Related topics:\n" + "\n".join(related_lines))

    if not sections:
        return (
            f"DuckDuckGo's instant answer API returned nothing for '{query}'. "
            "This endpoint only covers known topics and entities, not a general "
            "web crawl, so try a more specific or well-known term, or answer "
            "from existing knowledge and say the search came back empty."
        )

    return (
        f"DuckDuckGo instant answer results for '{query}'. Cite the sources "
        "given, and note this covers known topics/entities rather than a "
        "full web crawl, so treat it as a starting point rather than "
        "exhaustive.\n\n" + "\n\n".join(sections)
    )


# ---------------------------------------------------------------------------
# 4. Calculator (basic arithmetic only)
# ---------------------------------------------------------------------------
_OPERATIONS = {
    "add": lambda a, b: a + b, "+": lambda a, b: a + b,
    "subtract": lambda a, b: a - b, "-": lambda a, b: a - b,
    "multiply": lambda a, b: a * b, "*": lambda a, b: a * b, "x": lambda a, b: a * b,
    "divide": lambda a, b: a / b, "/": lambda a, b: a / b,
}
_SYMBOLS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}


@tool("calculator")
def calculator(a: float, b: float, operation: str) -> str:
    """Perform one basic arithmetic operation between two numbers. operation
    must be one of: add, subtract, multiply, divide (the symbols +, -, *, /
    are also accepted). For a multi-step calculation, call this tool more
    than once, one operation at a time."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return f"Could not parse numbers: a={a!r}, b={b!r}"

    op_key = operation.strip().lower()
    func = _OPERATIONS.get(op_key)
    if func is None:
        return (
            f"Unknown operation '{operation}'. Use add, subtract, multiply, "
            "or divide (or +, -, *, /)."
        )

    if op_key in ("divide", "/") and b == 0:
        return "Cannot divide by zero."

    result = func(a, b)
    symbol = _SYMBOLS.get(op_key, op_key)
    # Whole-number results print as "175,500" rather than "175,500.0", matching
    # how the operands themselves are already formatted with {:g}.
    result_str = f"{result:,.0f}" if result == int(result) else f"{result:,}"
    return f"{a:g} {symbol} {b:g} = {result_str}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PUBLIC_TOOLS: List[BaseTool] = [
    web_search,
    get_weather,
    calculator,
]


def build_toolset(user_id: int) -> List[BaseTool]:
    """All tools available to one user: their private RAG tool plus the public ones."""
    return [make_document_tool(user_id)] + PUBLIC_TOOLS