"""
tools.py

The FitFindr tools. Each tool is a standalone function that can be called and
tested independently before being wired into the agent loop (see agent.py).

Required tools:
    search_listings(description, size, max_price)        -> list[dict]
    suggest_outfit(new_item, wardrobe, trends)           -> str
    create_fit_card(outfit, new_item)                    -> str

Stretch tools:
    compare_price(item, listings)                        -> dict
    get_trending_styles(size, top_n)                     -> dict
    load_style_profile(user_id)                          -> dict
    save_style_profile(profile, user_id)                 -> bool

Design rule shared by every tool in this file: a tool never raises for an
expected failure. It returns an empty list, a sentinel string, or an
"unknown" verdict, and the planning loop decides what to do about it.
"""

import json
import os
import re
import statistics
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

# The assignment recommends meta-llama/llama-4-scout-17b-16e-instruct. That
# model has been retired from the Groq free tier (404 model_not_found), so the
# first entry that the account can actually serve is used instead. Order is
# preference order; see "Spec Reflection" in the README.
PREFERRED_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

_CLIENT = None
_RESOLVED_MODEL = None

# Sentinel prefixes. The planning loop branches on these instead of on
# exception types, so a tool failure is data, not a crash.
OUTFIT_UNAVAILABLE = "[outfit-unavailable]"
FIT_CARD_UNAVAILABLE = "[fit-card-unavailable]"


class LLMUnavailable(RuntimeError):
    """Raised internally when the LLM cannot be reached; never escapes a tool."""


def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY not set. Add it to a .env file in the project root."
            )
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def _resolve_model() -> str:
    """
    Return the first model in PREFERRED_MODELS the account can actually serve.

    Asks the API once per process and caches the answer. If the model list
    cannot be fetched, falls back to the first preference and lets the call
    itself fail (and be handled) downstream.
    """
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    try:
        available = {m.id for m in _get_groq_client().models.list().data}
    except Exception:
        _RESOLVED_MODEL = PREFERRED_MODELS[0]
        return _RESOLVED_MODEL
    for model in PREFERRED_MODELS:
        if model in available:
            _RESOLVED_MODEL = model
            return model
    _RESOLVED_MODEL = PREFERRED_MODELS[0]
    return _RESOLVED_MODEL


def _chat(prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> str:
    """
    Send one prompt to the LLM and return its text.

    Retries once on any transport/API error before giving up. The retry also
    triples the token budget: the available Groq models are reasoning models
    that spend tokens thinking before they emit any content, so a long prompt
    on a tight budget comes back with an empty message rather than an error.
    Raises LLMUnavailable on a hard failure or a still-empty completion — every
    caller in this file catches it and converts it into a user-facing string.
    """
    last_error = None
    for attempt in range(2):
        try:
            response = _get_groq_client().chat.completions.create(
                model=_resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens * (3 if attempt else 1),
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            # "length" means the answer was cut off mid-sentence. On the first
            # attempt that is worth redoing with a bigger budget; on the last
            # one, a truncated answer still beats no answer.
            truncated = getattr(choice, "finish_reason", None) == "length"
            if text and not (truncated and attempt == 0):
                return text
            last_error = (
                "completion truncated" if truncated
                else "empty completion (model spent the budget reasoning)"
            )
        except LLMUnavailable:
            raise
        except Exception as exc:  # network, rate limit, bad key, model gone
            last_error = f"{type(exc).__name__}: {exc}"
    raise LLMUnavailable(f"LLM call failed after 2 attempts ({last_error}).")


# ── shared text helpers ───────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with", "some", "something", "any",
    "i", "im", "i'm", "me", "my", "mine", "you", "your", "is", "are", "am",
    "want", "wanted", "need", "needed", "looking", "look", "find", "finding",
    "get", "got", "buy", "search", "searching", "show", "please", "thanks",
    "under", "below", "less", "than", "over", "about", "around", "max",
    "maximum", "budget", "price", "cheap", "size", "sized", "fits", "fit",
    "in", "on", "of", "to", "at", "that", "this", "it", "its", "would",
    "wear", "wearing", "wears", "mostly", "usually", "like", "really",
    "what", "whats", "out", "there", "how", "style", "styling", "dollars",
    "bucks", "usd", "thrift", "thrifted", "thrifting", "secondhand", "piece",
    # greetings and filler — a query made only of these has nothing to search on
    "hey", "hi", "hello", "yo", "hiya", "help", "ok", "okay", "hmm", "um",
    "anything", "stuff", "things", "thing", "idk", "whatever", "clothes",
}

# Common plural/singular pairs worth normalising so "tees" matches "tee".
_IRREGULAR = {"jeans": "jeans", "pants": "pants", "shorts": "shorts"}


def _stem(token: str) -> str:
    """Crude singulariser — enough to make tees/tee and boots/boot match."""
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stop-words and bare numbers removed."""
    raw = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [t for t in raw if t not in _STOPWORDS and not t.isdigit()]


def _token_set(text: str) -> set[str]:
    """Tokens plus their stems, for forgiving comparison."""
    tokens = _tokenize(text)
    return set(tokens) | {_stem(t) for t in tokens}


def _size_tokens(size: str) -> set[str]:
    """
    Split a listing's size string into comparable tokens.

    "S/M"              -> {"s", "m"}
    "XL (oversized)"   -> {"xl", "oversized"}
    "W30 L30"          -> {"w30", "l30", "30"}
    "US 8.5"           -> {"us", "8.5"}
    Splitting first is what keeps a query of "M" from matching inside "XL".
    """
    parts = re.split(r"[^a-z0-9.]+", (size or "").lower())
    tokens = {p for p in parts if p}
    for part in list(tokens):
        waist = re.fullmatch(r"[wl](\d+)", part)
        if waist:
            tokens.add(waist.group(1))
    return tokens


_WORD_SIZES = {
    "xs": "xs", "extra small": "xs", "small": "s", "s": "s",
    "medium": "m", "med": "m", "m": "m",
    "large": "l", "l": "l", "xl": "xl", "extra large": "xl", "xxs": "xxs",
}


def _normalize_size(size) -> str | None:
    """Normalise a user-supplied size to the token form used in the dataset."""
    if size is None:
        return None
    cleaned = str(size).strip().lower()
    if not cleaned:
        return None
    return _WORD_SIZES.get(cleaned, cleaned)


def _size_matches(listing_size: str, wanted: str) -> bool:
    """
    True if a listing in `listing_size` plausibly fits someone wanting `wanted`.

    "One Size" listings match everyone. Otherwise the wanted size has to appear
    as a whole token of the listing size.
    """
    tokens = _size_tokens(listing_size)
    if {"one", "size"} <= tokens:
        return True
    return wanted in tokens


# ── Tool 1: search_listings ───────────────────────────────────────────────────

# Field weights for keyword relevance scoring. style_tags is weighted highest
# because that is where the dataset encodes "vintage", "y2k", "grunge" etc.
_FIELD_WEIGHTS = {
    "style_tags": 3.0,
    "title": 2.5,
    "category": 2.0,
    "brand": 1.5,
    "colors": 1.5,
    "description": 1.0,
}


def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match
        first). Each dict is the dataset listing (id, title, description,
        category, style_tags, size, condition, price, colors, brand, platform)
        plus two fields added by this tool: "match_score" (float) and
        "matched_terms" (list[str] — which query words the listing matched).
        Returns an empty list if nothing matches — does NOT raise an exception.
    """
    try:
        listings = load_listings()
    except Exception:
        # Missing or malformed data file: report "nothing found" rather than
        # taking the whole agent down. The loop surfaces this to the user.
        return []

    wanted_size = _normalize_size(size)
    query_tokens = _tokenize(description)
    query_set = set(query_tokens) | {_stem(t) for t in query_tokens}
    phrase = " ".join(query_tokens)

    # Relevance floor: a listing has to cover at least half of the distinct
    # words in the query. Without this, "90s track jacket" matches a bucket hat
    # tagged "90s" — technically a hit, obviously not what was asked for.
    distinct_terms = {_stem(t) for t in query_tokens}
    min_terms = max(1, (len(distinct_terms) + 1) // 2)

    scored = []
    for listing in listings:
        if max_price is not None and float(listing.get("price", 0)) > float(max_price):
            continue
        if wanted_size and not _size_matches(listing.get("size", ""), wanted_size):
            continue

        score = 0.0
        matched_terms = set()
        for field, weight in _FIELD_WEIGHTS.items():
            value = listing.get(field)
            if value is None:
                continue
            text = " ".join(value) if isinstance(value, list) else str(value)
            field_tokens = _token_set(text)
            overlap = query_set & field_tokens
            score += weight * len(overlap)
            matched_terms |= {_stem(t) for t in overlap}
            # Whole-phrase hit ("graphic tee" inside the style tags) is worth
            # more than the sum of its words.
            if phrase and len(query_tokens) > 1 and phrase in text.lower():
                score += weight

        if score <= 0 or len(matched_terms & distinct_terms) < min_terms:
            continue

        result = dict(listing)
        result["match_score"] = round(score, 2)
        # Which of the user's words this listing actually matched. The planning
        # loop uses it to tell "an equally good match that costs less" apart
        # from "a cheaper item that happens to share one word".
        result["matched_terms"] = sorted(matched_terms & distinct_terms)
        scored.append(result)

    # Best keyword match first; cheaper item wins ties so the top result is
    # never needlessly expensive.
    scored.sort(key=lambda item: (-item["match_score"], item.get("price", 0)))
    return scored


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def _describe_item(item: dict) -> str:
    """One-line plain-text summary of a listing, for use inside a prompt."""
    return (
        f"{item.get('title', 'Unknown item')} "
        f"(category: {item.get('category', 'unknown')}; "
        f"colors: {', '.join(item.get('colors') or ['unspecified'])}; "
        f"style: {', '.join(item.get('style_tags') or ['unspecified'])}; "
        f"size {item.get('size', 'unknown')}; {item.get('condition', 'unknown')} condition; "
        f"${item.get('price', 0):.2f} on {item.get('platform', 'unknown')})"
    )


def _describe_wardrobe(wardrobe: dict) -> str:
    """Numbered plain-text list of wardrobe pieces, for use inside a prompt."""
    lines = []
    for item in wardrobe.get("items", []):
        note = f" — {item['notes']}" if item.get("notes") else ""
        lines.append(
            f"- {item.get('name', 'unnamed')} "
            f"[{item.get('category', '?')}; {', '.join(item.get('colors') or [])}; "
            f"{', '.join(item.get('style_tags') or [])}]{note}"
        )
    return "\n".join(lines)


def suggest_outfit(new_item: dict, wardrobe: dict, trends: list[str] | None = None) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1-2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handled gracefully.
        trends:   Optional list of trending style tags from get_trending_styles().
                  None means no trend context; the tool works either way.

    Returns:
        A non-empty string with outfit suggestions. If the wardrobe is empty it
        returns general styling advice instead of raising. If the LLM cannot be
        reached it returns a string beginning with "[outfit-unavailable]".
    """
    if not isinstance(new_item, dict) or not new_item:
        return (
            f"{OUTFIT_UNAVAILABLE} No item was passed in, so there's nothing to "
            "style. Search for a piece first."
        )

    wardrobe = wardrobe if isinstance(wardrobe, dict) else {"items": []}
    items = wardrobe.get("items") or []
    trend_line = (
        f"\nStyles trending in this catalogue right now: {', '.join(trends)}. "
        "Mention in one short clause if an outfit leans into one of them."
        if trends
        else ""
    )

    if not items:
        # Failure mode 1: empty wardrobe. Not an error — a different prompt.
        prompt = (
            "You are a thrift stylist. A shopper is considering this secondhand "
            f"piece:\n\n{_describe_item(new_item)}\n\n"
            "They have not told you anything about what they already own. "
            "In 3-4 sentences, give practical general styling advice: what kinds "
            "of pieces pair well with it, one specific styling move (a tuck, a "
            "cuff, a layer), and the vibe or occasion it suits. Name garment "
            "types, not brands. Do not invent items they own."
            f"{trend_line}"
        )
        try:
            advice = _chat(prompt, temperature=0.8, max_tokens=600)
        except LLMUnavailable as exc:
            return f"{OUTFIT_UNAVAILABLE} Couldn't reach the styling model ({exc})."
        return (
            "You haven't added any wardrobe pieces yet, so here's general styling "
            "for this piece — add a few items and I'll style it against what you "
            f"actually own.\n\n{advice}"
        )

    # Normal path: style against real, named wardrobe pieces.
    prompt = (
        "You are a thrift stylist. A shopper is considering buying this "
        f"secondhand piece:\n\n{_describe_item(new_item)}\n\n"
        f"Here is everything already in their wardrobe:\n{_describe_wardrobe(wardrobe)}\n\n"
        "Suggest one or two complete outfits built around the new piece. Rules:\n"
        "- Only use wardrobe pieces from the list above, and name them exactly.\n"
        "- Each outfit needs a top, a bottom and shoes (the new piece covers one).\n"
        "- Include one concrete styling move (tuck, cuff, layer, half-tuck).\n"
        "- Hard limit: 90 words total, 4 sentences maximum, one paragraph.\n"
        "- Conversational prose — no bullet points, no headings, no preamble.\n"
        "- Do not invent pieces they do not own."
        f"{trend_line}"
    )
    try:
        return _chat(prompt, temperature=0.75, max_tokens=700)
    except LLMUnavailable as exc:
        return (
            f"{OUTFIT_UNAVAILABLE} Couldn't reach the styling model ({exc}). "
            "The listing and price check above are still good."
        )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def _template_fit_card(new_item: dict) -> str:
    """
    Offline fallback caption, built from the item's own fields.

    Used only when the LLM is unreachable, so the user still leaves with
    something shareable instead of an error.
    """
    tags = new_item.get("style_tags") or []
    vibe = tags[0] if tags else "thrifted"
    return (
        f"thrifted this {new_item.get('title', 'find')} for "
        f"${new_item.get('price', 0):.2f} on {new_item.get('platform', 'resale')} "
        f"and it's exactly the {vibe} energy i was after. "
        "(caption written offline — the styling model was unreachable)"
    )


def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2-4 sentence string usable as an Instagram/TikTok caption. If the
        outfit is missing, empty, or itself unavailable, returns a descriptive
        error string beginning with "[fit-card-unavailable]" — it does NOT
        raise. Uses a high temperature so repeat calls differ.
    """
    if not isinstance(outfit, str) or not outfit.strip():
        return (
            f"{FIT_CARD_UNAVAILABLE} Can't write a fit card without an outfit — "
            "run a search and get a styling suggestion first so I have something "
            "to caption."
        )
    if outfit.strip().startswith(OUTFIT_UNAVAILABLE):
        return (
            f"{FIT_CARD_UNAVAILABLE} The styling step didn't produce an outfit, "
            "so there's nothing real to caption. Try the search again in a moment."
        )
    if not isinstance(new_item, dict) or not new_item:
        return (
            f"{FIT_CARD_UNAVAILABLE} Can't write a fit card without the item "
            "details (title, price, platform)."
        )

    # The caption only needs the gist of the outfit. Feeding in a long styling
    # answer verbatim crowds out the model's own response budget.
    outfit_gist = outfit.strip()
    if len(outfit_gist) > 600:
        outfit_gist = outfit_gist[:600].rsplit(" ", 1)[0] + "…"

    prompt = (
        "Write a caption for a social post about a secondhand clothing find.\n\n"
        f"The piece: {_describe_item(new_item)}\n"
        f"How it's being worn: {outfit_gist}\n\n"
        "Rules:\n"
        "- 2 to 4 short sentences, lowercase-leaning, casual, first person.\n"
        "- Sound like a real person posting their outfit, NOT a product listing.\n"
        f"- Mention the item, the ${new_item.get('price', 0):.0f} price and "
        f"{new_item.get('platform', 'the app')} once each, naturally.\n"
        "- Name one or two specific pieces from the outfit above.\n"
        "- At most one emoji. No hashtag walls, no quotation marks around the caption.\n"
        "- Return only the caption text."
    )
    try:
        # High temperature: the same item should not produce the same caption twice.
        return _chat(prompt, temperature=1.05, max_tokens=500)
    except LLMUnavailable:
        return _template_fit_card(new_item)


# ── Tool 4 [stretch]: compare_price ───────────────────────────────────────────

def compare_price(item: dict, listings: list[dict] | None = None) -> dict:
    """
    Estimate whether a listing is fairly priced against comparable listings.

    Comparables are listings in the same category, excluding the item itself,
    ranked by style_tag overlap. The verdict compares the item's price to the
    median of those comparables.

    Args:
        item:     The listing dict being judged.
        listings: Pool to compare against. None loads the full dataset.

    Returns:
        dict with keys:
            verdict (str)            "great deal" | "fair" | "a bit high" |
                                     "overpriced" | "unknown"
            item_price (float)
            median_comparable (float | None)
            pct_diff (float | None)  % above (+) or below (-) the median
            n_comparables (int)
            comparables (list[dict]) up to 3 closest, each {title, price, condition}
            reasoning (str)          one sentence explaining the verdict
    """
    if not isinstance(item, dict) or "price" not in item:
        return {
            "verdict": "unknown", "item_price": 0.0, "median_comparable": None,
            "pct_diff": None, "n_comparables": 0, "comparables": [],
            "reasoning": "No item price to compare.",
        }

    price = float(item.get("price", 0))
    if listings is None:
        try:
            listings = load_listings()
        except Exception:
            listings = []
    # A search result list is usually short; widen to the full catalogue so the
    # median is meaningful.
    if len(listings) < 4:
        try:
            listings = load_listings()
        except Exception:
            pass

    item_tags = set(item.get("style_tags") or [])
    pool = []
    for other in listings or []:
        if other.get("id") == item.get("id"):
            continue
        if other.get("category") != item.get("category"):
            continue
        overlap = len(item_tags & set(other.get("style_tags") or []))
        pool.append((overlap, other))

    if len(pool) < 2:
        return {
            "verdict": "unknown", "item_price": price, "median_comparable": None,
            "pct_diff": None, "n_comparables": len(pool), "comparables": [],
            "reasoning": (
                f"Only {len(pool)} other {item.get('category', 'similar')} listing(s) "
                "in the catalogue — not enough to judge this price."
            ),
        }

    # Prefer style-tag matches; if at least 3 share a tag, judge against those.
    tagged = [entry for entry in pool if entry[0] > 0]
    basis = "same category and overlapping style tags"
    if len(tagged) >= 3:
        pool = tagged
    else:
        basis = f"other {item.get('category', 'similar')} listings"

    pool.sort(key=lambda entry: (-entry[0], abs(entry[1].get("price", 0) - price)))
    prices = [float(entry[1].get("price", 0)) for entry in pool]
    median = round(statistics.median(prices), 2)
    pct = round(((price - median) / median) * 100, 1) if median else 0.0

    if pct <= -20:
        verdict = "great deal"
    elif pct <= 10:
        verdict = "fair"
    elif pct <= 30:
        verdict = "a bit high"
    else:
        verdict = "overpriced"

    direction = "below" if pct < 0 else "above"
    return {
        "verdict": verdict,
        "item_price": price,
        "median_comparable": median,
        "pct_diff": pct,
        "n_comparables": len(pool),
        "comparables": [
            {
                "title": entry[1].get("title"),
                "price": entry[1].get("price"),
                "condition": entry[1].get("condition"),
            }
            for entry in pool[:3]
        ],
        "reasoning": (
            f"${price:.2f} is {abs(pct):.0f}% {direction} the ${median:.2f} median "
            f"of {len(pool)} comparable listings ({basis})."
        ),
    }


# ── Tool 5 [stretch]: get_trending_styles ─────────────────────────────────────

def get_trending_styles(size: str | None = None, top_n: int = 5) -> dict:
    """
    Surface the style tags that are most common among currently-listed items,
    optionally restricted to the user's size.

    Data source: the live listing feed in data/listings.json — the same feed the
    search tool queries — treated as the resale platforms' current inventory.
    What sellers are listing right now in a given size is the trend signal.
    Listings in excellent condition and on depop (the fastest-moving of the
    three platforms in this dataset) are weighted slightly higher.

    Args:
        size:  Restrict the scan to listings that fit this size; None scans all.
        top_n: How many tags to return.

    Returns:
        dict with keys:
            source (str)            description of the data source
            size_scope (str)        "all sizes" or the size scanned
            trending (list[dict])   [{tag (str), count (int), share (float)}, ...]
            summary (str)           one human-readable sentence
    """
    scope = size if size else "all sizes"
    try:
        listings = load_listings()
    except Exception:
        return {
            "source": "data/listings.json (unavailable)", "size_scope": scope,
            "trending": [],
            "summary": "Couldn't read the listing feed, so no trend data this time.",
        }

    wanted = _normalize_size(size)
    in_scope = [
        listing for listing in listings
        if not wanted or _size_matches(listing.get("size", ""), wanted)
    ]

    if len(in_scope) < 3:
        return {
            "source": "current listing feed (data/listings.json)", "size_scope": scope,
            "trending": [],
            "summary": (
                f"Only {len(in_scope)} listings available in {scope} — too few to "
                "call a trend."
            ),
        }

    weights: dict[str, float] = {}
    for listing in in_scope:
        weight = 1.0
        if listing.get("condition") == "excellent":
            weight += 0.25
        if listing.get("platform") == "depop":
            weight += 0.25
        for tag in listing.get("style_tags") or []:
            weights[tag] = weights.get(tag, 0.0) + weight

    if not weights:
        return {
            "source": "current listing feed (data/listings.json)", "size_scope": scope,
            "trending": [], "summary": f"No style tags found in {scope}.",
        }

    total = sum(weights.values())
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    trending = [
        {"tag": tag, "count": round(weight, 1), "share": round(weight / total * 100, 1)}
        for tag, weight in ranked
    ]
    top_tags = ", ".join(entry["tag"] for entry in trending)
    return {
        "source": "current listing feed (data/listings.json), weighted by condition and platform",
        "size_scope": scope,
        "trending": trending,
        "summary": (
            f"Across {len(in_scope)} listings in {scope}, the most-listed styles "
            f"right now are: {top_tags}."
        ),
    }


# ── Tool 6 [stretch]: style profile memory ────────────────────────────────────

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "data", "style_profile.json")


def _empty_profile(user_id: str = "default") -> dict:
    return {
        "user_id": user_id,
        "size": None,
        "typical_max_price": None,
        "style_tags": [],
        "wardrobe": {"items": []},
        "query_count": 0,
        "updated_at": None,
    }


def load_style_profile(user_id: str = "default") -> dict:
    """
    Load a saved style profile from data/style_profile.json.

    Args:
        user_id: Which profile to load (the app uses "default").

    Returns:
        A profile dict: user_id (str), size (str | None),
        typical_max_price (float | None), style_tags (list[str]),
        wardrobe (dict), query_count (int), updated_at (str | None).
        A missing or corrupt file yields a valid empty profile — never raises.
    """
    try:
        with open(_PROFILE_PATH, "r", encoding="utf-8") as handle:
            store = json.load(handle)
        profile = store.get(user_id)
        if not isinstance(profile, dict):
            return _empty_profile(user_id)
        merged = _empty_profile(user_id)
        merged.update(profile)
        merged["user_id"] = user_id
        return merged
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _empty_profile(user_id)


def save_style_profile(profile: dict, user_id: str = "default") -> bool:
    """
    Persist a style profile to data/style_profile.json.

    Args:
        profile: The profile dict to store.
        user_id: Key to store it under.

    Returns:
        True on success, False if the file could not be written. Never raises —
        losing memory must not break the current run.
    """
    try:
        store = {}
        if os.path.exists(_PROFILE_PATH):
            try:
                with open(_PROFILE_PATH, "r", encoding="utf-8") as handle:
                    store = json.load(handle)
            except (json.JSONDecodeError, OSError):
                store = {}
        to_save = dict(profile)
        to_save["user_id"] = user_id
        to_save["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store[user_id] = to_save
        os.makedirs(os.path.dirname(_PROFILE_PATH), exist_ok=True)
        with open(_PROFILE_PATH, "w", encoding="utf-8") as handle:
            json.dump(store, handle, indent=2)
        return True
    except OSError:
        return False


# ── quick manual check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_empty_wardrobe, get_example_wardrobe

    print(f"Model in use: {_resolve_model()}\n")

    results = search_listings("vintage graphic tee", size=None, max_price=30)
    print(f"search_listings -> {len(results)} results")
    for listing in results[:3]:
        print(f"  {listing['title']} — ${listing['price']} ({listing['match_score']})")

    print("\nimpossible query ->", search_listings("designer ballgown", "XXS", 5))

    top = results[0]
    print("\ncompare_price ->", compare_price(top)["reasoning"])
    print("\nget_trending_styles ->", get_trending_styles(size="M")["summary"])

    outfit = suggest_outfit(top, get_example_wardrobe())
    print(f"\nsuggest_outfit (example wardrobe) ->\n{outfit}")

    print(f"\nsuggest_outfit (empty wardrobe) ->\n{suggest_outfit(top, get_empty_wardrobe())}")

    print(f"\ncreate_fit_card ->\n{create_fit_card(outfit, top)}")
    print(f"\ncreate_fit_card('') ->\n{create_fit_card('', top)}")
