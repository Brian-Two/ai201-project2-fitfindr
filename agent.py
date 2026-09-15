"""
agent.py

The FitFindr planning loop. Orchestrates the tools in response to a natural
language user query, passing state between them via a session dict.

The loop is a small state machine: each iteration runs one step, looks at what
that step returned, and writes session["next_action"] accordingly. Which tools
run — and in what order — therefore depends on the data, not on a fixed script.

Usage:
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import re

from tools import (
    FIT_CARD_UNAVAILABLE,
    OUTFIT_UNAVAILABLE,
    compare_price,
    create_fit_card,
    get_trending_styles,
    load_style_profile,
    save_style_profile,
    search_listings,
    suggest_outfit,
)
from tools import _tokenize  # shared tokenizer, so parse and search agree

MAX_STEPS = 12  # safety net against a state-machine bug; never hit in normal flow

# Price verdicts from best to worst, for comparing two listings' value.
_VERDICT_RANK = {"great deal": 0, "fair": 1, "a bit high": 2, "overpriced": 3, "unknown": 4}


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # chosen result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "error": None,               # set if the interaction ended early

        # --- added for the planning loop and the UI trace ---
        "price_check": None,         # dict returned by compare_price [stretch]
        "trends": None,              # list[str] from get_trending_styles [stretch]
        "warnings": [],              # non-fatal problems worth telling the user
        "adjustments": [],           # constraints the agent loosened, in words
        "decisions": [],             # choices the agent made and why
        "memory_applied": [],        # values filled in from the style profile
        "log": [],                   # ordered (step, summary) trace of the run
        "next_action": "parse",      # the state variable the while loop runs on
    }


def _log(session: dict, step: str, summary: str) -> None:
    """Record one planning-loop step so the run can be traced afterwards."""
    session["log"].append({"step": step, "summary": summary})


# ── query parsing ─────────────────────────────────────────────────────────────

_PRICE_PATTERNS = [
    r"under\s*\$?\s*(\d+(?:\.\d+)?)",
    r"below\s*\$?\s*(\d+(?:\.\d+)?)",
    r"less\s+than\s*\$?\s*(\d+(?:\.\d+)?)",
    r"(?:max|maximum|budget(?:\s+of)?|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)",
    r"\$\s*(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd)",
]

_SIZE_PATTERNS = [
    r"\bsize\s*:?\s*([a-z]{1,3}\b|\d{1,2}(?:\.5)?\b|w\d{2}\b)",
    r"\bin\s+(?:a\s+)?(x-?small|small|medium|large|x-?large)\b",
    r"\b(xxs|xs|s|m|l|xl|xxl)\s+size\b",
]

# Clauses like "I mostly wear baggy jeans" describe the wardrobe, not the thing
# being searched for — they must not leak into the search keywords.
_WARDROBE_ASIDE = re.compile(
    r"\b(?:i\s+(?:mostly|usually|normally|always|generally)?\s*(?:wear|own|have|live\s+in)|"
    r"my\s+(?:go-to|usual|style\s+is))\b[^.!?;]*[.!?;]?",
    re.IGNORECASE,
)


def _parse_query(query: str) -> dict:
    """
    Extract search constraints from a natural language query.

    Deterministic regex + stop-word removal rather than an LLM call: parsing is
    the one step that must not fail or hallucinate, and it runs on every query.

    Returns:
        dict with description (str), size (str | None), max_price (float | None).
    """
    text = (query or "").strip()
    working = text.lower()

    max_price = None
    for pattern in _PRICE_PATTERNS:
        match = re.search(pattern, working)
        if match:
            max_price = float(match.group(1))
            working = working[: match.start()] + " " + working[match.end():]
            break

    size = None
    for pattern in _SIZE_PATTERNS:
        match = re.search(pattern, working)
        if match:
            size = match.group(1).strip()
            working = working[: match.start()] + " " + working[match.end():]
            break

    working = _WARDROBE_ASIDE.sub(" ", working)
    description = " ".join(_tokenize(working))

    return {"description": description, "size": size, "max_price": max_price}


def _relaxation_ladder(parsed: dict) -> list[dict]:
    """
    Build the ordered fallback searches used when the first search finds nothing.

    Each rung is {"label": <what to tell the user>, "kwargs": <search args>}.
    Rungs that would not actually change anything are skipped.
    """
    description = parsed["description"]
    size = parsed["size"]
    max_price = parsed["max_price"]
    ladder = []

    if size:
        ladder.append({
            "label": f"couldn't find it in size {size.upper()}, so I searched every size",
            "kwargs": {"description": description, "size": None, "max_price": max_price},
        })

    if max_price:
        loosened = round((max_price * 1.5) / 5) * 5 or max_price + 5
        ladder.append({
            "label": (
                f"nothing came up under ${max_price:.0f}, so I stretched the budget "
                f"to ${loosened:.0f}"
            ),
            "kwargs": {"description": description, "size": None, "max_price": loosened},
        })

    tokens = description.split()
    if len(tokens) > 1:
        keyword = tokens[-1]  # the head noun: "vintage graphic tee" -> "tee"
        ladder.append({
            "label": f"broadened the search to just '{keyword}'",
            "kwargs": {"description": keyword, "size": None, "max_price": None},
        })

    return ladder


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    wardrobe: dict | None = None,
    user_id: str = "default",
    remember: bool = True,
) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py. None falls
                  back to the wardrobe saved in the style profile. [stretch]
        user_id:  Which saved style profile to read and update. [stretch]
        remember: False disables all style-profile reads and writes. [stretch]

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.
    """
    profile = load_style_profile(user_id) if remember else None

    if wardrobe is None:
        wardrobe = (profile or {}).get("wardrobe") or {"items": []}

    session = _new_session(query, wardrobe)
    ladder: list[dict] = []
    steps = 0

    while session["next_action"] != "done" and steps < MAX_STEPS:
        steps += 1
        action = session["next_action"]

        # ── 1. parse ──────────────────────────────────────────────────────────
        if action == "parse":
            parsed = _parse_query(query)

            # [stretch] fill gaps from remembered preferences
            if profile:
                if not parsed["size"] and profile.get("size"):
                    parsed["size"] = profile["size"]
                    session["memory_applied"].append(
                        f"size {profile['size'].upper()} (remembered from a previous session)"
                    )
                if not parsed["max_price"] and profile.get("typical_max_price"):
                    parsed["max_price"] = float(profile["typical_max_price"])
                    session["memory_applied"].append(
                        f"budget ${profile['typical_max_price']:.0f} (your usual)"
                    )
                if not parsed["description"] and profile.get("style_tags"):
                    parsed["description"] = " ".join(profile["style_tags"][:3])
                    session["memory_applied"].append(
                        f"styles you've searched before ({parsed['description']})"
                    )

            session["parsed"] = parsed

            if not parsed["description"]:
                # Branch A: nothing searchable. No tool is called at all.
                session["error"] = (
                    "I couldn't tell what you're looking for from that. Try naming the "
                    "piece and a style — for example \"vintage graphic tee under $30, "
                    "size M\" or \"90s track jacket\"."
                )
                _log(session, "parse", "no usable keywords — stopped before any tool ran")
                session["next_action"] = "done"
                continue

            _log(
                session, "parse",
                f"description='{parsed['description']}', size={parsed['size']}, "
                f"max_price={parsed['max_price']}",
            )
            session["next_action"] = "search"
            continue

        # ── 2. search ─────────────────────────────────────────────────────────
        if action == "search":
            parsed = session["parsed"]
            results = search_listings(
                parsed["description"], parsed["size"], parsed["max_price"]
            )
            session["search_results"] = results

            if results:
                _log(
                    session, "search_listings",
                    f"{len(results)} match(es); top = {results[0]['title']} "
                    f"(${results[0]['price']:.2f})",
                )
                session["next_action"] = "evaluate"
                continue

            # Empty. Build the ladder once, then start climbing it.
            if not ladder:
                ladder = _relaxation_ladder(parsed)
            if ladder:
                _log(session, "search_listings", "0 matches — loosening constraints")
                session["next_action"] = "retry_search"
                continue

            # Branch C: nothing left to loosen. Stop before suggest_outfit.
            session["error"] = _no_results_message(parsed, session["adjustments"])
            _log(
                session, "search_listings",
                "0 matches after every fallback — stopped; suggest_outfit never called",
            )
            session["next_action"] = "done"
            continue

        # ── 3. retry_search [stretch: retry logic with fallback] ──────────────
        if action == "retry_search":
            rung = ladder.pop(0)
            results = search_listings(**rung["kwargs"])
            session["adjustments"].append(rung["label"])

            if results:
                session["search_results"] = results
                session["parsed"] = {**session["parsed"], **rung["kwargs"]}
                _log(
                    session, "retry_search",
                    f"{rung['label']} → {len(results)} match(es)",
                )
                session["next_action"] = "evaluate"
            else:
                _log(session, "retry_search", f"{rung['label']} → still nothing")
                session["next_action"] = "search" if ladder else "give_up"
            continue

        if action == "give_up":
            session["error"] = _no_results_message(session["parsed"], session["adjustments"])
            _log(
                session, "give_up",
                "every fallback exhausted — stopped; suggest_outfit never called",
            )
            session["next_action"] = "done"
            continue

        # ── 4. evaluate (+ price check) ───────────────────────────────────────
        if action == "evaluate":
            results = session["search_results"]
            selected = results[0]
            check = compare_price(selected, results)          # [stretch]

            # Branch D: a tool's return value changes which item flows on.
            # The alternative has to answer the query at least as completely as
            # the top result (matching a leather jacket query with a denim one,
            # or with a leather belt, would be cheaper and useless), be the same
            # kind of garment, and be meaningfully cheaper.
            if check.get("verdict") == "overpriced":
                top_terms = set(selected.get("matched_terms") or [])
                best = None
                for item in results[1:6]:
                    if item.get("category") != selected.get("category"):
                        continue
                    if item.get("price", 0) > 0.85 * selected.get("price", 0):
                        continue
                    if not set(item.get("matched_terms") or []) >= top_terms:
                        continue
                    item_check = compare_price(item, results)
                    # The swap has to actually improve the price verdict, not
                    # just be cheaper — otherwise we'd recommend a second
                    # overpriced item and call it a saving.
                    if _VERDICT_RANK.get(item_check.get("verdict"), 4) >= _VERDICT_RANK.get(
                        check.get("verdict"), 4
                    ):
                        continue
                    ranking = (item["match_score"], -item["price"])
                    if best is None or ranking > best[2]:
                        best = (item, item_check, ranking)

                if best:
                    swapped, check = best[0], best[1]
                    session["decisions"].append(
                        f"The closest match ({selected['title']}, "
                        f"${selected['price']:.2f}) came out overpriced, so I switched to "
                        f"{swapped['title']} at ${swapped['price']:.2f} — it answers the "
                        f"same search and rates as a {check['verdict']}."
                    )
                    selected = swapped

            session["selected_item"] = selected
            session["price_check"] = check
            _log(
                session, "compare_price",
                f"{selected['title']} → {check.get('verdict')} ({check.get('reasoning')})",
            )
            session["next_action"] = "trends"
            continue

        # ── 5. trends [stretch] ───────────────────────────────────────────────
        if action == "trends":
            trend_data = get_trending_styles(size=session["parsed"].get("size"))
            tags = [entry["tag"] for entry in trend_data.get("trending", [])]
            session["trends"] = tags or None
            if tags:
                _log(session, "get_trending_styles", trend_data["summary"])
            else:
                _log(session, "get_trending_styles", "no trend signal — continuing without it")
            session["next_action"] = "outfit"
            continue

        # ── 6. outfit ─────────────────────────────────────────────────────────
        if action == "outfit":
            # The item dict handed over here is the same object search_listings
            # returned — the user never re-enters it.
            suggestion = suggest_outfit(
                session["selected_item"], session["wardrobe"], session["trends"]
            )
            session["outfit_suggestion"] = suggestion

            if suggestion.startswith(OUTFIT_UNAVAILABLE):
                # Branch F: degrade, don't crash. Keep the listing + price check,
                # skip the caption rather than inventing one.
                session["warnings"].append(
                    "I found the item and checked the price, but couldn't reach the "
                    "styling model, so there's no outfit or fit card this time. "
                    "Try again in a moment."
                )
                session["outfit_suggestion"] = None
                _log(session, "suggest_outfit", "unavailable — skipping create_fit_card")
                session["next_action"] = "done"
                continue

            if not session["wardrobe"].get("items"):
                # Branch E: empty wardrobe — same tool, different output path.
                session["warnings"].append(
                    "Your wardrobe is empty, so this is general styling advice. Add a "
                    "few pieces and I'll build outfits from what you actually own."
                )
                _log(session, "suggest_outfit", "empty wardrobe → general styling advice")
            else:
                _log(
                    session, "suggest_outfit",
                    f"styled against {len(session['wardrobe']['items'])} wardrobe pieces",
                )
            session["next_action"] = "fit_card"
            continue

        # ── 7. fit_card ───────────────────────────────────────────────────────
        if action == "fit_card":
            # Both arguments come straight out of the session.
            card = create_fit_card(session["outfit_suggestion"], session["selected_item"])
            if card.startswith(FIT_CARD_UNAVAILABLE):
                session["warnings"].append(card.replace(FIT_CARD_UNAVAILABLE, "").strip())
                session["fit_card"] = None
                _log(session, "create_fit_card", "unavailable — reported to the user")
            else:
                session["fit_card"] = card
                _log(session, "create_fit_card", f"caption written ({len(card)} chars)")
            session["next_action"] = "done"
            continue

        # Unknown state — should be unreachable.
        session["error"] = f"Internal planning error: unknown step '{action}'."
        session["next_action"] = "done"

    if steps >= MAX_STEPS and session["next_action"] != "done":
        session["error"] = "The agent ran too many steps without finishing. Try a simpler query."

    if remember:
        _update_profile(session, profile, user_id)

    return session


def _no_results_message(parsed: dict, adjustments: list[str]) -> str:
    """
    Build the specific, actionable no-results message.

    Names the constraints that were tried, what was already loosened, and the
    closest real thing in the catalogue so the user has somewhere to go next.
    """
    bits = [f"\"{parsed.get('description', '')}\""]
    if parsed.get("size"):
        bits.append(f"size {str(parsed['size']).upper()}")
    if parsed.get("max_price"):
        bits.append(f"under ${parsed['max_price']:.0f}")
    tried = ", ".join(bits)

    message = f"I couldn't find anything matching {tried} in the catalogue."
    if adjustments:
        message += " I also tried: " + "; ".join(adjustments) + " — still nothing."

    # Offer the nearest real data point rather than a generic apology.
    nearest = search_listings(parsed.get("description", ""), None, None)
    if nearest:
        cheapest = min(nearest, key=lambda item: item["price"])
        message += (
            f" The closest thing here is \"{cheapest['title']}\" at "
            f"${cheapest['price']:.2f} ({cheapest['size']}) — try raising your budget "
            "or dropping the size filter."
        )
    else:
        message += (
            " Nothing in the catalogue uses those words at all. Try different keywords "
            "(the catalogue covers tops, bottoms, outerwear, shoes and accessories in "
            "vintage, y2k, grunge, streetwear and cottagecore styles)."
        )
    return message


def _update_profile(session: dict, profile: dict | None, user_id: str) -> None:
    """[stretch] Fold this run's preferences into the saved style profile."""
    if profile is None:
        return
    parsed = session.get("parsed") or {}
    if parsed.get("size"):
        profile["size"] = parsed["size"]
    if parsed.get("max_price"):
        profile["typical_max_price"] = float(parsed["max_price"])
    if session.get("wardrobe", {}).get("items"):
        profile["wardrobe"] = session["wardrobe"]
    if session.get("selected_item"):
        tags = list(profile.get("style_tags") or [])
        for tag in session["selected_item"].get("style_tags") or []:
            if tag not in tags:
                tags.append(tag)
        profile["style_tags"] = tags[:12]
    profile["query_count"] = int(profile.get("query_count") or 0) + 1

    if not save_style_profile(profile, user_id):
        session["warnings"].append(
            "Couldn't save your style profile this time — preferences won't carry over."
        )


# ── CLI test ──────────────────────────────────────────────────────────────────

def _print_session(session: dict) -> None:
    for entry in session["log"]:
        print(f"  [{entry['step']}] {entry['summary']}")
    for note in session["adjustments"]:
        print(f"  ! adjusted: {note}")
    for note in session["decisions"]:
        print(f"  > decision: {note}")
    for note in session["warnings"]:
        print(f"  ! warning: {note}")
    if session["error"]:
        print(f"\nError: {session['error']}")
        print(f"fit_card is {session['fit_card']}")
        return
    print(f"\nFound: {session['selected_item']['title']} — ${session['selected_item']['price']:.2f}")
    print(f"Price check: {session['price_check']['verdict']} — {session['price_check']['reasoning']}")
    print(f"\nOutfit: {session['outfit_suggestion']}")
    print(f"\nFit card: {session['fit_card']}")


if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    _print_session(run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
        remember=False,
    ))

    print("\n\n=== No-results path ===\n")
    _print_session(run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
        remember=False,
    ))

    print("\n\n=== Retry path: loosened constraints ===\n")
    _print_session(run_agent(
        query="90s track jacket under $20 size M",
        wardrobe=get_example_wardrobe(),
        remember=False,
    ))

    print("\n\n=== Price-swap path: top result is overpriced ===\n")
    _print_session(run_agent(
        query="earth tones tops",
        wardrobe=get_example_wardrobe(),
        remember=False,
    ))

    print("\n\n=== Empty wardrobe path ===\n")
    _print_session(run_agent(
        query="vintage graphic tee under $30",
        wardrobe=get_empty_wardrobe(),
        remember=False,
    ))

    print("\n\n=== Unparseable query path ===\n")
    _print_session(run_agent(query="hey", wardrobe=get_example_wardrobe(), remember=False))
