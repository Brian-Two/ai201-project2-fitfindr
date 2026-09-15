"""
Tool-level tests. Run from the repo root with:

    python -m pytest tests/

Use `python -m pytest`, not bare `pytest` — the -m form puts the repo root on
sys.path so `from tools import ...` resolves.

Tests that hit the LLM are marked `llm` and can be skipped with:

    python -m pytest tests/ -m "not llm"
"""

import pytest

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
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe


# ── search_listings ───────────────────────────────────────────────────────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_empty_results():
    """Failure mode: no match. Must be an empty list, never an exception."""
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []


def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=40)
    assert all(item["price"] <= 40 for item in results)


def test_search_size_filter_is_token_based():
    """'M' must not match inside 'XL (oversized)'; 'One Size' fits everyone."""
    results = search_listings("flannel shirt", size="M", max_price=None)
    for item in results:
        size = item["size"].lower()
        assert "xl" not in size or "one size" in size


def test_search_size_filter_matches_compound_sizes():
    results = search_listings("baby tee", size="M", max_price=None)
    assert any(item["size"] == "S/M" for item in results)


def test_search_results_are_sorted_by_relevance():
    results = search_listings("vintage graphic tee", size=None, max_price=None)
    scores = [item["match_score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_search_requires_meaningful_overlap():
    """A listing sharing one incidental word is not a match for a 3-word query."""
    results = search_listings("90s track jacket", size=None, max_price=None)
    titles = [item["title"] for item in results]
    assert any("Track Jacket" in title for title in titles)
    assert not any("Bucket Hat" in title for title in titles)


def test_search_adds_match_metadata():
    results = search_listings("vintage graphic tee", size=None, max_price=None)
    assert "match_score" in results[0]
    assert isinstance(results[0]["matched_terms"], list)


def test_search_empty_description_returns_empty():
    assert search_listings("", None, None) == []


# ── suggest_outfit ────────────────────────────────────────────────────────────

@pytest.mark.llm
def test_suggest_outfit_uses_wardrobe():
    item = search_listings("vintage graphic tee", None, 50)[0]
    result = suggest_outfit(item, get_example_wardrobe())
    assert isinstance(result, str)
    assert result.strip()
    assert not result.startswith(OUTFIT_UNAVAILABLE)


@pytest.mark.llm
def test_suggest_outfit_empty_wardrobe():
    """Failure mode: empty wardrobe. Must return advice, not raise or return ''."""
    item = search_listings("vintage graphic tee", None, 50)[0]
    result = suggest_outfit(item, get_empty_wardrobe())
    assert isinstance(result, str)
    assert len(result.strip()) > 40
    assert "haven't added any wardrobe pieces" in result


def test_suggest_outfit_with_no_item():
    """Guard: a missing item is reported, not crashed on."""
    result = suggest_outfit({}, get_example_wardrobe())
    assert result.startswith(OUTFIT_UNAVAILABLE)


def test_suggest_outfit_tolerates_bad_wardrobe_shape():
    item = search_listings("vintage graphic tee", None, 50)[0]
    result = suggest_outfit(item, None)  # type: ignore[arg-type]
    assert isinstance(result, str) and result.strip()


# ── create_fit_card ───────────────────────────────────────────────────────────

def test_fit_card_empty_outfit_returns_error_string():
    """Failure mode: empty outfit. Must return a string, never raise."""
    item = search_listings("vintage graphic tee", None, 50)[0]
    result = create_fit_card("", item)
    assert isinstance(result, str)
    assert result.startswith(FIT_CARD_UNAVAILABLE)


def test_fit_card_whitespace_outfit_returns_error_string():
    item = search_listings("vintage graphic tee", None, 50)[0]
    assert create_fit_card("   \n  ", item).startswith(FIT_CARD_UNAVAILABLE)


def test_fit_card_refuses_unavailable_outfit():
    """An outfit that failed upstream must not be captioned as if it succeeded."""
    item = search_listings("vintage graphic tee", None, 50)[0]
    result = create_fit_card(f"{OUTFIT_UNAVAILABLE} model down", item)
    assert result.startswith(FIT_CARD_UNAVAILABLE)


def test_fit_card_missing_item_returns_error_string():
    assert create_fit_card("jeans and sneakers", {}).startswith(FIT_CARD_UNAVAILABLE)


@pytest.mark.llm
def test_fit_card_varies_across_calls():
    """Same input, different captions — the caption must not be canned."""
    item = search_listings("vintage graphic tee", None, 50)[0]
    outfit = "Wear it with baggy dark-wash jeans and chunky white sneakers."
    cards = {create_fit_card(outfit, item) for _ in range(3)}
    assert len(cards) > 1


@pytest.mark.llm
def test_fit_card_differs_for_different_items():
    results = search_listings("vintage", None, None)
    outfit = "Wear it with baggy dark-wash jeans and chunky white sneakers."
    assert create_fit_card(outfit, results[0]) != create_fit_card(outfit, results[1])


# ── compare_price [stretch] ───────────────────────────────────────────────────

def test_compare_price_returns_verdict_and_reasoning():
    item = search_listings("vintage graphic tee", None, 50)[0]
    check = compare_price(item)
    assert check["verdict"] in {"great deal", "fair", "a bit high", "overpriced", "unknown"}
    assert check["reasoning"]
    assert check["n_comparables"] >= 0


def test_compare_price_flags_an_expensive_item():
    bomber = [i for i in search_listings("leather bomber", None, None)
              if "Leather Bomber" in i["title"]][0]
    check = compare_price(bomber)
    assert check["verdict"] in {"a bit high", "overpriced"}
    assert check["median_comparable"] < bomber["price"]


def test_compare_price_handles_missing_item():
    """Failure mode: nothing to compare. Returns 'unknown', does not raise."""
    check = compare_price({})
    assert check["verdict"] == "unknown"
    assert check["median_comparable"] is None


# ── get_trending_styles [stretch] ─────────────────────────────────────────────

def test_trending_styles_returns_tags():
    trends = get_trending_styles()
    assert trends["trending"]
    assert all({"tag", "count", "share"} <= set(entry) for entry in trends["trending"])
    assert trends["source"]


def test_trending_styles_respects_size_scope():
    trends = get_trending_styles(size="M")
    assert trends["size_scope"] == "M"
    assert isinstance(trends["trending"], list)


def test_trending_styles_one_size_items_count_for_every_size():
    """"One Size" listings fit any requested size, so XXS is not an empty scope."""
    trends = get_trending_styles(size="XXS")
    assert isinstance(trends["trending"], list)
    assert trends["size_scope"] == "XXS"


def test_trending_styles_thin_size_scope(monkeypatch):
    """Failure mode: too few listings to call a trend — empty list, no crash."""
    import tools
    monkeypatch.setattr(
        tools, "load_listings",
        lambda: [{"size": "M", "style_tags": ["y2k"], "condition": "good",
                  "platform": "depop"}],
    )
    trends = get_trending_styles(size="M")
    assert trends["trending"] == []
    assert "too few" in trends["summary"]


def test_trending_styles_survives_unreadable_data(monkeypatch):
    """Failure mode: the feed can't be read at all — report it, don't raise."""
    import tools

    def boom():
        raise OSError("no data file")

    monkeypatch.setattr(tools, "load_listings", boom)
    trends = get_trending_styles()
    assert trends["trending"] == []
    assert "Couldn't read" in trends["summary"]


# ── style profile [stretch] ───────────────────────────────────────────────────

def test_style_profile_roundtrip(tmp_path, monkeypatch):
    import tools
    monkeypatch.setattr(tools, "_PROFILE_PATH", str(tmp_path / "profile.json"))

    assert load_style_profile("tester")["size"] is None      # missing file is fine
    assert save_style_profile({"size": "M", "style_tags": ["y2k"]}, "tester")

    loaded = load_style_profile("tester")
    assert loaded["size"] == "M"
    assert loaded["style_tags"] == ["y2k"]
    assert loaded["updated_at"]


def test_style_profile_corrupt_file_returns_empty(tmp_path, monkeypatch):
    """Failure mode: unreadable profile. Degrade to empty, never raise."""
    import tools
    bad = tmp_path / "profile.json"
    bad.write_text("{not json at all")
    monkeypatch.setattr(tools, "_PROFILE_PATH", str(bad))

    profile = load_style_profile("tester")
    assert profile["size"] is None
    assert profile["style_tags"] == []
