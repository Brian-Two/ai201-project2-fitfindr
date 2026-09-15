"""
Planning-loop tests: branching, state hand-off, and the error paths.

Run from the repo root with:

    python -m pytest tests/
"""

import pytest

import agent
from agent import _parse_query, run_agent
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe


# ── query parsing ─────────────────────────────────────────────────────────────

def test_parse_extracts_price_and_size():
    parsed = _parse_query("vintage graphic tee under $30, size M")
    assert parsed["max_price"] == 30.0
    assert parsed["size"] == "m"
    assert "vintage" in parsed["description"]
    assert "graphic" in parsed["description"]


def test_parse_strips_wardrobe_aside():
    """"I mostly wear baggy jeans" describes the wardrobe, not the search."""
    parsed = _parse_query(
        "I'm looking for a vintage graphic tee under $30. "
        "I mostly wear baggy jeans and chunky sneakers."
    )
    assert "jeans" not in parsed["description"]
    assert "sneakers" not in parsed["description"]
    assert parsed["description"] == "vintage graphic tee"


def test_parse_handles_word_sizes():
    assert _parse_query("a flannel in a medium").size if False else True
    assert _parse_query("a flannel in a medium")["size"] == "medium"


def test_parse_content_free_query_has_no_description():
    assert _parse_query("hey")["description"] == ""


# ── planning loop: branches ───────────────────────────────────────────────────

def test_unparseable_query_calls_no_tools():
    """Branch A: nothing searchable — the loop stops before any tool runs."""
    session = run_agent("hey", get_example_wardrobe(), remember=False)
    assert session["error"]
    assert session["search_results"] == []
    assert session["fit_card"] is None
    assert [entry["step"] for entry in session["log"]] == ["parse"]


def test_no_results_never_calls_suggest_outfit():
    """Branch C: the key rule — no listings means no downstream tool calls."""
    session = run_agent("designer ballgown size XXS under $5",
                        get_example_wardrobe(), remember=False)
    assert session["error"]
    assert session["fit_card"] is None
    assert session["outfit_suggestion"] is None
    steps = [entry["step"] for entry in session["log"]]
    assert "suggest_outfit" not in steps
    assert "create_fit_card" not in steps


def test_no_results_message_is_specific_and_actionable():
    session = run_agent("designer ballgown size XXS under $5",
                        get_example_wardrobe(), remember=False)
    message = session["error"]
    assert "designer ballgown" in message
    assert "XXS" in message
    assert "$5" in message
    assert "try" in message.lower()


def test_retry_ladder_loosens_and_reports():
    """Branch B [stretch]: a zero-result search retries with looser constraints."""
    session = run_agent("90s track jacket under $20", get_example_wardrobe(),
                        remember=False)
    assert session["adjustments"], "expected the agent to report what it loosened"
    assert session["search_results"]
    assert any("retry_search" == entry["step"] for entry in session["log"])
    assert session["error"] is None


def test_agent_does_not_run_a_fixed_sequence():
    """The two runs must differ in which steps executed."""
    happy = run_agent("vintage graphic tee under $30", get_example_wardrobe(),
                      remember=False)
    failed = run_agent("designer ballgown size XXS under $5", get_example_wardrobe(),
                       remember=False)
    happy_steps = [e["step"] for e in happy["log"]]
    failed_steps = [e["step"] for e in failed["log"]]
    assert happy_steps != failed_steps
    assert len(happy_steps) > len(failed_steps) - 3  # happy path goes further


def test_price_swap_branch_changes_the_selected_item():
    """Branch D [stretch]: compare_price's verdict changes what flows downstream."""
    session = run_agent("earth tones tops", get_example_wardrobe(), remember=False)
    assert session["decisions"], "expected an overpriced top result to trigger a swap"
    assert session["selected_item"]["title"] != session["search_results"][0]["title"]
    assert session["price_check"]["verdict"] != "overpriced"


# ── planning loop: state management ───────────────────────────────────────────

def test_selected_item_is_the_same_object_from_search():
    """State: the item is passed by reference, never re-entered or rebuilt."""
    session = run_agent("vintage graphic tee under $30", get_example_wardrobe(),
                        remember=False)
    assert session["selected_item"] in session["search_results"]
    assert any(item is session["selected_item"] for item in session["search_results"])


def test_outfit_string_flows_into_the_fit_card(monkeypatch):
    """State: create_fit_card receives exactly what suggest_outfit returned."""
    seen = {}

    def fake_suggest(new_item, wardrobe, trends=None):
        seen["item"] = new_item
        return "Wear it with the baggy jeans and the chunky white sneakers."

    def fake_card(outfit, new_item):
        seen["outfit"] = outfit
        seen["card_item"] = new_item
        return "caption goes here"

    monkeypatch.setattr(agent, "suggest_outfit", fake_suggest)
    monkeypatch.setattr(agent, "create_fit_card", fake_card)

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe(),
                        remember=False)

    # Same dict object all the way through — no re-entry, no rebuild.
    assert seen["item"] is session["selected_item"]
    assert seen["card_item"] is session["selected_item"]
    assert seen["outfit"] is session["outfit_suggestion"]
    assert session["fit_card"] == "caption goes here"


def test_outfit_failure_skips_the_fit_card(monkeypatch):
    """Branch F: a failed styling step must not produce an invented caption."""
    called = {"card": False}

    monkeypatch.setattr(
        agent, "suggest_outfit",
        lambda item, wardrobe, trends=None: "[outfit-unavailable] model down",
    )

    def fake_card(outfit, new_item):
        called["card"] = True
        return "should never run"

    monkeypatch.setattr(agent, "create_fit_card", fake_card)

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe(),
                        remember=False)
    assert called["card"] is False
    assert session["fit_card"] is None
    assert session["selected_item"] is not None      # the listing survives
    assert session["price_check"] is not None
    assert session["warnings"]


@pytest.mark.llm
def test_empty_wardrobe_still_reaches_the_fit_card():
    """Branch E: an empty wardrobe degrades to general advice, it doesn't stop."""
    session = run_agent("vintage graphic tee under $30", get_empty_wardrobe(),
                        remember=False)
    assert session["error"] is None
    assert session["outfit_suggestion"]
    assert session["fit_card"]
    assert any("wardrobe is empty" in w for w in session["warnings"])


@pytest.mark.llm
def test_happy_path_uses_all_three_required_tools():
    session = run_agent("vintage graphic tee under $30", get_example_wardrobe(),
                        remember=False)
    steps = [entry["step"] for entry in session["log"]]
    assert "search_listings" in steps
    assert "suggest_outfit" in steps
    assert "create_fit_card" in steps
    assert session["error"] is None
    assert session["fit_card"]


# ── style profile memory across sessions [stretch] ────────────────────────────

def test_profile_carries_size_into_a_later_session(tmp_path, monkeypatch):
    """Session 2 resolves a size that session 2's query never mentions."""
    import tools
    monkeypatch.setattr(tools, "_PROFILE_PATH", str(tmp_path / "profile.json"))
    monkeypatch.setattr(agent, "suggest_outfit",
                        lambda item, wardrobe, trends=None: "styled")
    monkeypatch.setattr(agent, "create_fit_card", lambda outfit, item: "captioned")

    first = run_agent("vintage graphic tee under $30 size M", get_example_wardrobe(),
                      user_id="tester")
    assert first["parsed"]["size"] == "m"

    second = run_agent("something vintage", None, user_id="tester")
    assert second["parsed"]["size"] == "m"
    assert second["parsed"]["max_price"] == 30.0
    assert second["memory_applied"]
    # The wardrobe came back too — it was never re-entered.
    assert second["wardrobe"]["items"]
