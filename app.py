"""
app.py

Gradio interface for FitFindr. handle_query() calls run_agent() and maps the
resulting session dict onto four output panels: the chosen listing (with its
price check), the outfit idea, the fit card, and a trace of which tools the
planning loop actually ran for that query.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── formatting helpers ────────────────────────────────────────────────────────

def _format_listing(session: dict) -> str:
    """Render the chosen listing plus the price check for the first panel."""
    item = session["selected_item"]
    lines = [
        item["title"],
        f"${item['price']:.2f} · {item['platform']} · {item['condition']} condition",
        f"size {item['size']} · {item['category']}",
        f"style: {', '.join(item.get('style_tags') or [])}",
        "",
        item["description"],
    ]

    check = session.get("price_check") or {}
    if check.get("verdict") and check["verdict"] != "unknown":
        lines += ["", f"💰 Price check: {check['verdict'].upper()}", check["reasoning"]]
        for comp in check.get("comparables", [])[:3]:
            lines.append(f"   · {comp['title']} — ${comp['price']:.2f} ({comp['condition']})")
    elif check.get("reasoning"):
        lines += ["", f"💰 Price check: {check['reasoning']}"]

    others = len(session.get("search_results", [])) - 1
    if others > 0:
        lines += ["", f"({others} other match{'es' if others > 1 else ''} found)"]
    return "\n".join(lines)


def _format_trace(session: dict) -> str:
    """
    Render the planning-loop trace: which tools ran, in what order, and why.

    This is what makes the agent's branching visible instead of implied.
    """
    lines = ["🧭 What the agent did:"]
    for i, entry in enumerate(session.get("log", []), start=1):
        lines.append(f"{i}. {entry['step']} — {entry['summary']}")

    for note in session.get("memory_applied", []):
        lines.append(f"🧠 Remembered: {note}")
    for note in session.get("adjustments", []):
        lines.append(f"🔁 Adjusted: {note}")
    for note in session.get("decisions", []):
        lines.append(f"🤔 Decision: {note}")
    for note in session.get("warnings", []):
        lines.append(f"⚠️  {note}")

    parsed = session.get("parsed") or {}
    if parsed:
        lines.append(
            f"\nParsed query → description={parsed.get('description')!r}, "
            f"size={parsed.get('size')}, max_price={parsed.get('max_price')}"
        )
    if session.get("selected_item"):
        # Proof that one object flows through the whole run without re-entry.
        lines.append(
            f"State handoff → search_listings returned "
            f"'{session['selected_item']['title']}' (id "
            f"{session['selected_item']['id']}); the same dict was passed to "
            "suggest_outfit, and its output string was passed to create_fit_card."
        )
    return "\n".join(lines)


# ── query handler ─────────────────────────────────────────────────────────────

def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:     The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of four strings:
            (listing_text, outfit_suggestion, fit_card, agent_trace)
        Each string maps to one of the four output panels in the UI.
    """
    if not user_query or not user_query.strip():
        return (
            "Type what you're looking for first — for example "
            "\"vintage graphic tee under $30, size M\".",
            "", "", "",
        )

    wardrobe = (
        get_empty_wardrobe()
        if wardrobe_choice == "Empty wardrobe (new user)"
        else get_example_wardrobe()
    )

    session = run_agent(query=user_query.strip(), wardrobe=wardrobe)
    trace = _format_trace(session)

    # Error branch: the run stopped early, so the downstream panels stay empty
    # rather than showing stale or invented content.
    if session["error"]:
        return f"❌ {session['error']}", "", "", trace

    outfit = session["outfit_suggestion"] or ""
    fit_card = session["fit_card"] or ""

    # Degraded branch: item found, styling unavailable. Say so in the panel
    # instead of leaving the user staring at an empty box.
    if not outfit and session["warnings"]:
        outfit = "⚠️ " + " ".join(session["warnings"])
    if not fit_card and not session["outfit_suggestion"]:
        fit_card = "⚠️ No fit card — there's no outfit to caption yet."

    return _format_listing(session), outfit, fit_card, trace


# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers.",
    "90s track jacket in size M",
    "a leather jacket",                      # triggers the price-driven swap
    "90s track jacket under $20",            # triggers the retry ladder
    "designer ballgown size XXS under $5",   # deliberate no-results test
    "hey",                                   # deliberate unparseable test
]

def build_interface():
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        trace_output = gr.Textbox(
            label="🧭 Agent trace (which tools ran, and why)",
            lines=10,
            interactive=False,
        )

        gr.Examples(
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output, trace_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output, trace_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
