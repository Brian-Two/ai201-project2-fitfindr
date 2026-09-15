# FitFindr 🛍️

A multi-tool AI agent that finds secondhand pieces and figures out how to wear
them. Type one natural language request — *"I'm looking for a vintage graphic
tee under $30. I mostly wear baggy jeans and chunky sneakers."* — and a planning
loop decides which tools to call, carries results between them, and degrades
gracefully when something breaks.

**Demo video:** [3–5 min walkthrough](ADD_YOUR_VIDEO_LINK_HERE) · **Spec:** [planning.md](planning.md)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

Create a `.env` in the repo root (it's gitignored):

```
GROQ_API_KEY=your_key_here
```

Run it:

```bash
python app.py            # Gradio UI — open the URL printed in your terminal
python agent.py          # CLI: walks all five planning-loop branches
python tools.py          # CLI: exercises each tool on its own
python -m pytest tests/  # 45 tests (use -m "not llm" to skip live API calls)
```

> Use `python -m pytest`, not bare `pytest` — the `-m` form puts the repo root
> on `sys.path` so `from tools import ...` resolves.

---

## Tool Inventory

Six tools: the three required ones, plus three built for the stretch features.
Signatures below are copied from `tools.py`.

### 1. `search_listings(description, size, max_price) -> list[dict]`

Filters and ranks the 40-listing mock dataset. Pure Python, no LLM call.

| Input | Type | Meaning |
|---|---|---|
| `description` | `str` | Keywords, e.g. `"vintage graphic tee"`. Tokenised, stop-words dropped, matched against each listing's title, description, style_tags, category, colors and brand. |
| `size` | `str \| None` | Size filter. `None` skips it. Case-insensitive and token-based, so `"M"` matches `"M"`, `"S/M"`, `"M/L"` and `"One Size"`, but **not** `"XL (oversized)"`. |
| `max_price` | `float \| None` | Inclusive price ceiling. `None` skips it. |

**Returns:** a list of listing dicts sorted best-match-first (cheaper wins a
tie). Each dict carries the dataset fields — `id` (str), `title` (str),
`description` (str), `category` (str), `style_tags` (list[str]), `size` (str),
`condition` (str), `price` (float), `colors` (list[str]), `brand` (str | None),
`platform` (str) — plus two fields this tool adds: `match_score` (float) and
`matched_terms` (list[str], which query words the listing actually matched).
Returns `[]` when nothing matches.

**Purpose:** turn a vague request into a ranked shortlist. A listing has to
match at least half the query's distinct words to count at all — without that
floor, "90s track jacket" returned a bucket hat tagged `90s`.

### 2. `suggest_outfit(new_item, wardrobe, trends=None) -> str`

| Input | Type | Meaning |
|---|---|---|
| `new_item` | `dict` | A listing dict — the piece being considered. |
| `wardrobe` | `dict` | `{"items": [...]}`; each item has `id`, `name`, `category`, `colors`, `style_tags`, `notes`. May be empty. |
| `trends` | `list[str] \| None` | Trending tags from `get_trending_styles()`. `None` = no trend context. |

**Returns:** a non-empty string — 1–2 outfits (≤90 words) naming specific
wardrobe pieces plus one concrete styling move. With an empty wardrobe it
returns general styling advice prefixed with an explanation. If the LLM is
unreachable it returns a string starting with `"[outfit-unavailable]"`.

**Purpose:** answer "would this actually work with what I own?"

### 3. `create_fit_card(outfit, new_item) -> str`

| Input | Type | Meaning |
|---|---|---|
| `outfit` | `str` | The string returned by `suggest_outfit()`. |
| `new_item` | `dict` | The listing dict, for the item name, price and platform. |

**Returns:** a 2–4 sentence casual caption. Runs at `temperature=1.05`, so
repeated calls on the same input differ (asserted in
`test_fit_card_varies_across_calls`). On empty/failed input it returns a string
starting with `"[fit-card-unavailable]"`; if the LLM dies it falls back to a
deterministic template caption.

**Purpose:** the shareable payoff — something you'd actually post.

### 4. `compare_price(item, listings=None) -> dict` *(stretch)*

**Returns:** `{verdict, item_price, median_comparable, pct_diff, n_comparables,
comparables, reasoning}` — `verdict` is `"great deal"` / `"fair"` /
`"a bit high"` / `"overpriced"` / `"unknown"`; `comparables` is up to 3 dicts of
`{title, price, condition}`; `reasoning` is one sentence naming the median and
the sample size. **How comparisons are made:** candidates are listings in the
same `category`, ranked by `style_tags` overlap; if at least 3 share a tag the
median is taken over those, otherwise over the whole category. The verdict is
the item's % distance from that median (≤ −20% great deal, ≤ +10% fair,
≤ +30% a bit high, above that overpriced).

### 5. `get_trending_styles(size=None, top_n=5) -> dict` *(stretch)*

**Returns:** `{source, size_scope, trending, summary}` where `trending` is a
list of `{tag (str), count (int), share (float)}`. **Data source:** the live
listing feed in `data/listings.json` — the same inventory the search tool
queries — treated as what sellers on depop/thredUp/poshmark are listing right
now, weighted up for `excellent` condition and for depop (the fastest-moving
platform in this dataset). What's being listed in your size *is* the trend
signal. This is a local dataset, not a live scrape of a public platform.

### 6. `load_style_profile(user_id="default") -> dict` / `save_style_profile(profile, user_id="default") -> bool` *(stretch)*

**Storage approach:** a single JSON file, `data/style_profile.json`, keyed by
`user_id`. `load` returns `{user_id, size, typical_max_price, style_tags,
wardrobe, query_count, updated_at}`; a missing or corrupt file yields a valid
empty profile rather than raising. `save` returns `False` on a write failure
instead of raising — losing memory must never break the current run.

---

## How the Planning Loop Works

`run_agent()` is a `while` loop over a `session["next_action"]` state variable.
Each pass runs **one** step, inspects what that step returned, and writes the
next action from it. No step is unconditional; the sequence of tool calls is a
function of the data.

```
parse → search → [retry_search…] → evaluate → trends → outfit → fit_card → done
             ↘ (no keywords) ────────────────────────────────────────────→ done
             ↘ (nothing found, ladder exhausted) ──────────────────────────→ done
                                                    ↘ (styling failed) ────→ done
```

The conditional logic, branch by branch:

| Step | What it checks | What it does |
|---|---|---|
| `parse` | Is the description empty after stop-word removal? | **Yes →** set `error`, go to `done`. **No tool is called at all.** (Branch A) **No →** go to `search`. |
| `search` | `len(search_results)` | **> 0 →** `evaluate`. **== 0 and rungs remain →** `retry_search` (Branch B). **== 0 and ladder exhausted →** set `error`, `done` (Branch C). |
| `retry_search` | Did the loosened search return anything? | **Yes →** record what was loosened in `adjustments`, go to `evaluate`. **No →** next rung, or `give_up`. |
| `evaluate` | `compare_price(...)["verdict"]` | **`"overpriced"` and a qualifying better-value alternative exists →** swap `selected_item` (Branch D). **Otherwise →** keep `results[0]`. Then `trends`. |
| `trends` | Did the trend scan return tags? | Either way → `outfit`. Trends never block the run. |
| `outfit` | Does the result start with `"[outfit-unavailable]"`? | **Yes →** warn, **skip the fit card**, `done` (Branch F). **No →** `fit_card`. If the wardrobe was empty, add a warning first (Branch E). |
| `fit_card` | Does the result start with `"[fit-card-unavailable]"`? | **Yes →** move it into `warnings`, leave `fit_card = None`. **No →** store it. Then `done`. |

**What happens specifically when `search_listings` returns no results:** the
agent does *not* call `suggest_outfit`. It builds a relaxation ladder and
retries automatically (stretch: retry logic with fallback):

1. **Drop the size filter** — *"couldn't find it in size M, so I searched every size"*
2. **Raise the budget 50%**, rounded to the nearest $5 — *"nothing came up under $20, so I stretched the budget to $30"*
3. **Search the head noun only** — *"broadened the search to just 'jacket'"*

Each rung that runs is recorded in `session["adjustments"]` and shown to the
user, so they always know what was changed on their behalf. If every rung still
returns `[]`, the loop sets a specific error naming the constraints it tried,
what it already loosened, and the nearest real listing in the catalogue — then
terminates with `fit_card = None`.

**Loop termination:** `next_action == "done"`, reachable three ways — a
completed fit card, an early error branch (A or C), or a degraded-but-useful
branch (F). A `MAX_STEPS = 12` cap guards against a state-machine bug; it is
never hit in normal flow.

Real traces of two different queries, straight from the app's trace panel:

```
"vintage graphic tee under $30"          "designer ballgown size XXS under $5"
1. parse                                  1. parse
2. search_listings (6 matches)            2. search_listings — 0 matches
3. compare_price — a bit high             3. retry_search — dropped size → nothing
4. get_trending_styles                    4. retry_search — budget → $10 → nothing
5. suggest_outfit                         5. retry_search — just 'ballgown' → nothing
6. create_fit_card                        6. give_up — suggest_outfit never called
```

---

## State Management

**What is stored:** one `session` dict per interaction, created by
`_new_session()` in `agent.py`. It is the single source of truth — every step
reads the keys it needs and writes the keys it owns.

| Key | Written by | Read by |
|---|---|---|
| `query` | caller | `parse` |
| `parsed` (`description`, `size`, `max_price`) | `parse` | `search`, `retry_search`, `trends` |
| `search_results` | `search` / `retry_search` | `evaluate` |
| `selected_item` | `evaluate` | `outfit`, `fit_card` |
| `price_check` | `evaluate` | UI, the swap branch |
| `trends` | `trends` | `outfit` |
| `outfit_suggestion` | `outfit` | `fit_card` |
| `fit_card` | `fit_card` | UI |
| `wardrobe` | caller / style profile | `outfit` |
| `error` | any early-terminating branch | caller, UI |
| `warnings`, `adjustments`, `decisions`, `memory_applied`, `log` | every step | UI, demo trace |
| `next_action` | every step | the `while` loop |

**When and how it passes between tools:** by reference, immediately, with no
re-prompting and nothing recomputed.

- `search_listings` returns a list → the loop stores `results[0]` as
  `session["selected_item"]` → that **same dict object** is handed to
  `suggest_outfit`. The user never re-types the item.
- `suggest_outfit` returns a string → stored as
  `session["outfit_suggestion"]` → that **same string object** is passed to
  `create_fit_card`.

This is asserted with identity (`is`), not equality, in
`tests/test_agent.py::test_outfit_string_flows_into_the_fit_card`:

```python
assert seen["item"] is session["selected_item"]
assert seen["outfit"] is session["outfit_suggestion"]
```

**Across sessions (stretch — style profile memory):** `run_agent()` loads
`data/style_profile.json` during `parse` and uses it to fill gaps the query
didn't specify (size, budget, styles, and the wardrobe itself), recording each
substitution in `session["memory_applied"]` so the user can see what was
remembered. It saves the updated profile at `done`.

---

## Error Handling and Fail Points

Every tool handles its own failure and returns a value the loop can branch on —
a sentinel string, an empty list, or an `"unknown"` verdict. No tool raises for
an expected failure, and nothing fails silently.

| Tool | Failure mode | Agent response |
|---|---|---|
| `search_listings` | No listing matches description + size + price (returns `[]`) | Does **not** call `suggest_outfit`. Runs the 3-rung relaxation ladder and tells the user exactly what it changed. If everything fails, stops with a message naming the constraints tried and the nearest real listing. `fit_card` stays `None`. |
| `search_listings` | `data/listings.json` missing or malformed | Returns `[]` instead of raising; surfaces as the no-results path rather than a stack trace. |
| `suggest_outfit` | Wardrobe is empty (`items == []`) | Not an error. Switches to a general-styling prompt, prefixes *"You haven't added any wardrobe pieces yet…"*, adds a warning, and **continues** to the fit card. |
| `suggest_outfit` | LLM errors, times out, or returns an empty/truncated completion | Retries once with triple the token budget, then returns `"[outfit-unavailable] …"`. The agent warns, **skips `create_fit_card`** rather than captioning an outfit that doesn't exist, and still returns the listing and price check. |
| `create_fit_card` | `outfit` empty, whitespace, or the unavailable sentinel | Returns `"[fit-card-unavailable] Can't write a fit card without an outfit — …"`. No exception, no invented caption. |
| `create_fit_card` | LLM errors after one retry | Falls back to a deterministic template caption built from the item's own fields, tagged *"(caption written offline…)"*. |
| `compare_price` | Fewer than 2 comparable listings | `verdict = "unknown"` with a reason; the run continues. A missing price check never blocks the outfit. |
| `get_trending_styles` | Too few listings in scope, or the feed can't be read | Returns `trending: []`; the loop calls `suggest_outfit(trends=None)` and proceeds. |
| style profile | File missing, corrupt, or unwritable | Degrades to an empty in-memory profile; the run works, nothing is remembered, a warning is recorded. |

### Concrete examples from testing

**1. Zero results (triggered deliberately).**

```
$ python -c "from tools import search_listings; print(search_listings('designer ballgown', size='XXS', max_price=5))"
[]
```

Empty list, no exception. Through the full agent, the same query produces:

> ❌ I couldn't find anything matching "designer ballgown", size XXS, under $5 in
> the catalogue. I also tried: couldn't find it in size XXS, so I searched every
> size; nothing came up under $5, so I stretched the budget to $10; broadened the
> search to just 'ballgown' — still nothing. Nothing in the catalogue uses those
> words at all. Try different keywords (the catalogue covers tops, bottoms,
> outerwear, shoes and accessories in vintage, y2k, grunge, streetwear and
> cottagecore styles).

The trace confirms `suggest_outfit` was never reached, and `session["fit_card"]`
is `None`.

**2. Empty wardrobe (triggered deliberately).** `suggest_outfit(item,
get_empty_wardrobe())` returns advice, not an exception:

> You haven't added any wardrobe pieces yet, so here's general styling for this
> piece — add a few items and I'll style it against what you actually own.
>
> Pair the black 2003 tour-bootleg graphic tee with relaxed denim — think
> straight-leg or slightly distressed jeans — and a heavyweight, oversized
> flannel left unbuttoned…

**3. Empty outfit string (triggered deliberately).** `create_fit_card('', item)`:

> [fit-card-unavailable] Can't write a fit card without an outfit — run a search
> and get a styling suggestion first so I have something to caption.

**4. Styling model unreachable (simulated in
`test_outfit_failure_skips_the_fit_card`).** The agent keeps the listing and the
price check, warns the user, and `create_fit_card` is never called — asserted,
not assumed.

---

## Interaction Walkthrough

**User query:** *"I'm looking for a vintage graphic tee under $30. I mostly wear
baggy jeans and chunky sneakers."*

**Step 0 — parse (no tool call).** Regex pulls `max_price=30.0`; no size given.
The clause *"I mostly wear baggy jeans and chunky sneakers"* is stripped as a
wardrobe aside — it describes what the user owns, not what they're shopping for,
and letting "jeans" into the search would return jeans. Result:
`{"description": "vintage graphic tee", "size": None, "max_price": 30.0}`.

**Step 1 — `search_listings("vintage graphic tee", None, 30.0)`**
- *Why:* the parse produced usable keywords.
- *Output:* 6 matches. Top: **Graphic Tee — 2003 Tour Bootleg Style**, $24.00,
  depop, good condition, size L.
- Non-empty, so the retry ladder is skipped entirely → `evaluate`.

**Step 2 — `compare_price(<the tee>, <the 6 results>)`** *(stretch)*
- *Why:* before recommending a purchase, check it's worth the money.
- *Output:* `verdict="a bit high"` — *"$24.00 is 26% above the $19.00 median of
  5 comparable listings."* Not `"overpriced"`, so **no swap**;
  `selected_item` stays the bootleg tee.

**Step 3 — `get_trending_styles(size=None)`** *(stretch)*
- *Output:* `vintage, classic, streetwear, earth tones, cottagecore` → stored in
  `session["trends"]` and passed into the next call.

**Step 4 — `suggest_outfit(session["selected_item"], <10-item wardrobe>, trends)`**
- *Why:* we have an item and a wardrobe; this is the "would it work?" question.
- *Input:* the **same dict object** Step 1 returned — nothing re-entered.
- *Output:* *"Pair the Graphic Tee with the Baggy straight-leg jeans, dark wash
  and Chunky white sneakers, cuff the jeans at the ankle for a clean streetwear
  vibe. For a vintage-earthy spin, half-tuck the Graphic Tee into the Wide-leg
  khaki trousers and finish with Black combat boots."*

**Step 5 — `create_fit_card(session["outfit_suggestion"], session["selected_item"])`**
- *Input:* both arguments read straight out of the session.
- *Output:* *"snagged this 2003 tour bootleg graphic tee for $24 on depop and
  it's already a staple. pairing it with baggy dark-wash straight-leg jeans and
  chunky white sneakers gives me that clean streetwear vibe. for a grunge twist
  i half-tuck it into wide-leg khaki trousers and throw on black combat boots 😎"*

**Final output to user:** four panels — the listing with its price verdict and
comparables, the outfit, the fit card, and the agent trace listing every step
above so the branching is visible rather than implied.

---

## Stretch Features

| Feature | Where | Notes |
|---|---|---|
| **Price comparison tool** | `compare_price()` in `tools.py` | Verdict + median + reasoning from comparable listings. Also feeds Branch D of the loop. |
| **Style profile memory** | `load_style_profile()` / `save_style_profile()`, applied in `run_agent`'s `parse` step | Session 2 resolves a size and budget session 2 never mentioned — see `test_profile_carries_size_into_a_later_session`. |
| **Trend awareness** | `get_trending_styles()` | Tags are passed into `suggest_outfit`, which is asked to note when an outfit leans current. Source is the local listing feed, not a live platform scrape. |
| **Retry logic with fallback** | `_relaxation_ladder()` + the `retry_search` state | 3 rungs, each reported to the user in plain language. |

---

## Spec Reflection

**One way planning.md helped during implementation.** Writing the error-handling
table before any code forced me to decide that a tool failure should be *data*,
not an exception — every tool returns a sentinel (`[outfit-unavailable]`), an
empty list, or an `"unknown"` verdict, and only the planning loop decides what
that means. That single decision made the loop trivial to write: each state
inspects a return value and picks the next state, with no try/except scattered
through the orchestration layer. It also made the branches directly testable —
`test_outfit_failure_skips_the_fit_card` monkeypatches a tool to return the
sentinel and asserts `create_fit_card` is never called, which would have been
much harder if failure meant an exception thrown from three layers down.

**One divergence from the spec, and why.** The spec named
`meta-llama/llama-4-scout-17b-16e-instruct` as the model. It has been retired
from Groq's free tier and returns `404 model_not_found`, so `tools.py` now keeps
a `PREFERRED_MODELS` list — the assignment's model first, then the models the
account can actually serve — and `_resolve_model()` asks the API once per
process and caches the first available one (currently `openai/gpt-oss-120b`).
The replacements turned out to be *reasoning* models, which caused a second,
unplanned divergence: they spend tokens thinking before emitting any content, so
my originally-specced token budgets produced empty or mid-sentence-truncated
completions. `_chat()` now treats both an empty completion and a
`finish_reason == "length"` as retryable and triples the budget on the retry,
and `create_fit_card` trims a long outfit to ~600 characters before prompting.
Neither behaviour was in the spec; both were forced by what the available models
actually do.

---

## AI Usage

I used **Claude (Claude Code in the terminal)** throughout, because it could
read the repo, run the code, and run pytest in the same loop — which made
"verify the generated code against the spec" cheap enough to actually do every
time. Four specific instances:

**1. `search_listings` — directed it to implement Tool 1 from my spec.**
I gave it the Tool 1 block from planning.md (all three parameters with types,
the `match_score` return contract, the `[]`-on-failure rule) plus
`utils/data_loader.py`, and told it to use `load_listings()` rather than
re-reading the JSON, with no LLM call.
**What I reviewed and overrode:** I specifically checked the size-matching trap
I'd flagged in my spec — whether `size="M"` could match inside `"XL
(oversized)"` — and confirmed it tokenised the size string first. Then I ran it
against real queries and **caught a relevance bug the spec hadn't anticipated**:
`"90s track jacket"` returned a *bucket hat*, because the hat is tagged `90s`
and one keyword hit was enough to qualify. I added a relevance floor (a listing
must match at least half the query's distinct words) and went back and amended
the Tool 1 spec to record it. That's `test_search_requires_meaningful_overlap`.

**2. The planning loop — directed it to implement Milestone 4 from the diagram.**
I gave it the Planning Loop section, the State Management table, and the ASCII
architecture diagram together, and asked for a `while` loop over
`session["next_action"]` rather than a straight-line script.
**What I reviewed and overrode:** I checked the three things my AI Tool Plan
listed — that it branches on the `search_listings` return value, that
`suggest_outfit` is unreachable when `search_results == []`, and that nothing is
recomputed between steps. The part I **overrode was the price-swap branch
(Branch D)**. My own spec said "swap if another top-5 result is cheaper and
scores within 25%", and when I actually traced it, that rule swapped a $75
leather jacket for a **$12 leather belt** — cheaper, same keyword, useless. I
rewrote the condition to require the same `category`, a `matched_terms`
superset, and a strictly better price verdict, then had to add `matched_terms`
to `search_listings`'s return value to support it. A first fix that only
required "cheaper" still recommended a *second overpriced item* as a saving, so
the verdict-improvement check went in on a second pass.

**3. The Groq wrapper — directed it to add a retry with a sentinel return.**
I gave it the failure-mode lines from Tools 2 and 3 ("retry once, then return
`[outfit-unavailable]`; never raise").
**What I reviewed and revised:** the first version only retried on exceptions. I
found by testing that the actual failure was subtler — the model returned a
*successful* response with empty content, because it had spent the whole token
budget reasoning. I revised `_chat()` to treat an empty completion as retryable
and to triple the budget on the retry. Later, the fit card came back truncated
mid-word ("snagged this 200"), which was the same class of bug with
`finish_reason == "length"`, so I extended the same handling and capped the
outfit text fed into the caption prompt.

**4. Tests — directed it to write one test per documented failure mode.**
I gave it the error-handling table and asked for a test per row plus the three
assignment-provided search tests.
**What I reviewed and overrode:** one generated test asserted that
`get_trending_styles(size="XXS")` returns an empty trend list. It failed — and
the *test* was wrong, not the code: `"One Size"` listings legitimately fit XXS,
so that scope is never thin. I replaced it with a test documenting the real
behaviour and a separate one that monkeypatches `load_listings` to a single item
to genuinely exercise the thin-scope branch.

---

## Repo Layout

```
├── agent.py                 # planning loop (state machine + query parsing)
├── tools.py                 # all 6 tools
├── app.py                   # Gradio UI, 4 panels incl. the agent trace
├── planning.md              # the spec, written before implementation
├── tests/
│   ├── test_tools.py        # per-tool tests, incl. every failure mode
│   └── test_agent.py        # branch + state-handoff tests
├── data/
│   ├── listings.json        # 40 mock secondhand listings
│   ├── wardrobe_schema.json # wardrobe format + example wardrobe
│   └── style_profile.json   # created at runtime (gitignored)
└── utils/data_loader.py     # load_listings / get_example_wardrobe / get_empty_wardrobe
```

**Test status:** 45 passing (`python -m pytest tests/`); 39 of them need no API
key (`python -m pytest tests/ -m "not llm"`).
