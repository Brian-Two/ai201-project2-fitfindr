# FitFindr — planning.md

> Spec written before implementation. Updated once more before the stretch
> features were built (the four stretch tools/behaviours are marked
> **[stretch]** and were added to this document before their code existed).

**What FitFindr does, in three sentences:** FitFindr takes one natural language
thrifting request ("vintage graphic tee under $30, size M"), parses it into
search constraints, and searches a local dataset of 40 secondhand listings with
`search_listings` — that tool is triggered by the user's query itself. If the
search returns something, the top listing flows automatically into
`suggest_outfit` (which styles it against the user's saved wardrobe) and then
into `create_fit_card` (which writes a shareable caption), with the item and
outfit carried in a session dict so the user never re-types them. If any step
fails — no matches, an empty wardrobe, an LLM error, an empty outfit string —
the agent stops or degrades on that specific branch, tells the user exactly what
failed and what to change, and never calls a downstream tool with empty input.

---

## Tools

List every tool your agent will use. For each tool, fill in all four fields.
You must have at least 3 tools. The three required tools are listed — add any additional tools below them.

### Tool 1: search_listings

**What it does:**
Filters the 40-listing mock dataset (`data/listings.json`, loaded via
`load_listings()`) down to items that match the user's keywords, size and price
ceiling, scores each survivor by how well it matches the keywords, and returns
them best-match-first. It is pure Python — no LLM call — so it is deterministic
and cheap to test.

**Input parameters:**
- `description` (str): free-text keywords describing the wanted piece, e.g.
  `"vintage graphic tee"`. Tokenised, stop-words dropped, then matched against
  each listing's `title`, `description`, `style_tags`, `category`, `colors`
  and `brand`.
- `size` (str | None): size filter, e.g. `"M"`, `"8"`, `"W30"`. `None` skips
  size filtering. Matching is case-insensitive and substring-token based so
  `"M"` matches `"M"`, `"S/M"`, `"M/L"` and `"One Size"` (which fits everyone),
  but not `"XL"` — a listing size of `"XL (oversized)"` is tokenised so that
  `"XL"` is one token and `"M"` does not match inside it.
- `max_price` (float | None): inclusive price ceiling. `None` skips price
  filtering.

**Relevance floor:** a listing must match at least half of the distinct words in
the query (rounded up) to count as a hit at all. *(Added during Milestone 3:
without it, "90s track jacket" returned a bucket hat tagged `90s` — technically
a keyword hit, obviously not what was asked for.)*

**What it returns:**
`list[dict]` — the matching listing dicts, sorted by descending relevance score
(cheaper item wins a tie). Every dict is a listing straight out of the dataset
plus two added keys: `id` (str), `title` (str), `description` (str),
`category` (str), `style_tags` (list[str]), `size` (str), `condition` (str),
`price` (float), `colors` (list[str]), `brand` (str | None), `platform` (str),
`match_score` (float, added by this tool — higher is a better keyword match) and
`matched_terms` (list[str], added by this tool — which of the query's words this
listing actually matched, used by the planning loop's value-swap branch).
Returns `[]` when nothing matches.

**What happens if it fails or returns nothing:**
It never raises. On a bad/missing data file it returns `[]`. When the result is
empty the **agent** (not the tool) runs the relaxation ladder described under
"Retry with fallback" below; if every rung still returns `[]`, the agent sets
`session["error"]` to a message naming the three constraints it tried and what
to change ("nothing under $5 in XXS — the cheapest formal piece in the catalogue
is $34; try raising your budget or dropping the size filter") and returns
**before** `suggest_outfit` is ever called.

---

### Tool 2: suggest_outfit

**What it does:**
Takes one listing and the user's wardrobe and asks the LLM for one or two
complete, wearable outfits that combine the new item with *specific, named*
pieces the user already owns (plus trend context when available).

**Input parameters:**
- `new_item` (dict): a listing dict — the piece the user is considering. Its
  `title`, `category`, `colors`, `style_tags`, `price` and `platform` go into
  the prompt.
- `wardrobe` (dict): `{"items": [...]}` where each item has `id`, `name`,
  `category`, `colors`, `style_tags`, `notes`. May be `{"items": []}`.
- `trends` (list[str] | None, optional, **[stretch]**): trending style tags from
  `get_trending_styles()`; when present the prompt asks the LLM to note which
  outfit leans into a current trend. `None` means "no trend context" and the
  tool behaves exactly as without the stretch feature.

**What it returns:**
A non-empty `str`: 2–5 sentences describing 1–2 outfits, naming wardrobe pieces
explicitly ("your wide-leg khakis + the chunky white sneakers") and giving one
concrete styling move (tuck, cuff, layer).

**What happens if it fails or returns nothing:**
Two distinct failure modes, two distinct behaviours.
1. **Empty wardrobe** (`wardrobe["items"] == []`) — *not* an error. The tool
   switches to a second prompt that returns general styling advice ("this pairs
   with straight-leg denim and white sneakers; it suits a 90s-casual vibe") and
   prefixes the answer with a line telling the user that adding wardrobe items
   would make this specific. The agent continues to the fit card.
2. **LLM error / empty completion** — the tool retries once, and if it still
   fails returns the sentinel string starting with `"[outfit-unavailable]"`.
   The agent detects that prefix, records a warning, **skips**
   `create_fit_card` (a caption about an outfit we never got would be
   fabricated), and still returns the listing and price check so the run is not
   a total loss.

---

### Tool 3: create_fit_card

**What it does:**
Turns the outfit suggestion plus the item into a short, casual, shareable
caption — the kind of thing someone actually posts, not a product description.
Runs at `temperature=1.05` so repeated calls on the same input read differently.

**Input parameters:**
- `outfit` (str): the string returned by `suggest_outfit()`.
- `new_item` (dict): the listing dict, used for the item name, price and
  platform that the caption mentions once each.

**What it returns:**
A `str` of 2–4 short sentences, lowercase-leaning and casual, mentioning the
item, its price and its platform naturally. Different inputs (and repeat calls
on the same input) produce visibly different captions.

**What happens if it fails or returns nothing:**
- `outfit` empty / whitespace-only / the `"[outfit-unavailable]"` sentinel →
  returns the descriptive error **string**
  `"[fit-card-unavailable] Can't write a fit card without an outfit — ..."`.
  No exception, no fabricated caption.
- LLM error → retries once, then falls back to a deterministic template caption
  built from the item fields (`"thrifted this {title} for ${price} on
  {platform} 🫶"`) so the user still gets something usable, and the agent notes
  in the session that the fallback was used.

---

### Additional Tools (if any)

### Tool 4 **[stretch]**: compare_price

**What it does:**
Estimates whether a listing is fairly priced by comparing it against comparable
listings in the same dataset — same `category`, ranked by overlap of
`style_tags` and `condition` — and returns a verdict with the numbers behind it.

**Input parameters:**
- `item` (dict): the listing being judged.
- `listings` (list[dict] | None): the pool to compare against; `None` loads the
  full dataset with `load_listings()`.

**What it returns:**
`dict` with keys: `verdict` (str — one of `"great deal"`, `"fair"`,
`"a bit high"`, `"overpriced"`, `"unknown"`), `item_price` (float),
`median_comparable` (float | None), `pct_diff` (float | None — % above/below
the median), `n_comparables` (int), `comparables` (list[dict] with `title`,
`price`, `condition` for the 3 closest matches), and `reasoning` (str — one
sentence naming the median, the sample size and the comparison basis).

**What happens if it fails or returns nothing:**
Fewer than 2 comparables → `verdict="unknown"`, `median_comparable=None`, and a
`reasoning` string saying the catalogue is too thin in that category to judge.
The agent prints the note and carries on — a missing price check never blocks
the outfit or the fit card.

### Tool 5 **[stretch]**: get_trending_styles

**What it does:**
Builds a "what's popular right now" feed: counts `style_tags` across listings
available in the user's size, weights recent/high-demand signals (condition and
platform freshness proxy), and returns the top tags with counts.

**Input parameters:**
- `size` (str | None): restrict the trend scan to listings that fit this size;
  `None` scans the whole catalogue.
- `top_n` (int, default 5): how many tags to return.

**What it returns:**
`dict` with `source` (str — names the data source), `size_scope` (str),
`trending` (list[dict] with `tag` (str), `count` (int), `share` (float)), and
`summary` (str — one human-readable sentence). Empty `trending` list if the
size scope has no listings.

**What happens if it fails or returns nothing:**
Returns `trending: []` with a `summary` explaining that there aren't enough
listings in that size to call a trend. The agent then calls `suggest_outfit`
with `trends=None` and the run proceeds normally.

### Tool 6 **[stretch]**: style profile memory — `load_style_profile` / `save_style_profile`

**What it does:**
Persists a user's style preferences to `data/style_profile.json` between
sessions: their usual size, typical budget, favourite style tags (accumulated
from past queries) and their wardrobe, so a second session can run
`"something for the weekend"` with no size, no budget and no wardrobe re-entry.

**Input parameters:**
- `load_style_profile(user_id: str = "default") -> dict`
- `save_style_profile(profile: dict, user_id: str = "default") -> bool`

**What it returns:**
`load_style_profile` → `dict` with `user_id` (str), `size` (str | None),
`typical_max_price` (float | None), `style_tags` (list[str]),
`wardrobe` (dict), `query_count` (int), `updated_at` (str ISO timestamp).
A missing/corrupt file returns a valid empty profile rather than raising.
`save_style_profile` → `bool` (False on write failure, logged not raised).

**What happens if it fails or returns nothing:**
Any read/write failure degrades to an in-memory-only profile: the current run
works normally, nothing is remembered, and the session records a warning. The
agent never blocks on memory.

---

## Planning Loop

**How does your agent decide which tool to call next?**

`run_agent()` is a `while` loop over an explicit `session["next_action"]` state
variable. Each iteration runs exactly one step, inspects what that step
returned, and **writes the next action based on that return value** — so the
sequence of tool calls is different for different inputs. The loop exits when
`next_action` becomes `"done"` (or after a hard cap of 12 iterations, which is
a safety net against a state-machine bug, not part of normal flow).

The branches, concretely:

1. **`parse`** — regex-extract `max_price` (`under $30`, `below 30`, `$30`),
   `size` (`size M`, `in a medium`, `size 8`), and `description` (the remaining
   words with stop-words removed). Fill any gap from the saved style profile
   **[stretch]**, recording what was filled in `session["memory_applied"]`.
   - If `description` has **no** usable keyword after stop-word removal →
     `session["error"] = "I couldn't tell what you're looking for..."`,
     `next_action = "done"`. **No tool is called at all.** (Branch A)
   - Otherwise → `next_action = "search"`.
2. **`search`** — call `search_listings(description, size, max_price)`.
   - `len(results) > 0` → `session["search_results"] = results`,
     `next_action = "evaluate"`.
   - `results == []` and relaxation rungs remain → `next_action = "retry_search"`.
     (Branch B)
   - `results == []` and the ladder is exhausted → set the specific
     `session["error"]`, `next_action = "done"`. **`suggest_outfit` is never
     called.** (Branch C)
3. **`retry_search`** **[stretch: retry logic with fallback]** — pop the next
   rung off the relaxation ladder, re-run `search_listings`, append a
   human-readable note to `session["adjustments"]`, then go back to `search`'s
   evaluation. The ladder, in order:
   1. drop the size filter → *"I couldn't find it in size M, so I searched all sizes."*
   2. raise `max_price` by 50% (rounded to the nearest $5) → *"Nothing under $20, so I looked up to $30."*
   3. drop both, and search on the single strongest keyword only → *"I broadened to just 'tee'."*
4. **`evaluate`** — set `selected_item = search_results[0]`, then call
   `compare_price(selected_item, search_results)` **[stretch]**.
   - If the verdict is `"overpriced"`, look for a better-value alternative among
     the next five results. A candidate qualifies only if **all four** hold:
     same `category`; at least 15% cheaper; its `matched_terms` are a superset
     of the top result's (it answers the query at least as completely); and its
     own `compare_price` verdict ranks strictly better. If one qualifies →
     **switch `selected_item` to it** and record the swap in
     `session["decisions"]`. This is the clearest case of a tool's *return
     value* changing which item flows downstream. (Branch D)
     *(All four conditions were added during Milestone 4. A looser "cheaper and
     vaguely similar" rule swapped a $75 leather jacket for a $12 leather belt,
     and a cheaper-only rule recommended a second overpriced item as a saving.)*
   - Otherwise keep the top result.
   - → `next_action = "trends"`.
5. **`trends`** **[stretch]** — call `get_trending_styles(size=parsed size)`.
   Non-empty → store tags in `session["trends"]`. Empty/failed → store `None`
   and note it. Either way → `next_action = "outfit"` (trends never block).
6. **`outfit`** — call `suggest_outfit(selected_item, wardrobe, trends)`.
   - Wardrobe empty → the tool returns general advice; the agent additionally
     sets `session["warnings"]` with a prompt to add wardrobe items, then
     `next_action = "fit_card"`. (Branch E — same tool, different output path)
   - Result starts with `"[outfit-unavailable]"` → set a warning, **skip the
     fit card**, `next_action = "done"`. (Branch F)
   - Otherwise → `next_action = "fit_card"`.
7. **`fit_card`** — call `create_fit_card(outfit_suggestion, selected_item)`.
   Store it; if it comes back with the `"[fit-card-unavailable]"` prefix, move
   it into `session["warnings"]` and leave `session["fit_card"] = None`.
   → `next_action = "done"`.
8. **`done`** — persist the style profile **[stretch]**, then return the session.

Every iteration appends a `(step, summary)` pair to `session["log"]` so the
trace of *which* tools ran for *this* query is visible in the UI and the demo.

**How does it know it's done?** `next_action == "done"`, which is reached three
ways: a successful fit card, an early error branch (A/C), or a degraded-but-
useful branch (F).

---

## State Management

**How does information from one tool get passed to the next?**

One `session` dict per interaction, created by `_new_session()`, is the single
source of truth. Nothing is passed by re-prompting the user and nothing is
recomputed — each step *reads* the keys it needs and *writes* the keys it owns:

| Key | Written by | Read by |
|-----|-----------|---------|
| `query` | caller | `parse` |
| `parsed` (`description`, `size`, `max_price`) | `parse` | `search`, `retry_search`, `trends` |
| `search_results` | `search` / `retry_search` | `evaluate` |
| `selected_item` | `evaluate` | `outfit`, `fit_card`, `price_check` |
| `price_check` | `evaluate` | UI, `evaluate`'s swap branch |
| `trends` | `trends` | `outfit` |
| `outfit_suggestion` | `outfit` | `fit_card` |
| `fit_card` | `fit_card` | UI |
| `wardrobe` | caller / style profile | `outfit` |
| `error` | any branch that terminates early | caller, UI |
| `warnings`, `adjustments`, `decisions`, `log` | every step | UI, demo |
| `next_action` | every step | the `while` loop |

The concrete hand-offs the rubric asks about:
`session["selected_item"]` is the *same dict object* that `search_listings`
returned (`results[0]`) and the *same object* passed into `suggest_outfit` —
verifiable with `id()` / `is`, and asserted in `tests/test_agent.py`.
`session["outfit_suggestion"]` is the exact string returned by `suggest_outfit`
and the exact string passed into `create_fit_card`. The user types their query
once; nothing is re-entered.

**Across sessions [stretch]:** `data/style_profile.json` stores size, typical
budget, accumulated style tags and the wardrobe. `run_agent` loads it at
`parse` to fill gaps in the query and saves it at `done`.

---

## Error Handling

For each tool, describe the specific failure mode you're handling and what the agent does in response.

| Tool | Failure mode | Agent response |
|------|-------------|----------------|
| search_listings | No listings match the description + size + price combination (returns `[]`) | Does **not** call `suggest_outfit`. Runs the relaxation ladder automatically (drop size → raise budget 50% → single strongest keyword), and if a rung succeeds, tells the user exactly what it changed: *"No size M matches under $20 — I searched all sizes up to $30 instead and found these."* If every rung fails, it stops with a specific, actionable message naming the binding constraint and the closest real data point: *"Nothing in the catalogue matches 'designer ballgown' in XXS under $5. The cheapest formal piece here is the $34 slip dress — try raising your budget, dropping the size filter, or searching 'slip dress'."* `session["fit_card"]` stays `None`. |
| search_listings | Dataset missing or malformed JSON | Returns `[]` instead of raising; agent reports *"I couldn't read the listings catalogue — check that data/listings.json exists."* |
| suggest_outfit | Wardrobe is empty (`items == []`) | Not treated as an error. Switches to a general-styling prompt so the user still gets advice, prefixed with *"You haven't added any wardrobe pieces yet, so here's general styling for this piece — add a few items and I'll style it against what you actually own."* Flow continues to the fit card. |
| suggest_outfit | LLM call errors, times out, or returns an empty completion | Retries once, then returns `"[outfit-unavailable] ..."`. The agent surfaces *"I found the item but couldn't reach the styling model — here's the listing and the price check; try again in a moment."* and **skips the fit card** rather than captioning an outfit that doesn't exist. |
| create_fit_card | `outfit` is empty, whitespace, or the unavailable sentinel | Returns the string *"[fit-card-unavailable] Can't write a fit card without an outfit — run a search first so I have something to caption."* No exception, no invented caption. |
| create_fit_card | LLM errors after one retry | Falls back to a deterministic template caption built from the item's own fields so the user still leaves with something shareable, and the session warns *"(caption written offline — the styling model was unreachable)"*. |
| compare_price **[stretch]** | Fewer than 2 comparable listings | `verdict="unknown"` with a reason; the agent shows *"Not enough comparable listings to judge this price"* and continues to the outfit step. |
| get_trending_styles **[stretch]** | No listings in the size scope | Returns an empty trend list; the agent calls `suggest_outfit(trends=None)` and the outfit step is unaffected. |
| style profile **[stretch]** | Profile file missing, unreadable, or not writable | Falls back to an empty in-memory profile; the run works, nothing is remembered, and a warning is added to the session. |

---

## Architecture

```
                          ┌───────────────────────────────┐
   User query ───────────►│  data/style_profile.json      │  [stretch: cross-session memory]
   ("vintage graphic tee  │  size / budget / style tags / │
    under $30, size M")   │  wardrobe                     │
        │                 └───────────┬───────────────────┘
        │                    load ▲   │ save (on "done")
        ▼                         │   ▼
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║  PLANNING LOOP  —  while session["next_action"] != "done"                ║
 ╠══════════════════════════════════════════════════════════════════════════╣
 ║                                                                          ║
 ║  [parse] regex → description / size / max_price ─┐                       ║
 ║      │                                           │ no usable keywords    ║
 ║      │                                           └─► [ERROR A] ──────┐   ║
 ║      ▼ writes session["parsed"]                                      │   ║
 ║  [search] ──► search_listings(description, size, max_price)          │   ║
 ║      │              returns list[dict] | []                          │   ║
 ║      │                                                               │   ║
 ║      ├── results == [] and rungs left ──► [retry_search] ────┐       │   ║
 ║      │      loosen: drop size → +50% price → 1 keyword       │       │   ║
 ║      │      writes session["adjustments"]                    │       │   ║
 ║      │      ◄───────────────────────────────────────────────┘       │   ║
 ║      │                                                               │   ║
 ║      ├── results == [] and ladder exhausted ──► [ERROR C] ───────────┤   ║
 ║      │        "Nothing matches X in size Y under $Z — try ..."       │   ║
 ║      │                                                               │   ║
 ║      ▼ results = [item, ...]  →  session["search_results"]           │   ║
 ║  [evaluate] selected_item = results[0]                               │   ║
 ║      │                                                               │   ║
 ║      ├──► compare_price(selected_item, search_results)  [stretch]    │   ║
 ║      │        returns {verdict, median_comparable, reasoning, ...}   │   ║
 ║      │        verdict == "overpriced" and cheaper near-match exists  │   ║
 ║      │            └─► selected_item = that cheaper listing           │   ║
 ║      │                session["decisions"] += "swapped for value"    │   ║
 ║      ▼ writes session["selected_item"], session["price_check"]       │   ║
 ║  [trends] ──► get_trending_styles(size)   [stretch]                  │   ║
 ║      │            returns {trending:[{tag,count,share}], summary}    │   ║
 ║      ▼ writes session["trends"]  (None on failure — never blocks)    │   ║
 ║  [outfit] ──► suggest_outfit(selected_item, wardrobe, trends)        │   ║
 ║      │              ▲ same dict object from search — no re-entry     │   ║
 ║      │                                                               │   ║
 ║      ├── wardrobe["items"] == [] ──► general-advice prompt + warning │   ║
 ║      ├── "[outfit-unavailable]" ──► warn, SKIP fit card ──► [DONE F] │   ║
 ║      ▼ writes session["outfit_suggestion"]                           │   ║
 ║  [fit_card] ──► create_fit_card(outfit_suggestion, selected_item)    │   ║
 ║      │              ▲ exact string from the step above               │   ║
 ║      ├── empty/sentinel outfit ──► "[fit-card-unavailable] ..."      │   ║
 ║      ├── LLM down ──► deterministic template caption + warning       │   ║
 ║      ▼ writes session["fit_card"]                                    │   ║
 ║  [done] ◄────────────────────────────────────────────────────────────┘   ║
 ╚══════════════════════════╤═══════════════════════════════════════════════╝
                            │  all error branches converge here
                            ▼
      SESSION STATE (single dict, one per interaction)
      (also: an unparseable query — "hey" — terminates at [parse],
       before any tool is called at all: ERROR A above)
      { query, parsed{description,size,max_price}, search_results[],
        selected_item{}, price_check{}, trends[], wardrobe{},
        outfit_suggestion, fit_card, error, warnings[], adjustments[],
        decisions[], memory_applied[], log[], next_action }
                            │
                            ▼
      Gradio UI: 🛍️ listing + price check  |  👗 outfit  |  ✨ fit card  |  🧭 agent trace
```

---

## AI Tool Plan

The AI tool used throughout was **Claude (Claude Code in the terminal)**, because
it can read the repo, run the code and run pytest in the same loop — which is
what makes the "verify before trusting" step below cheap enough to actually do.

**Milestone 3 — Individual tool implementations:**

- **`search_listings`** — Input: the Tool 1 block above verbatim (all three
  parameters with types, the `match_score` return contract, the `[]`-on-failure
  rule) plus `utils/data_loader.py` so it uses `load_listings()` instead of
  re-reading the JSON. Expected output: one pure-Python function, no LLM call.
  Verification before trusting it: (a) confirm all three parameters are actually
  applied — grep the body for `max_price`, `size` and the keyword scoring;
  (b) confirm `size="M"` does **not** match `"XL (oversized)"` (the substring
  trap I called out in the spec); (c) run the three pytest cases from the
  assignment plus an impossible query that must return `[]`.
- **`suggest_outfit`** — Input: the Tool 2 block, the wardrobe schema from
  `data/wardrobe_schema.json`, and the two-failure-mode rule (empty wardrobe =
  graceful different prompt; LLM error = `"[outfit-unavailable]"` sentinel).
  Expected output: a function with two prompt paths and a `try/except` with one
  retry. Verification: call it with `get_example_wardrobe()` and check the reply
  names real wardrobe pieces ("wide-leg khaki trousers"), then call it with
  `get_empty_wardrobe()` and check it returns advice rather than raising or
  returning `""`.
- **`create_fit_card`** — Input: the Tool 3 block, especially "must differ
  across calls" and "return an error *string*, never raise". Expected output:
  a guard clause, a caption prompt, `temperature≈1.05`. Verification: call it
  three times on the same item and assert the three strings are not identical;
  call it with `outfit=""` and assert the return type is `str` and starts with
  the documented sentinel.
- **Stretch tools** — Input: the Tool 4/5/6 blocks. Verification: for
  `compare_price`, hand-check the verdict for one cheap listing and one
  expensive listing against the median I compute separately in the REPL; for the
  profile, run two agent sessions in one process and assert the second one
  resolves a size the second query never mentioned.

**Milestone 4 — Planning loop and state management:**

- Input: the **Planning Loop** section above (all eight states with their exact
  branch conditions), the **State Management** table (who writes and who reads
  each key), and the ASCII **Architecture** diagram — given together, because
  the diagram carries the error branches and the table carries the data flow.
- Expected output: `run_agent()` as a `while` loop over `session["next_action"]`,
  not a straight-line script.
- Verification before trusting it, against the spec:
  1. Does it *branch* on the `search_listings` return value, or call all three
     tools unconditionally? (If `suggest_outfit` is reachable when
     `search_results == []`, reject it.)
  2. Does every value live in the session dict, with no re-prompting and no
     recomputation between steps?
  3. Assert `session["selected_item"] is` the object passed to `suggest_outfit`
     and that the fit card's input string `is` `session["outfit_suggestion"]`.
  4. Run the impossible query and assert `error` is set **and** `fit_card`
     is `None` **and** the outfit tool was never entered (checked via the
     `log`).

---

## A Complete Interaction (Step by Step)

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers. What's out there and how would I style it?"

**Step 0 — parse (no tool call).**
Regex finds `under $30` → `max_price=30.0`; no `size N` pattern → `size=None`
(and the saved style profile has no size yet either). Stop-words and the
wardrobe aside are stripped, leaving `description="vintage graphic tee"`.
`session["parsed"] = {"description": "vintage graphic tee", "size": None, "max_price": 30.0}`.
Keywords exist → next action is `search`.

**Step 1 — `search_listings("vintage graphic tee", size=None, max_price=30.0)`.**
Called because the parse produced usable keywords. Price filter drops the $45
track jacket and the $38 Levi's; keyword scoring rewards listings whose
`style_tags` contain `vintage` / `graphic tee`. Returns a non-empty list, top
result: **"Y2K Baby Tee — Butterfly Print"**, $18.00, depop, condition
excellent, `style_tags: ["y2k","vintage","graphic tee","cottagecore"]`.
Because the list is non-empty, the retry ladder is skipped entirely →
`session["search_results"]`, next action `evaluate`.

**Step 2 — `compare_price(<baby tee>, <search results>)` [stretch].**
Compares against other `tops` sharing style tags. Returns
`{"verdict": "great deal", "item_price": 18.0, "median_comparable": 26.0,
"pct_diff": -30.8, "n_comparables": 9, "reasoning": "..."}`. The verdict is not
`"overpriced"`, so **no swap** — `session["selected_item"]` stays the baby tee.
Next action `trends`.

**Step 3 — `get_trending_styles(size=None)` [stretch].**
Returns the top tags across the catalogue, e.g.
`["vintage", "streetwear", "y2k", "minimal", "grunge"]` with counts. Stored in
`session["trends"]`, passed into the next call. Next action `outfit`.

**Step 4 — `suggest_outfit(new_item=session["selected_item"], wardrobe=<10 items>, trends=[...])`.**
The item dict is handed over directly from the session — the user never re-types
it. The wardrobe contains the baggy straight-leg jeans and chunky white sneakers
the user mentioned, so the LLM returns something like: *"Wear the butterfly baby
tee with your baggy dark-wash jeans and the chunky white sneakers — the cropped
length works with that high waist, so let the hem sit right at the waistband.
Layer the vintage black denim jacket over it when it cools off; y2k is one of
the most common tags in the catalogue right now, so this leans current."*
Stored in `session["outfit_suggestion"]`. Next action `fit_card`.

**Step 5 — `create_fit_card(outfit=session["outfit_suggestion"], new_item=session["selected_item"])`.**
Both arguments come straight out of the session. Returns something like:
*"$18 butterfly baby tee off depop and it already feels like it's been in my
closet for years 🦋 wearing it with the baggy jeans + chunky white sneakers,
denim jacket on standby. thrift gods came through."*
Next action `done` → the style profile is saved (`style_tags` gains `vintage`,
`graphic tee`; `typical_max_price` = 30.0) and the session is returned.

**Final output to user:**
Four panels. **🛍️ Listing:** "Y2K Baby Tee — Butterfly Print — $18.00 · depop ·
excellent · size S/M", followed by the price verdict ("great deal — 31% below
the $26 median for 9 comparable tops"). **👗 Outfit:** the styling paragraph
from Step 4. **✨ Fit card:** the caption from Step 5. **🧭 Agent trace:** the
ordered `log` — parse → search (3 results) → evaluate (great deal, kept top
result) → trends → outfit → fit card — so the user can see which tools ran and
why.

**Contrast — the error path for "designer ballgown size XXS under $5":**
parse → search returns `[]` → retry rung 1 (drop size) → still `[]` → rung 2
(budget to $10) → still `[]` → rung 3 (strongest keyword "ballgown") → still
`[]` → ladder exhausted → `session["error"]` is set with the specific message
and the loop goes straight to `done`. `suggest_outfit` and `create_fit_card` are
never called; `session["fit_card"]` is `None`. Three tool calls in the happy
path, one (retried) in the error path — the agent does not run a fixed sequence.
