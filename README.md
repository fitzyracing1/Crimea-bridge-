# Difference Engine

A small agent pipeline that **pulls data from the [eartheareconputer.com](https://eartheareconputer.com) API, scores it with a "difference engine", sorts it, lays it out as a relationship graph, and reads off what the collective logic says about a topic** — while keeping a full log of every record pulled in a search.

```
pull (API)  ->  score & difference (engine)  ->  sort  ->  graph + clusters  ->  log all data
```

## What it does

1. **Pulls** records for a search term from the eartheareconputer.com API (`GET /search?q=...` with a bearer token). If the API key is missing or the host is unreachable, it falls back to a deterministic offline **mock** so the whole pipeline still runs.
2. **Scores** every record with the difference engine:
   - **relevance** — how much the record overlaps the topic (0…1)
   - **polarity** — sentiment of the text using a lexicon, with negation handling (−1…1)
   - **stance** — topic-directed support/opposition (−1 oppose … +1 support)
   - **support score** & **confidence**
3. Computes the **pairwise difference** between every record (a content component from TF‑IDF cosine distance + a stance component) and classifies each pair as **agrees / contradicts / differs**.
4. **Sorts** the records (by support, stance, relevance, or confidence).
5. Builds a **graph** (nodes = records colored by stance; edges = agreement/contradiction), detects **logic clusters** (communities that line up) and **centrality**, and renders it to PNG/JSON/GraphML.
6. Derives a **verdict** — what the logic says (leans support / leans oppose / contested / mixed).
7. **Logs all pulled data** for every search to `logs/` (a master `searches.jsonl` plus a full per-search JSON).

## Install

```bash
pip install -r requirements.txt
```

`requests`, `numpy` and `networkx` are required. `matplotlib` is optional and only used to render the PNG graph (everything else works without it).

## Usage

### Command line

```bash
# offline mock (no API needed) — great for trying it out
python -m difference_engine "electric vehicles" --mock

# against the live API
EARTHEAR_API_KEY=your_token python -m difference_engine "electric vehicles"

# options
python -m difference_engine "remote work" --limit 15 --sort stance
python -m difference_engine "solar power" --mock --json     # machine-readable output
python -m difference_engine "topic" --mock --no-image       # skip PNG rendering
```

Example (abridged) output:

```
WHAT THE LOGIC SAYS
  verdict        : contested: sources contradict each other with no clear majority
  net support    : +0.020   (-1 oppose … +1 support)
  support/oppose/neutral : 4 / 4 / 4
  agreement links: 18   contradiction links: 16

LOGIC CLUSTERS (groups that line up)
  cluster 0 [opposing,   size 4, mean stance -0.92]: Critics argue electric vehicles is harmful...
  cluster 1 [supporting, size 4, mean stance +0.95]: Data shows electric vehicles works well...
  cluster 2 [neutral,    size 4, mean stance +0.00]: Open questions remain about electric vehicles...
```

The graph image shows supporting (green) and opposing (red) clusters linked by dashed contradiction edges, with neutral/background records standing apart.

### Library

```python
from difference_engine import Config, DifferencePipeline

pipeline = DifferencePipeline(Config.from_env(use_mock=True))
result = pipeline.run("electric vehicles", limit=12)

print(result.summary["verdict"])
for idx in result.sorted_indices:
    item = result.result.items[idx]
    print(item.stance, item.record.text)

print("graph image:", result.outputs.get("graph_png"))
print("search log:", result.log_paths["detail"])
```

## Betting on "pairs" (head-to-head AI matchups)

> **Simulation / decision-support only.** This does **not** connect to real money or any betting venue, and a model probability is **not** a guarantee. Wagering real money requires a licensed, regulated venue and carries real risk. Use this to study value and practice discipline, not as a guaranteed edge.

The `difference_engine.markets` layer turns a pair of AI subjects (two models, companies, technologies, …) into a market:

1. the difference engine pulls and scores data about **each** subject and reduces it to a *strength* (net support/sentiment) and a *confidence*;
2. the strength gap is mapped through a logistic curve to a **model win probability** for each side (low confidence pulls the price toward 50/50);
3. that probability becomes **fair decimal odds** and **house odds** (with a configurable margin);
4. if you supply the odds a real market is offering, it computes your **edge**, **expected value**, and the **Kelly-optimal stake** — i.e. whether there is value;
5. a **paper bankroll** lets you place and settle simulated bets, and every bet is logged to `logs/bets.jsonl`.

```bash
# price a pair (offline mock data)
python -m difference_engine.markets odds "GPT-5" "Claude" --mock

# find value against the odds a real market is offering (decimal odds)
python -m difference_engine.markets odds "GPT-5" "Claude" --mock \
    --market-odds-a 1.8 --market-odds-b 2.2

# place a simulated bet, then settle it
python -m difference_engine.markets reset --bankroll 1000
python -m difference_engine.markets bet "GPT-5" "Claude" --on a --stake 100 --mock
python -m difference_engine.markets settle <bet_id> --winner a
python -m difference_engine.markets wallet
python -m difference_engine.markets history
```

Example value read-out (model thinks the underdog is mispriced):

```
Claude
  model win prob : 0.500
  fair odds      : 2.00   house odds (margin 5%): 1.90
  market odds    : 2.20 (implied 0.455)
  edge           : +0.045   EV/unit: +0.100   [VALUE (+EV)]
  Kelly fraction : 0.083   suggested stake: 83.30 UNITS

VALUE PICK : Claude  (positive expected value vs the market odds you supplied)
```

The workflow mirrors quantitative betting: your model produces a probability, the market quotes odds, and you only bet when your estimated probability beats the market's implied probability — sized by Kelly and capped (`--kelly-cap`). Run it against the **live API** by setting `EARTHEAR_API_KEY` and dropping `--mock`.

Library use:

```python
from difference_engine.markets import assess_subject, make_market, PaperBook

a = assess_subject("GPT-5", limit=20)
b = assess_subject("Claude", limit=20)
market = make_market(a, b, market_odds_a=1.8, market_odds_b=2.2, bankroll=1000)
print(market.value_pick, market.side_b.ev, market.side_b.suggested_stake)

book = PaperBook(starting_balance=1000)
bet = book.place_bet("GPT-5 vs Claude", side="GPT-5", stake=100, odds=2.5)
book.settle_bet(bet["id"], winner="GPT-5")
print(book.summary())
```

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `EARTHEAR_API_BASE_URL` | `https://eartheareconputer.com/api` | API base URL |
| `EARTHEAR_API_KEY` | _(unset)_ | Bearer token. If unset, the mock is used. |
| `EARTHEAR_SEARCH_PATH` | `/search` | Search endpoint path |
| `EARTHEAR_QUERY_PARAM` | `q` | Query string parameter for the term |
| `EARTHEAR_TIMEOUT` | `15` | Request timeout (seconds) |
| `EARTHEAR_USE_MOCK` | `false` | Force the offline mock source |
| `DIFFENGINE_LOG_DIR` | `logs` | Where search logs are written |
| `DIFFENGINE_OUTPUT_DIR` | `output` | Where graph exports are written |

> The client tolerates several response shapes: a bare JSON list, or an object wrapping the rows under `results` / `data` / `items` / `records` / `hits`. Each row is normalized using common field aliases (`text`/`content`/`title`, `source`/`url`, `created_at`/`timestamp`, `id`/`_id`). Adjust the connector in `difference_engine/client.py` if your API differs.

## Outputs

- `output/graph-<ts>-<topic>.png` — the difference graph (needs matplotlib)
- `output/graph-<ts>-<topic>.json` — node-link graph data
- `output/graph-<ts>-<topic>.graphml` — GraphML for Gephi/Cytoscape
- `logs/searches.jsonl` — master audit log, one line per search
- `logs/search-<ts>-<topic>.json` — full log of **all** records and scores for that search (API keys redacted)

## Project layout

```
difference_engine/
  config.py      env-driven configuration
  client.py      eartheareconputer.com API client + offline mock
  engine.py      the difference engine: scoring, pairwise difference, verdict
  graph.py       graph build, clustering, centrality, PNG/JSON/GraphML export
  search_log.py  append-only log of all pulled data
  pipeline.py    end-to-end orchestration
  cli.py         command-line interface
  markets.py     pairs betting layer: subject strength, odds, Kelly, paper bankroll
  markets_cli.py CLI for the betting layer
tests/           pytest suite (hermetic, uses the mock)
```

## Tests

```bash
python -m pytest -q
```

## Notes & limitations

- Stance/polarity is lexicon-based, which is fast and dependency-light but not as nuanced as an LLM classifier. The lexicon lives at the top of `engine.py` and is easy to extend, or you can swap in a model-based scorer behind the same `DifferenceEngine.analyze` interface.
- The engine measures what sources *say* and how they line up — it does not verify whether claims are *true*. Treat the verdict as "what the gathered logic says", with a human in the loop for fact-checking.
