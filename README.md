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
tests/           pytest suite (hermetic, uses the mock)
```

## Tests

```bash
python -m pytest -q
```

## Notes & limitations

- Stance/polarity is lexicon-based, which is fast and dependency-light but not as nuanced as an LLM classifier. The lexicon lives at the top of `engine.py` and is easy to extend, or you can swap in a model-based scorer behind the same `DifferenceEngine.analyze` interface.
- The engine measures what sources *say* and how they line up — it does not verify whether claims are *true*. Treat the verdict as "what the gathered logic says", with a human in the loop for fact-checking.
