"""Command-line interface for the difference engine.

Examples:
    python -m difference_engine "electric vehicles"
    python -m difference_engine "remote work" --limit 12 --sort stance
    EARTHEAR_API_KEY=xxx python -m difference_engine "topic"   # hits the live API
    python -m difference_engine "topic" --mock --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .pipeline import DifferencePipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="difference_engine",
        description="Score eartheareconputer.com data, sort it, graph it, and read off what the logic says.",
    )
    p.add_argument("query", help="topic / search term to investigate")
    p.add_argument("--limit", type=int, default=25, help="max records to pull (default 25)")
    p.add_argument(
        "--sort",
        default="support",
        choices=["support", "relevance", "stance", "confidence"],
        help="how to sort the pulled records (default: support)",
    )
    p.add_argument("--mock", action="store_true", help="force the offline mock data source")
    p.add_argument("--api-base-url", default=None, help="override the API base URL")
    p.add_argument("--api-key", default=None, help="override the API bearer token")
    p.add_argument("--log-dir", default=None, help="directory for the search log")
    p.add_argument("--output-dir", default=None, help="directory for graph exports")
    p.add_argument("--no-image", action="store_true", help="skip PNG rendering")
    p.add_argument("--no-outputs", action="store_true", help="skip writing graph files")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = Config.from_env(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        use_mock=True if args.mock else None,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
    )

    pipeline = DifferencePipeline(config=config)
    result = pipeline.run(
        args.query,
        limit=args.limit,
        sort_by=args.sort,
        render_image=not args.no_image,
        write_outputs=not args.no_outputs,
    )

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
        return 0

    _print_report(result)
    return 0


def _print_report(result) -> None:
    s = result.summary
    bar = "=" * 70
    print(bar)
    print(f"DIFFERENCE ENGINE  |  topic: {result.query!r}")
    print(f"source: {result.origin}   records: {len(result.records)}")
    print(bar)

    print("\nWHAT THE LOGIC SAYS")
    print(f"  verdict        : {s['verdict']}")
    print(f"  net support    : {s['net_support']:+.3f}   (-1 oppose … +1 support)")
    print(f"  support/oppose/neutral : {s['support_count']} / {s['oppose_count']} / {s['neutral_count']}")
    print(f"  agreement links: {s['agreement_edges']}   contradiction links: {s['contradiction_edges']}")

    print("\nRANKED RECORDS (sorted)")
    by_index = {i.index: i for i in result.result.items}
    for rank, idx in enumerate(result.sorted_indices, 1):
        item = by_index[idx]
        text = item.record.text
        snippet = (text[:90] + "…") if len(text) > 90 else text
        print(f"  {rank:>2}. [stance {item.stance:+.2f} | support {item.support_score:.2f} "
              f"| rel {item.relevance:.2f}] {snippet}")

    if result.clusters:
        print("\nLOGIC CLUSTERS (groups that line up)")
        for c in result.clusters:
            rep = c["representative"]
            rep = (rep[:80] + "…") if len(rep) > 80 else rep
            print(f"  cluster {c['cluster']} [{c['leaning']}, size {c['size']}, "
                  f"mean stance {c['mean_stance']:+.2f}]: {rep}")

    if s["top_supporting"]:
        print("\nTOP SUPPORTING")
        for b in s["top_supporting"]:
            print(f"  + ({b['stance']:+.2f}) {b['text']}")
    if s["top_opposing"]:
        print("\nTOP OPPOSING")
        for b in s["top_opposing"]:
            print(f"  - ({b['stance']:+.2f}) {b['text']}")

    if result.outputs:
        print("\nGRAPH OUTPUTS")
        for k, v in result.outputs.items():
            print(f"  {k}: {v}")

    print("\nSEARCH LOG")
    for k, v in result.log_paths.items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
