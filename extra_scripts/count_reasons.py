#!/usr/bin/env python3
import sys, json, argparse
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(
        description="Count 'reason' values from NDJSON.",
        epilog="""Examples:
        # Read from a file
        python count_reasons.py ../data/outputs/1/decision_log.jsonl

        # Read from stdin
        cat ../data/outputs/1/decision_log.jsonl | python count_reasons.py -
        
        # Specify custom order of reasons
        python count_reasons.py ../data/outputs/1/decision_log.jsonl --order=buy,sell,hold
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "file",
        nargs="?",
        help="Input file (NDJSON). Omit or use '-' for stdin.",
    )

    p.add_argument(
        "--order",
        help="Comma-separated reasons to print first (in this exact order). "
        "Reasons not listed will follow in alphabetical order.",
    )

    return p.parse_args()

def main():
    args = parse_args()
    fh = sys.stdin if (args.file in (None, "-")) else open(args.file, "r", encoding="utf-8")

    counts = Counter()
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines
            reason = obj.get("reason")
            if isinstance(reason, str):
                counts[reason] += 1

    # Determine output order
    pre_order = []
    if args.order:
        pre_order = [x for x in (r.strip() for r in args.order.split(",")) if x]

    seen = set(pre_order)
    remaining = sorted([r for r in counts.keys() if r not in seen])

    # Print in the requested fixed order first, then the rest alphabetically
    for r in pre_order:
        print(f"{r}: {counts.get(r, 0)}")
    for r in remaining:
        print(f"{r}: {counts[r]}")

if __name__ == "__main__":
    main()
