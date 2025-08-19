#!/usr/bin/env python3
import json
import sys
import os

DEFAULT_FILE = "../data/outputs/1/decision_log.jsonl"

# Use first argument if given, otherwise fall back to default
infile = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

if not os.path.exists(infile):
    print(f"Error: file not found: {infile}")
    sys.exit(1)

with open(infile) as f:
    first = json.loads(f.readline())
    for line in f:
        pass
    last = json.loads(line)

def portfolio_value(d):
    return d["cash"] + d["btc"] * d["price"]

pv0 = portfolio_value(first)
pv1 = portfolio_value(last)

p0, p1 = first["price"], last["price"]

pratio = pv1 / pv0 if pv0 else float("nan")
bratio = p1 / p0 if p0 else float("nan")
outperf = pratio / bratio if bratio else float("nan")

print(f"Outperformance: {outperf:.2f}  BTC_Ratio: {bratio:.2f}  Portfolio_Ratio: {pratio:.2f}")
#print(f"{outperf:.2f}")
