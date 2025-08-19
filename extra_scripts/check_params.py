#!/usr/bin/env python3
"""
Scan Python files for config parameter usage.

- Loads parameters from config/config.json
- Recursively scans ALL Python files under src/ (src/**/*.py)
- Reports lines where each parameter is used
- Lists parameters that were never referenced

Usage:
    python check_params.py
"""

import json
import os
import io
import ast
import tokenize
from collections import defaultdict
from pathlib import Path

# --------- CONFIG ---------
CONFIG_PATH = Path("config/config.json")
SRC_ROOT = Path("src")  # recursively scans under this folder
# --------------------------

def load_config_params(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaf_paths = []  # dotted paths like "ml_buy.enabled"
    string_only_keys = set()  # keys that can't be Python identifiers (e.g., "rsi>65")

    def walk(obj, prefix=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                walk(v, path)
        else:
            # reached a leaf (non-dict)
            leaf_paths.append(prefix)

    walk(data)

    # Unique leaf key names (last component)
    leaf_names = set(p.split(".")[-1] for p in leaf_paths)

    # Keys that are not valid identifiers -> must be found as string literals
    for p in leaf_paths:
        k = p.split(".")[-1]
        if not k.isidentifier():
            string_only_keys.add(k)

    return leaf_paths, leaf_names, string_only_keys

def list_python_files():
    return [str(p) for p in SRC_ROOT.rglob("*.py")]

def tokenize_file(path):
    """
    Return per-line token info:
      - names_by_line: dict[line_no] -> set of NAME tokens (identifiers)
      - strings_by_line: dict[line_no] -> set of STRING literal contents (unquoted)
      - ops_by_line: dict[line_no] -> list of operator tokens ('.', '[', etc.)
    """
    names_by_line = defaultdict(set)
    strings_by_line = defaultdict(set)
    ops_by_line = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()

    try:
        tokgen = tokenize.generate_tokens(io.StringIO(src).readline)
    except Exception:
        return names_by_line, strings_by_line, ops_by_line, src.splitlines()

    for tok in tokgen:
        ttype, tstr, (srow, _), (_erow, _ecol), _line = tok
        if ttype == tokenize.NAME:
            names_by_line[srow].add(tstr)
        elif ttype == tokenize.STRING:
            # Safely get literal string content (handles single/double/triple quotes, r/u/b prefixes)
            val = tstr
            try:
                lit = ast.literal_eval(tstr)
                if isinstance(lit, str):
                    strings_by_line[srow].add(lit)
                # ignore bytes etc.
            except Exception:
                # Fallback: naive strip if symmetrical quotes
                if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0]:
                    strings_by_line[srow].add(val[1:-1])
                else:
                    strings_by_line[srow].add(val)
        elif ttype == tokenize.OP:
            ops_by_line[srow].append(tstr)

    return names_by_line, strings_by_line, ops_by_line, src.splitlines()

def find_usages(files, leaf_paths, leaf_names, string_only_keys):
    """
    Heuristics:
      - If leaf is a valid identifier: match NAME tokens and STRING tokens.
      - If leaf is not an identifier: match STRING tokens only.
      - Also try dotted parent.child on a single line when we see the parent NAME + child NAME/STRING
        and an attribute/index operator present ('.' or '[').
    """
    usage = defaultdict(list)  # dotted leaf path -> list of (file, line_no, line_text)

    # Build parent->children map from dotted leaf paths
    parent_children = defaultdict(set)
    for p in leaf_paths:
        parts = p.split(".")
        if len(parts) >= 2:
            parent = parts[-2]
            child = parts[-1]
            parent_children[parent].add(child)

    for path in files:
        names_by_line, strings_by_line, ops_by_line, lines = tokenize_file(path)

        max_line = len(lines)
        for lineno in range(1, max_line + 1):
            names = names_by_line.get(lineno, set())
            strs = strings_by_line.get(lineno, set())
            ops = ops_by_line.get(lineno, [])

            # 1) Leaf-name matches
            for leaf in leaf_names:
                dotted_candidates = [p for p in leaf_paths if p.split(".")[-1] == leaf]

                matched = False
                if leaf in string_only_keys:
                    if leaf in strs:
                        matched = True
                else:
                    if (leaf in names) or (leaf in strs):
                        matched = True

                if matched:
                    for dotted in dotted_candidates:
                        usage[dotted].append((path, lineno, lines[lineno-1].rstrip()))

            # 2) Dotted parent.child on the same line
            for parent, children in parent_children.items():
                if parent in names:
                    for child in children:
                        if (child in names) or (child in strs):
                            if ('.' in ops) or ('[' in ops):
                                dotted = f"{parent}.{child}"
                                if dotted in leaf_paths:
                                    usage[dotted].append((path, lineno, lines[lineno-1].rstrip()))

    return usage

def main():
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        return

    leaf_paths, leaf_names, string_only_keys = load_config_params(CONFIG_PATH)
    files = list_python_files()

    if not files:
        print(f"No Python files found under {SRC_ROOT!s}.")
        return

    usage = find_usages(files, leaf_paths, leaf_names, string_only_keys)

    used = {k for k, hits in usage.items() if hits}
    unused = [p for p in leaf_paths if p not in used]

    # --- Report ---
    print("\n=== PARAMETER USAGE REPORT ===\n")
    print(f"Scanned {len(files)} files under {SRC_ROOT}/\n")

    for param in sorted(leaf_paths):
        hits = usage.get(param, [])
        if hits:
            print(f"[USED] {param}  ({len(hits)} hit{'s' if len(hits)!=1 else ''})")
            for fpath, lineno, text in hits:
                rel = os.path.relpath(fpath)
                print(f"  {rel}:{lineno}: {text}")
            print()
    if not used:
        print("No parameters appeared to be used.\n")

    print("=== UNUSED PARAMETERS ===")
    if unused:
        for p in sorted(unused):
            print(f"  {p}")
    else:
        print("  (none)")

if __name__ == "__main__":
    main()
