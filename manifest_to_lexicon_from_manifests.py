#!/usr/bin/env python3
"""
Build lexicon.json by scanning manifest.csv files under a root tree.
First, copy every per-sentence manifest.csv into a central manifests/ folder.

Produces a JSON mapping: orthographic_key -> [tokenName, ...]
Usage:
  python3 manifest_to_lexicon_from_manifests.py --root ./syllables --out lexicon_from_manifests.json
Options:
  --manifests-dir  Destination folder for aggregated manifests (default: ./manifests)
  --copy-manifests If provided (default: on), copy found per-sentence manifest.csv into manifests-dir.
"""

import os
import csv
import json
import argparse
import re
import shutil
from collections import defaultdict

def orth_key_from_parsed(parsed):
    # parsed is like "Ra/3" or "aad/1" -> return "ra" or "aad"
    if not parsed:
        return None
    return parsed.split('/')[0].strip().lower()

def token_from_arpabet(arp):
    if not arp:
        return None
    return os.path.splitext(arp.strip())[0]

def sanitize_manifest_destname(root, dirpath):
    """
    Produce a filesystem-safe filename for a manifest copy based on the directory path.
    Example: root=./syllables, dirpath=./syllables/Sentence 1 -> 'Sentence_1_manifest.csv'
    For deeper paths replace os.sep with '_' and remove unsafe chars.
    """
    rel = os.path.relpath(dirpath, root)
    if rel == ".":
        base = "root"
    else:
        base = rel
    # replace os separators and spaces with underscore, remove other unsafe chars
    name = re.sub(r'[\\/]+', '_', base)
    name = name.replace(' ', '_')
    name = re.sub(r'[^0-9A-Za-z_\-]', '', name)
    return f"{name}_manifest.csv"

def collect_manifests_into(manifests_dir, root):
    """
    Walk the root tree and copy every manifest.csv found into manifests_dir.
    Returns list of paths (copied files in manifests_dir).
    """
    os.makedirs(manifests_dir, exist_ok=True)
    copied = []
    for dirpath, dirs, files in os.walk(root):
        if "manifest.csv" in files:
            src = os.path.join(dirpath, "manifest.csv")
            dst_name = sanitize_manifest_destname(root, dirpath)
            dst = os.path.join(manifests_dir, dst_name)
            try:
                shutil.copy2(src, dst)
                copied.append(dst)
            except Exception as e:
                print(f"[WARN] Failed to copy {src} -> {dst}: {e}")
    return copied

def build_from_manifests_dir(manifests_root):
    lex = defaultdict(list)
    for dirpath, dirs, files in os.walk(manifests_root):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, newline='', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        parsed = row.get("parsed") or row.get("Parsed") or row.get("parsed_text") or ""
                        arp = row.get("arpabet_name") or row.get("arpabet") or row.get("arpabet_name ") or ""
                        orig = row.get("orig") or row.get("Orig") or ""
                        key = orth_key_from_parsed(parsed)
                        token = token_from_arpabet(arp)
                        # fallback: if parsed missing, use the orig filename (remove non-alnum)
                        if not key and orig:
                            key = re.sub(r'[^0-9a-z]', '', os.path.splitext(orig)[0].lower())
                        if key and token:
                            lex[key].append(token)
            except Exception as e:
                print("Failed to read", path, ":", e)
    # dedupe while preserving order and sort token lists
    out = {}
    for k, v in lex.items():
        # preserve first-seen order then sort (we'll keep unique but keep original ordering then sort by token)
        seen = list(dict.fromkeys(v))
        out[k] = sorted(seen)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./syllables", help="Root tree to search for per-sentence manifest.csv files")
    p.add_argument("--out", default="lexicon_from_manifests.json", help="Output lexicon JSON")
    p.add_argument("--manifests-dir", default="./manifests", help="Directory to collect per-sentence manifests into")
    p.add_argument("--no-copy", dest="copy_manifests", action="store_false", help="Don't copy manifests into manifests-dir (just scan root directly)")
    args = p.parse_args()

    root = args.root
    manifests_dir = args.manifests_dir
    copy_flag = args.copy_manifests

    if not os.path.isdir(root):
        print(f"[ERR] root '{root}' does not exist or is not a directory.")
        raise SystemExit(1)

    if copy_flag:
        print(f"[INFO] Collecting per-sentence 'manifest.csv' files from '{root}' into '{manifests_dir}' ...")
        copied = collect_manifests_into(manifests_dir, root)
        print(f"[INFO] Copied {len(copied)} manifests into {manifests_dir}")
        scan_root = manifests_dir
    else:
        print(f"[INFO] --no-copy set; scanning manifests directly under root: {root}")
        scan_root = root

    lex = build_from_manifests_dir(scan_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(lex, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(lex)} lexicon entries to {args.out}")

if __name__ == "__main__":
    main()