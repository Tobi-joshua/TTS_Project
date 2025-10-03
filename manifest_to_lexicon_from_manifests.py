#!/usr/bin/env python3
"""
Build lexicon.json by scanning manifest.csv files under a root tree.

Fixed behavior:
 - Prefer the 'orig' column to derive the orthographic key (e.g. "Ra3" -> "ra")
 - When using arpabet_name tokens, strip uniqueness suffixes like "_1", "_5" so token names match DB tokens.
Usage:
  python3 manifest_to_lexicon_from_manifests.py --root ./syllables --out lexicon_from_manifests.json
"""
import os
import csv
import json
import argparse
import re
import shutil
from collections import defaultdict

def orth_key_from_parsed(parsed):
    # If parsed looks like "Ra/3" or "ra/3" try the left-side; otherwise None
    if not parsed:
        return None
    # if parsed contains '/', left side is often orthographic (e.g., "Ra/3")
    left = parsed.split('/')[0].strip().lower()
    if left:
        # remove any non-alphanumeric (defensive) and return
        return re.sub(r'[^0-9a-z]', '', left)
    return None

def orth_key_from_orig(orig):
    """
    Derive orthographic key from the 'orig' column (filename or label).
    Examples:
      "Ra3.wav" -> "ra"
      "Ra3" -> "ra"
      "Ra3 (1).wav" -> "ra"
      "aad1.wav" -> "aad"
    """
    if not orig:
        return None
    # remove extension if present
    base = os.path.splitext(orig)[0]
    # remove parenthetical parts
    base = re.sub(r'\(.*?\)', '', base)
    # remove non-alphanumeric
    base = re.sub(r'[^0-9a-zA-Z]', '', base).lower()
    # strip trailing tone digits (1/2/3) if present
    base = re.sub(r'[123]$', '', base)
    return base if base else None

def token_from_arpabet(arp):
    """
    Convert an arpabet_name like 'R-AA1.wav' or 'R-AA1_5.wav' into a canonical token name:
      - remove extension
      - strip trailing uniqueness suffix like _<digits> (produced by make_unique_path)
      - strip trailing spaces/parenthesis fragments if any
    Returns token stem, or None if arp empty.
    """
    if not arp:
        return None
    stem = os.path.splitext(arp.strip())[0]
    # remove trailing parenthetical bits if present: "B_IY_(Bi1i2)" -> "B_IY"
    stem = re.sub(r'\(.*\)$', '', stem).strip()
    # remove a uniqueness suffix like _1, _5, _12 appended by make_unique_path
    stem = re.sub(r'_[0-9]+$', '', stem)
    # normalize multiple underscores/spaces
    stem = stem.strip()
    return stem if stem else None

def sanitize_manifest_destname(root, dirpath):
    rel = os.path.relpath(dirpath, root)
    if rel == ".":
        base = "root"
    else:
        base = rel
    name = re.sub(r'[\\/]+', '_', base)
    name = name.replace(' ', '_')
    name = re.sub(r'[^0-9A-Za-z_\-]', '', name)
    return f"{name}_manifest.csv"

def collect_manifests_into(manifests_dir, root):
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
                        # prefer orig (original orthographic label) to derive orth key
                        orig = (row.get("orig") or row.get("Orig") or "").strip()
                        parsed = (row.get("parsed") or row.get("Parsed") or row.get("parsed_text") or "").strip()
                        arp = (row.get("arpabet_name") or row.get("arpabet") or row.get("arpabet_name ") or "").strip()

                        # derive key: prefer orig, else parsed fallback
                        key = orth_key_from_orig(orig) or orth_key_from_parsed(parsed)
                        token = token_from_arpabet(arp)

                        if key and token:
                            lex[key].append(token)
            except Exception as e:
                print("Failed to read", path, ":", e)

    # dedupe token lists and sort for stable output
    out = {}
    for k, v in lex.items():
        # preserve order then dedupe, then sort for predictable order
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