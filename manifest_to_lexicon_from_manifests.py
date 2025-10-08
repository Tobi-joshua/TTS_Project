#!/usr/bin/env python3
"""
Build lexicon.json by scanning manifest.csv files under a root tree.

Fixed behavior:
 - Prefer the 'orig' column to derive the orthographic key (e.g. "Ra3" -> "ra")
 - When using arpabet_name tokens, strip uniqueness suffixes like "_1", "_5" so token names match DB tokens.
 - Insert orthographic variants: digits-stripped and vowel-collapsed versions for robust mapping.
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
    if not parsed:
        return None
    left = parsed.split('/')[0].strip().lower()
    if left:
        return re.sub(r'[^0-9a-z]', '', left)
    return None

def orth_key_from_orig(orig):
    if not orig:
        return None
    base = os.path.splitext(orig)[0]
    base = re.sub(r'\(.*?\)', '', base)
    base = re.sub(r'[^0-9a-zA-Z]', '', base).lower()
    base = re.sub(r'[123]$', '', base)
    return base if base else None

def token_from_arpabet(arp):
    if not arp:
        return None
    stem = os.path.splitext(arp.strip())[0]
    stem = re.sub(r'\(.*\)$', '', stem).strip()
    stem = re.sub(r'_[0-9]+$', '', stem)
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

def collapse_repeated_vowels(s: str) -> str:
    if not s:
        return s
    return re.sub(r'([aeiou])\1{2,}', r'\1\1', s)

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
                        orig = (row.get("orig") or row.get("Orig") or "").strip()
                        parsed = (row.get("parsed") or row.get("Parsed") or row.get("parsed_text") or "").strip()
                        arp = (row.get("arpabet_name") or row.get("arpabet") or row.get("arpabet_name ") or "").strip()

                        key = orth_key_from_orig(orig) or orth_key_from_parsed(parsed)
                        token = token_from_arpabet(arp)

                        if key and token:
                            # primary insertion
                            lex[key].append(token)
                            # digits-stripped variant
                            key_nod = re.sub(r'\d+$', '', key)
                            if key_nod and key_nod != key:
                                lex[key_nod].append(token)
                            # vowel-collapsed variant
                            key_vc = collapse_repeated_vowels(key_nod)
                            if key_vc and key_vc != key_nod:
                                lex[key_vc].append(token)
            except Exception as e:
                print("Failed to read", path, ":", e)

    out = {}
    for k, v in lex.items():
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