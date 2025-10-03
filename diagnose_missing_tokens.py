#!/usr/bin/env python3
"""
diagnose_missing_tokens.py

Usage:
  python3 diagnose_missing_tokens.py            # project-wide checks
  python3 diagnose_missing_tokens.py --sentence "Sentence 1"
  python3 diagnose_missing_tokens.py --text "aad san wusg zaa me bre"

It:
 - checks manifests/*.csv arpabet_name -> file exists in syllables_all_links
 - checks lexicon files vs files in syllables_all_links
 - simulates mapper.map_sentence for a sample text and shows unknown tokens
"""
import os, csv, json, argparse, sys, re

ROOT = os.path.abspath(os.path.dirname(__file__))
DB_LINKS = os.path.join(ROOT, "syllables_all_links")
MANIFESTS = os.path.join(ROOT, "manifests")
CANONICAL_LEX = os.path.join(ROOT, "lexicon_canonical.json")
MANIFEST_LEX = os.path.join(ROOT, "lexicon_from_manifests.json")
FILTERED_LEX = os.path.join(ROOT, "lexicon_filtered_all.json")


def list_db_tokens():
    if not os.path.isdir(DB_LINKS):
        print(f"[ERR] DB folder not found: {DB_LINKS}")
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(DB_LINKS) if f.lower().endswith(".wav")}


def check_manifests(db_tokens):
    if not os.path.isdir(MANIFESTS):
        print(f"[WARN] manifests dir not found: {MANIFESTS}")
        return []
    missing = []
    for fn in sorted(os.listdir(MANIFESTS)):
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(MANIFESTS, fn)
        try:
            with open(path, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    arp = (row.get("arpabet_name") or "").strip()
                    if not arp:
                        continue
                    tok = os.path.splitext(arp)[0]
                    if tok not in db_tokens:
                        missing.append((fn, arp))
        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
    return missing


def check_lexicon(db_tokens, lexpath):
    if not os.path.isfile(lexpath):
        return None
    try:
        with open(lexpath, "r", encoding="utf-8") as f:
            lex = json.load(f)
    except Exception as e:
        print(f"[WARN] failed to load {lexpath}: {e}")
        return None
    missing = {}
    for key, toks in lex.items():
        for t in toks:
            tnorm = re.sub(r'_[0-9]+$', '', t)
            if t not in db_tokens and tnorm not in db_tokens:
                missing.setdefault(key, []).append(t)
    return missing


def simulate_mapping(text):
    # Try to import classes from tonal_tts_full.py to get exact mapper behavior
    try:
        import tonal_tts_full as ttf  # expects file tonal_tts_full.py in same folder
        db = ttf.SyllableDB(DB_LINKS)
        mapper = ttf.TextToSyllableMapper(db, lexicon_path=CANONICAL_LEX if os.path.isfile(CANONICAL_LEX) else None)
        if any(ch.isdigit() for ch in text):
            tokens = text.split()
        else:
            tokens = mapper.map_sentence(text)
        unknowns = [t for t in tokens if db.get(t) is None]
        return tokens, unknowns
    except Exception as e:
        # fallback: use lexicon files as best-effort mapping
        lexfile = CANONICAL_LEX if os.path.isfile(CANONICAL_LEX) else (MANIFEST_LEX if os.path.isfile(MANIFEST_LEX) else FILTERED_LEX)
        if not os.path.isfile(lexfile):
            return [], []
        with open(lexfile, "r", encoding="utf-8") as f:
            lex = json.load(f)
        mapped = []
        unknowns = []
        for w in text.strip().split():
            key = w.lower()
            if key in lex and lex[key]:
                mapped.extend(lex[key])
            else:
                unknowns.append(w)
        return mapped, unknowns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", default=None, help="Sentence folder to inspect (e.g. 'Sentence 1')")
    p.add_argument("--text", default=None, help="Sample text to simulate mapping")
    args = p.parse_args()

    db_tokens = list_db_tokens()
    print(f"DB tokens found: {len(db_tokens)} (example: {list(db_tokens)[:10]})")

    missing_from_manifests = check_manifests(db_tokens)
    if missing_from_manifests:
        print("\n[MANIFESTS] Missing ARPABET files referenced from manifests (manifest_file, arpabet_name):")
        for m in missing_from_manifests[:200]:
            print(" ", m)
    else:
        print("\n[MANIFESTS] All manifests' arpabet_name entries exist in DB_LINKS (or manifests dir missing).")

    # check lexicons
    print("\n[LEXICON CHECK] canonical & manifest-derived:")
    for lex in [CANONICAL_LEX, MANIFEST_LEX, FILTERED_LEX]:
        if os.path.isfile(lex):
            miss = check_lexicon(db_tokens, lex)
            try:
                with open(lex, "r", encoding="utf-8") as fh:
                    total_keys = len(json.load(fh))
            except Exception:
                total_keys = "?"
            missing_count = len(miss) if isinstance(miss, dict) else "N/A"
            print(f"  {os.path.basename(lex)}: keys={total_keys}  missing token keys={missing_count}")
            if isinstance(miss, dict) and miss and len(miss) < 200:
                print("   sample missing entries:")
                for k, v in list(miss.items())[:50]:
                    print("    ", k, "->", v)
        else:
            print(f"  {os.path.basename(lex)}: NOT FOUND")

    # sentence manifest inspect
    if args.sentence:
        mf_guess = os.path.join(MANIFESTS, f"{args.sentence.replace(' ', '_')}_manifest.csv")
        mf = mf_guess if os.path.isfile(mf_guess) else None
        if not mf:
            # find any manifest starting with the sentence name
            cand = None
            if os.path.isdir(MANIFESTS):
                for f in os.listdir(MANIFESTS):
                    if f.lower().startswith(args.sentence.replace(' ', '_').lower()):
                        cand = os.path.join(MANIFESTS, f)
                        break
            mf = cand
        if mf and os.path.isfile(mf):
            print("\n[SENTENCE] Inspecting manifest:", mf)
            with open(mf, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                print(" manifest rows:", len(rows))
                not_found = []
                for r in rows:
                    a = (r.get("arpabet_name") or "").strip()
                    if a and os.path.splitext(a)[0] not in db_tokens:
                        not_found.append(a)
                print(" missing arpabet files in DB for that sentence:", not_found[:200])
        else:
            print("[SENTENCE] manifest not found for", args.sentence)

    # simulate mapping of text
    if args.text:
        tokens, unknowns = simulate_mapping(args.text)
        print("\n[SIM] mapped tokens:", tokens)
        print("[SIM] unknown tokens (not found in DB):", unknowns)
    else:
        # try a tiny example mapping from canonical lex keys if available
        if os.path.isfile(CANONICAL_LEX):
            try:
                some_keys = list(json.load(open(CANONICAL_LEX, encoding='utf-8')).keys())[:6]
                sample = " ".join(some_keys)
                tokens, unknowns = simulate_mapping(sample)
                print("\n[SIM] example mapping for some lexicon keys:", sample)
                print(" tokens:", tokens[:40])
                print(" unknowns:", unknowns[:20])
            except Exception:
                pass


if __name__ == "__main__":
    main()