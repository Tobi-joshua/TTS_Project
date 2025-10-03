#!/usr/bin/env python3
"""
build_clean_manifests.py

Scan each Sentence folder's "Arpabet Transcription" folder, parse the ARPABET
filenames, pick the best matching original WAV in the Sentence root, and write
a clean manifest.csv per-sentence plus a copy into ./manifests/.

Usage:
  python3 build_clean_manifests.py ./syllables

Options:
  --out-dir DIR      Copy per-sentence manifests into DIR (default: ./manifests)
  --dry-run          Don't write files; print what would be done
  --verbose          Print more information
"""

import os, re, csv, argparse, shutil
from pathlib import Path

# -----------------------
# Parsing helpers (from your map_to_arpabet style)
# -----------------------
def sanitize_filename_token(tok):
    return re.sub(r'[^A-Za-z0-9]', '', tok).strip()

def parse_arpabet_transcription_filename(fname):
    # Skip ADS/Zone.Identifier extras
    if fname.endswith(":Zone.Identifier"):
        return None
    stem = os.path.splitext(fname)[0]
    # find parentheses content (original label)
    m = re.search(r'\(([^)]+)\)', stem)
    orig = None
    tokens_part = stem
    if m:
        orig = m.group(1).strip()
        tokens_part = stem[:m.start()].rstrip('_ ').strip()
    else:
        tokens_part = stem

    if not tokens_part:
        return None

    raw_items = [it.strip() for it in re.split(r'[_]+', tokens_part) if it.strip() != '']
    if not raw_items:
        return None

    syllables = []
    cur_tokens = []
    for it in raw_items:
        it_clean = it.replace(' ', '')
        m2 = re.match(r'^([A-Za-z]+)(\d+)$', it_clean)
        if m2:
            tkn = m2.group(1)
            tone = m2.group(2)
            cur_tokens.append(tkn)
            syllables.append(([sanitize_filename_token(x).upper() for x in cur_tokens], tone))
            cur_tokens = []
            continue
        if re.fullmatch(r'\d+', it_clean):
            if cur_tokens:
                syllables.append(([sanitize_filename_token(x).upper() for x in cur_tokens], it_clean))
                cur_tokens = []
            else:
                # stray numeric item with no preceding tokens -> ignore
                continue
        else:
            cur_tokens.append(sanitize_filename_token(it_clean).upper())

    # leftover tokens become a syllable with default tone "2"
    if cur_tokens:
        syllables.append(([sanitize_filename_token(x).upper() for x in cur_tokens], "2"))

    if len(syllables) == 0:
        return None
    return orig, syllables

def arpabet_syllables_to_stem(syllables):
    parts = []
    for tokens, tone in syllables:
        if not tokens:
            continue
        joined = "-".join(tokens)
        parts.append(f"{joined}{tone if tone else '2'}")
    stem = "_".join(parts)
    stem_sanit = re.sub(r'[^A-Z0-9_\-]', '_', stem.upper())
    return stem_sanit

# -----------------------
# Matching heuristics to find "best" original wav
# -----------------------
IGNORED_KEYWORDS = ['expression', 'practice', 'unknown', 'mono', 'demo']

def normalize(s: str):
    # Lowercase, remove parens and extra spaces, remove trailing numeric duplicates like " (1)"
    s2 = s.lower()
    s2 = re.sub(r'\(.*?\)', '', s2)   # remove parentheses pieces
    s2 = re.sub(r'[^a-z0-9]', '', s2) # keep alnum only for matching
    return s2

def score_candidate(orig_label: str, candidate_filename: str):
    """
    Heuristic scoring:
      - exact stem match (case-insensitive) -> best
      - stem startswith orig_label -> good
      - candidate contains orig_label -> ok
      - penalize if candidate name contains IGNORED_KEYWORDS (e.g., 'expression')
      - shorter filenames preferred
    Returns a tuple (score, tie_breaker) where higher score better.
    """
    stem = os.path.splitext(candidate_filename)[0]
    n_orig = normalize(orig_label)
    n_stem = normalize(stem)

    score = 0
    # exact match
    if n_stem == n_orig:
        score += 100
    # exact match ignoring trailing numeric parentheses, spaces etc.
    if n_stem.startswith(n_orig) and len(n_stem) - len(n_orig) <= 3:
        score += 50
    # contains
    if n_orig in n_stem:
        score += 20
    # penalize messy names
    low = stem.lower()
    for kw in IGNORED_KEYWORDS:
        if kw in low:
            score -= 20
    # penalize long names
    score -= max(0, (len(stem) - len(orig_label)) // 3)
    # prefer shorter overall path (tie breaker)
    tie = -len(stem)
    return (score, tie)

def choose_best_original(orig_label: str, candidates: list):
    """
    Choose best candidate WAV filename from list of filenames in sentence folder.
    Returns filename (basename) or None.
    """
    if not candidates:
        return None
    # if a file exactly equals orig_label(.wav) ignore case -> pick it
    orig_base = orig_label
    if not orig_base.lower().endswith('.wav'):
        orig_base = orig_base + '.wav'
    for c in candidates:
        if c.lower() == orig_base.lower():
            return c

    # else score all candidates, pick highest
    scored = [(score_candidate(orig_label, c), c) for c in candidates]
    scored.sort(reverse=True)
    return scored[0][1]

# -----------------------
# Main per-sentence processor
# -----------------------
def build_manifest_for_sentence(sentence_dir: str, dry_run=False, verbose=False):
    """
    Process one Sentence folder: parse Arpabet Transcription files and build manifest rows.
    Writes manifest.csv into sentence_dir (unless dry_run=True).
    Returns list of manifest rows (dicts).
    """
    arpa_dir = os.path.join(sentence_dir, "Arpabet Transcription")
    if not os.path.isdir(arpa_dir):
        if verbose:
            print(f"[INFO] no 'Arpabet Transcription' in {sentence_dir}, skipping.")
        return []

    files = sorted(os.listdir(arpa_dir))
    # candidate originals in sentence dir (list only wavs at top-level)
    candidates = [f for f in os.listdir(sentence_dir) if f.lower().endswith('.wav')]

    rows = []
    seen_arpabet_names = set()
    for fn in files:
        if fn.lower().endswith('.mp3'):
            if verbose: print(f"[SKIP] mp3 in Arpabet Transcription: {fn}")
            continue
        parsed = parse_arpabet_transcription_filename(fn)
        if not parsed:
            if verbose: print(f"[WARN] could not parse: {fn}")
            continue
        orig_label, syllables = parsed
        if not orig_label:
            # fallback to stem without parentheses
            orig_label = os.path.splitext(fn)[0]
        arpabet_stem = arpabet_syllables_to_stem(syllables)
        arpabet_fname = f"{arpabet_stem}.wav"

        # avoid duplicate ARPABET entries mapping to same arpabet_fname
        if arpabet_fname in seen_arpabet_names:
            if verbose:
                print(f"[DUP] skipping duplicate arpabet stem {arpabet_fname} from {fn}")
            continue
        seen_arpabet_names.add(arpabet_fname)

        # choose best original - prefer exact matches; otherwise heuristic
        matched = choose_best_original(orig_label, candidates)
        if matched is None:
            # fallback try {orig_label}.wav strictly
            test = f"{orig_label}.wav"
            if test in candidates:
                matched = test
            else:
                # still none -> use a synthesized name (orig_label + .wav) (this will NOT point to a real file)
                matched = f"{orig_label}.wav"

        parsed_parts = []
        # build parsed string similar to your other scripts (e.g., "Ra/3")
        for toks, tone in syllables:
            # re-create orth label from tokens if possible; best-effort: join tokens with '-'
            parsed_parts.append(f"{'-'.join(toks)}/{tone if tone else '2'}")
        parsed_str = ";".join(parsed_parts)

        row = {
            "orig": matched,
            "parsed": parsed_str,
            "arpabet_name": arpabet_fname,
            "unmapped": ""
        }
        rows.append(row)
        if verbose:
            print(f"[ROW] {fn} -> arpabet:{arpabet_fname}  orig_match:{matched}")

    # write manifest.csv
    manifest_path = os.path.join(sentence_dir, "manifest.csv")
    if dry_run:
        if verbose:
            print(f"[DRY] would write {len(rows)} rows to {manifest_path}")
        return rows

    # backup old manifest if exists
    if os.path.exists(manifest_path):
        bak = manifest_path + ".bak"
        shutil.copy2(manifest_path, bak)

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["orig","parsed","arpabet_name","unmapped"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    if verbose:
        print(f"[WROTE] {manifest_path} ({len(rows)} rows)")
    return rows

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Build cleaned per-sentence manifest.csv from Arpabet Transcription folders.")
    p.add_argument("root", help="Root folder (e.g., ./syllables)")
    p.add_argument("--out-dir", default="./manifests", help="Directory to copy per-sentence manifests into (default ./manifests)")
    p.add_argument("--dry-run", action="store_true", help="Don't write manifests; only print actions")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    root = args.root
    out_dir = args.out_dir
    dry = args.dry_run
    verbose = args.verbose

    if not os.path.isdir(root):
        print(f"[ERR] root not found: {root}")
        return

    # find Sentence* folders
    folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d)) and d.lower().startswith("sentence")]
    if not folders:
        print("[WARN] no Sentence* folders found under", root)
        return

    # ensure out_dir exists (unless dry-run)
    if not dry:
        os.makedirs(out_dir, exist_ok=True)

    all_rows_total = 0
    for f in folders:
        if verbose: print(f"[PROCESS] {f}")
        rows = build_manifest_for_sentence(f, dry_run=dry, verbose=verbose)
        all_rows_total += len(rows)
        # copy manifest to out_dir
        if not dry and rows:
            src = os.path.join(f, "manifest.csv")
            dst_name = re.sub(r'[\\/]+', '_', os.path.relpath(f, root))
            dst_name = dst_name.replace(' ', '_') + "_manifest.csv"
            dst = os.path.join(out_dir, dst_name)
            try:
                shutil.copy2(src, dst)
                if verbose: print(f"[COPIED] {src} -> {dst}")
            except Exception as e:
                print(f"[WARN] failed to copy manifest for {f}: {e}")

    print(f"[DONE] processed {len(folders)} sentence folders. Total rows: {all_rows_total}. Manifests in: {out_dir if not dry else '(dry-run)'}")

if __name__ == "__main__":
    main()