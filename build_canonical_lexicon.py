#!/usr/bin/env python3
"""
Build a canonical lexicon: for each orthographic key pick one token (the
most likely single-syllable token) from the lexicon candidates by:
  - excluding tokens whose path looks like a Sentence or Arpabet Transcription file,
  - choosing the shortest-duration WAV among remaining candidates.

Usage:
  python build_canonical_lexicon.py --db ./syllables_all_links --lexicon lexicon_filtered_all.json --out lexicon_canonical.json
"""
import os, json, argparse, wave, contextlib, re

BAD_SUBSTRINGS = ["sentence", "arpabet", "arpabet transcription", "arpabet_db"]


def duration_ms_from_wav(path):
    # robustly compute duration in ms for a WAV file (uses wave module)
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return int((frames / float(rate)) * 1000.0)
    except Exception:
        # If wave can't read it, fall back to file size heuristic
        try:
            return int(os.path.getsize(path) / 100.0)  # arbitrary proxy
        except Exception:
            return None


def find_token_path_map(db_dir):
    """
    Build mapping token_basename -> full_path by scanning db_dir.
    If multiple files have same basename, keep first discovered (user can use sanitized symlink db).
    """
    mapping = {}
    for root, dirs, files in os.walk(db_dir):
        for fn in files:
            if not fn.lower().endswith('.wav'):
                continue
            token = os.path.splitext(fn)[0]
            if token in mapping:
                # keep first encountered to mimic SyllableDB behaviour; change if you prefer last
                continue
            mapping[token] = os.path.join(root, fn)
            # also register lowercase variant for robust lookup
            low = token.lower()
            if low not in mapping:
                mapping[low] = mapping[token]
    return mapping


def choose_best_token(key, candidates, token_path_map):
    """
    Given orthographic key and candidate token names (from lexicon), pick best token.
    Returns chosen token (exact token string) or None.
    """
    # gather candidate paths that exist in DB
    rows = []
    for c in candidates:
        # try some lookup variants
        for try_tok in (c, c.lower(), c.upper()):
            path = token_path_map.get(try_tok)
            if path:
                rows.append((c, path))
                break
    if not rows:
        return None

    # score: exclude BAD_SUBSTRINGS preferentially
    good = []
    bad = []
    for tok, path in rows:
        pl = path.lower()
        if any(bs in pl for bs in BAD_SUBSTRINGS):
            bad.append((tok, path))
        else:
            good.append((tok, path))

    pick_from = good if good else bad

    # compute durations
    dur_rows = []
    for tok, path in pick_from:
        d = duration_ms_from_wav(path)
        dur_rows.append((tok, path, d if d is not None else 10**9))

    # pick smallest duration (tie-breaker: lexicographic token)
    dur_rows.sort(key=lambda x: (x[2], x[0]))
    chosen_tok = dur_rows[0][0]
    return chosen_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to syllable DB (folder with wavs or sanitized links)")
    p.add_argument("--lexicon", required=True, help="Input lexicon JSON (orth -> [token,...])")
    p.add_argument("--out", default="lexicon_canonical.json", help="Output canonical lexicon JSON")
    args = p.parse_args()

    if not os.path.exists(args.db):
        print("DB folder not found:", args.db); return
    if not os.path.exists(args.lexicon):
        print("Lexicon file not found:", args.lexicon); return

    lex = json.load(open(args.lexicon, encoding='utf-8'))
    token_path_map = find_token_path_map(args.db)
    print("Found DB tokens:", len(token_path_map))
    canonical = {}
    skipped = []
    for key, candidates in sorted(lex.items()):
        chosen = choose_best_token(key, candidates, token_path_map)
        if chosen:
            canonical[key] = [chosen]
        else:
            skipped.append(key)

    json.dump(canonical, open(args.out, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
    print("Wrote canonical lexicon:", args.out)
    print("Entries:", len(canonical), "Skipped keys (no tokens found):", len(skipped))
    if skipped:
        print("Skipped examples:", skipped[:50])

if __name__ == "__main__":
    main()