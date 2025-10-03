#!/usr/bin/env bash
# prepare_and_run_gui.sh
# Create sanitized symlink DB, generate per-sentence ARPABET manifests (in-place),
# collect all Sentence manifests into ./manifests/, build/filter lexicons, then run tonal_tts_full.py --gui

set -euo pipefail
cd "$(dirname "$0")"

DB_LINKS=./syllables_all_links
MANIFEST_LEX=lexicon_from_manifests.json
FILTERED_LEX=lexicon_filtered_all.json
FILENAME_LEX=lexicon_from_filenames.json
CANONICAL_LEX=lexicon_canonical.json
MANIFESTS_DIR=./manifests

echo "=== prepare_and_run_gui.sh - starting ==="

# 0) Basic checks
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3 and try again."
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Warning: no virtualenv detected (VIRTUAL_ENV unset)."
  echo "If you want to use your project's venv, activate it first (e.g. 'source venv/bin/activate')"
fi

# 1) Create sanitized symlink folder for all syllable wavs
echo "--> Creating sanitized symlinks in $DB_LINKS ..."
rm -rf "$DB_LINKS"
mkdir -p "$DB_LINKS"

shopt -s nullglob
for f in ./syllables/*/*.[wW][aA][vV]; do
  [ -f "$f" ] || continue
  bn=$(basename "$f")
  # sanitize: replace spaces with _, remove parentheses and commas
  safe=$(echo "$bn" | tr ' ' '_' | tr -d '(),')
  # avoid collisions
  if [ -e "$DB_LINKS/$safe" ]; then
    i=1
    base="${safe%.*}"
    ext="${safe##*.}"
    while [ -e "$DB_LINKS/${base}_$i.$ext" ]; do i=$((i+1)); done
    safe="${base}_$i.$ext"
  fi
  ln -s "$(realpath "$f")" "$DB_LINKS/$safe"
done

num_files=$(ls -1 "$DB_LINKS" 2>/dev/null | wc -l || echo 0)
echo "  -> Created symlinks: $num_files files"

# --- NEW: generate in-place arpabet manifests inside each Sentence folder ---
echo "--> Generating per-Sentence ARPABET manifests using map_to_arpabet.py (in-place)..."
if [ -f map_to_arpabet.py ]; then
  # Use --all to process all Sentence* folders and prefer Arpabet Transcription when present.
  python3 map_to_arpabet.py ./syllables --all --use-arpabet || {
    echo "map_to_arpabet.py failed; aborting."
    exit 1
  }
else
  echo "--> map_to_arpabet.py not found; skipping ARPABET manifest generation."
fi
# --- end new ---

# --- NEW: collect all per-sentence manifest.csv into a single manifests/ folder ---
echo "--> Aggregating per-Sentence manifest.csv files into $MANIFESTS_DIR ..."
rm -rf "$MANIFESTS_DIR"
mkdir -p "$MANIFESTS_DIR"

# find Sentence folders and copy manifest.csv if present
shopt -s nullglob
for d in ./syllables/*; do
  [ -d "$d" ] || continue
  base=$(basename "$d")
  # only process folders that start with "Sentence" (case-insensitive)
  if [[ "${base,,}" != sentence* ]]; then
    continue
  fi
  src_manifest="$d/manifest.csv"
  if [ -f "$src_manifest" ]; then
    # sanitize folder name to use as prefix (replace spaces with underscores)
    prefix=$(echo "$base" | tr ' ' '_')
    dst="$MANIFESTS_DIR/${prefix}_manifest.csv"
    cp "$src_manifest" "$dst"
    echo "  -> copied $src_manifest -> $dst"
  else
    echo "  -> no manifest in $d (skipping)"
  fi
done
echo "--> Aggregation complete. Manifests copied to $MANIFESTS_DIR"
# --- end new ---

# 2) Generate manifest-derived lexicon if manifest script present (preferred)
#    Now run manifest_to_lexicon_from_manifests.py using manifests/ as the root
if [ -f manifest_to_lexicon_from_manifests.py ]; then
  echo "--> Generating manifest lexicon (manifest_to_lexicon_from_manifests.py) from $MANIFESTS_DIR ..."
  python3 manifest_to_lexicon_from_manifests.py --root "$MANIFESTS_DIR" --out "$MANIFEST_LEX"
elif [ -f manifest_to_lexicon.py ]; then
  echo "--> Generating manifest lexicon (manifest_to_lexicon.py) - legacy script (may expect different layout)."
  # If the legacy script expects Sentence-root layout, point it at ./syllables (so it finds per-sentence manifests)
  python3 manifest_to_lexicon.py
else
  if [ -f "$MANIFEST_LEX" ]; then
    echo "--> Using existing $MANIFEST_LEX"
  else
    echo "--> No manifest-to-lexicon script found and $MANIFEST_LEX not present. Skipping manifest lexicon generation."
  fi
fi

# 3) Filter the manifest lexicon to tokens that actually exist in the sanitized DB
if [ -f "$MANIFEST_LEX" ]; then
  echo "--> Filtering $MANIFEST_LEX to tokens present in $DB_LINKS -> $FILTERED_LEX"
  python3 - <<PY
import json, os, re
src="$MANIFEST_LEX"
out="$FILTERED_LEX"
dbdir="$DB_LINKS"
# tokens present in DB_LINKS
have={os.path.splitext(f)[0] for f in os.listdir(dbdir) if f.lower().endswith('.wav') or os.path.islink(os.path.join(dbdir,f))}
lex=json.load(open(src))
filtered={}
for k,v in lex.items():
    kept=[t for t in v if t in have]
    if kept:
        filtered[k]=kept
json.dump(filtered, open(out,"w"), indent=2, ensure_ascii=False)
print("Wrote", out, "with", len(filtered), "entries; tokens in DB:", len(have))
PY
else
  echo "--> No manifest lexicon to filter (skipping)."
fi

# 4) Also generate a filename-derived lexicon (fallback)
echo "--> Generating filename-derived lexicon from $DB_LINKS -> $FILENAME_LEX"
python3 - <<PY
import os, json, re
from collections import defaultdict
root="$DB_LINKS"
lex=defaultdict(list)
for fn in sorted(os.listdir(root)):
    if not fn.lower().endswith('.wav'): continue
    token=os.path.splitext(fn)[0]
    m=re.match(r"^(.+?)(\d+)$", token)
    base=(m.group(1).lower() if m else token.lower())
    lex[base].append(token)
json.dump({k:sorted(list(set(v))) for k,v in lex.items()}, open("$FILENAME_LEX","w"), indent=2, ensure_ascii=False)
print("Wrote $FILENAME_LEX with", len(lex), "entries")
PY

# 4.5) Build canonical lexicon (always prefer shortest single-syllable tokens)
echo "--> Building canonical lexicon (one token per key) ..."
if [ -f build_canonical_lexicon.py ]; then
  python3 build_canonical_lexicon.py --db "$DB_LINKS" --lexicon "$FILTERED_LEX" --out "$CANONICAL_LEX"
  if [ -f "$CANONICAL_LEX" ]; then
    echo "  -> Canonical lexicon created: $CANONICAL_LEX"
    FILTERED_LEX="$CANONICAL_LEX"
  else
    echo "  !! Canonical lexicon creation failed, falling back to filtered lexicon."
  fi
else
  echo "  !! build_canonical_lexicon.py not found — skipping canonicalization."
fi

# 5) Summary
echo "=== Summary ==="
echo "DB folder: $DB_LINKS ($num_files files)"
if [ -f "$FILTERED_LEX" ]; then
  echo "Canonical lexicon in use: $FILTERED_LEX (used by GUI command)"
else
  echo "Canonical lexicon not found; falling back to filename lexicon: $FILENAME_LEX"
  FILTERED_LEX="$FILENAME_LEX"
fi
echo "Launching GUI with: python3 tonal_tts_full.py --db $DB_LINKS --lexicon $FILTERED_LEX --gui"
echo "If the GUI fails to start, ensure you have tkinter installed (sudo apt install python3-tk) or run the CLI mode."

# 6) Launch the GUI (will block until GUI closed)
python3 tonal_tts_full.py --db "$DB_LINKS" --lexicon "$FILTERED_LEX" --gui

echo "=== Done ==="