#!/usr/bin/env bash
# prepare_and_run_gui.sh
# Create sanitized symlink DB, build/filter lexicons, then run tonal_tts_full.py --gui
set -euo pipefail
cd "$(dirname "$0")"

DB_LINKS=./syllables_all_links
MANIFEST_LEX=lexicon_from_manifests.json
FILTERED_LEX=lexicon_filtered_all.json
FILENAME_LEX=lexicon_from_filenames.json

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

# 2) Generate manifest-derived lexicon if manifest script present (preferred)
if [ -f manifest_to_lexicon_from_manifests.py ]; then
  echo "--> Generating manifest lexicon (manifest_to_lexicon_from_manifests.py)..."
  python3 manifest_to_lexicon_from_manifests.py --root ./syllables --out "$MANIFEST_LEX"
elif [ -f manifest_to_lexicon.py ]; then
  echo "--> Generating manifest lexicon (manifest_to_lexicon.py)..."
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
import json, os
src="$MANIFEST_LEX"
out="$FILTERED_LEX"
dbdir="$DB_LINKS"
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

# 5) Summary
echo "=== Summary ==="
echo "DB folder: $DB_LINKS ($num_files files)"
if [ -f "$FILTERED_LEX" ]; then
  echo "Filtered manifest lexicon: $FILTERED_LEX (used by GUI command)"
else
  echo "Filtered manifest lexicon not found; falling back to filename lexicon: $FILENAME_LEX"
  FILTERED_LEX="$FILENAME_LEX"
fi
echo "Launching GUI with: python3 tonal_tts_full.py --db $DB_LINKS --lexicon $FILTERED_LEX --gui"
echo "If the GUI fails to start, ensure you have tkinter installed (sudo apt install python3-tk) or run the CLI mode."

# 6) Launch the GUI (will block until GUI closed)
python3 tonal_tts_full.py --db "$DB_LINKS" --lexicon "$FILTERED_LEX" --gui

echo "=== Done ==="