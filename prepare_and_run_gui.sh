#!/usr/bin/env bash
# prepare_and_run_gui.sh - simplified canonical pipeline (map -> clean manifests -> refresh DB -> lexicon -> GUI)
set -euo pipefail
cd "$(dirname "$0")"

DB_LINKS="./syllables_all_links"
MANIFESTS_DIR="./manifests"
MANIFEST_LEX="lexicon_from_manifests.json"
FILTERED_LEX="lexicon_filtered_all.json"
CANONICAL_LEX="lexicon_canonical.json"

echo "=== prepare_and_run_gui.sh (canonical pipeline) ==="

# 0) checks
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found."
  exit 1
fi

# 1) Create sanitized symlink DB (initial pass: existing syllable wavs)
echo "--> Creating sanitized symlinks in $DB_LINKS ..."
rm -rf "$DB_LINKS"
mkdir -p "$DB_LINKS"
shopt -s nullglob
for f in ./syllables/*/*.[wW][aA][vV]; do
  [ -f "$f" ] || continue
  bn=$(basename "$f")
  safe=$(echo "$bn" | tr ' ' '_' | tr -d '(),')
  if [ -e "$DB_LINKS/$safe" ]; then
    i=1; base="${safe%.*}"; ext="${safe##*.}"
    while [ -e "$DB_LINKS/${base}_$i.$ext" ]; do i=$((i+1)); done
    safe="${base}_$i.$ext"
  fi
  ln -s "$(realpath "$f")" "$DB_LINKS/$safe"
done
num_files=$(ls -1 "$DB_LINKS" 2>/dev/null | wc -l || echo 0)
echo "  -> Created symlinks: $num_files files"

# 2) Generate per-sentence ARPABET WAVs / conversion (map -> arpabet)
echo "--> Generating/converting ARPABET WAVs via map_to_arpabet.py (in-place) ..."
if [ -f map_to_arpabet.py ]; then
  python3 map_to_arpabet.py ./syllables --all --use-arpabet
else
  echo "  !! map_to_arpabet.py not found; ensure Arpabet Transcription exists or run conversion separately."
fi

# 3) Build clean per-sentence manifests from Arpabet Transcription
#    This writes manifest.csv into each sentence folder and copies cleaned manifests into MANIFESTS_DIR.
echo "--> Building cleaned per-sentence manifests via build_clean_manifests.py -> $MANIFESTS_DIR ..."
rm -rf "$MANIFESTS_DIR"
mkdir -p "$MANIFESTS_DIR"
if [ -f build_clean_manifests.py ]; then
  python3 build_clean_manifests.py ./syllables --out-dir "$MANIFESTS_DIR" --verbose || {
    echo "  !! build_clean_manifests.py failed; falling back to manual aggregation of existing manifests."
  }
else
  echo "  !! build_clean_manifests.py not found; will attempt to aggregate existing manifests."
fi

# Fallback: if MANIFESTS_DIR is empty, aggregate any per-sentence manifest.csv files
if [ -z "$(ls -A "$MANIFESTS_DIR" 2>/dev/null || true)" ]; then
  echo "--> Fallback: aggregating existing per-sentence manifest.csv files into $MANIFESTS_DIR ..."
  for d in ./syllables/*; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    if [[ "${base,,}" != sentence* ]]; then continue; fi
    src_manifest="$d/manifest.csv"
    if [ -f "$src_manifest" ]; then
      cp "$src_manifest" "$MANIFESTS_DIR/${base}_manifest.csv"
      echo "  -> copied $src_manifest"
    fi
  done
fi
echo "--> Manifests are now in: $MANIFESTS_DIR"

# 3.5) REFRESH symlink DB to include any newly-created ARPABET WAVs in sentence folders
echo "--> Refreshing symlink DB to include ARPABET files (recreating $DB_LINKS) ..."
rm -rf "$DB_LINKS"
mkdir -p "$DB_LINKS"
shopt -s nullglob
# include any .wav in top-level sentence folders (includes ARPABET wavs created in-place)
for f in ./syllables/*/*.wav; do
  [ -f "$f" ] || continue
  bn=$(basename "$f")
  safe=$(echo "$bn" | tr ' ' '_' | tr -d '(),')
  if [ -e "$DB_LINKS/$safe" ]; then
    i=1; base="${safe%.*}"; ext="${safe##*.}"
    while [ -e "$DB_LINKS/${base}_$i.$ext" ]; do i=$((i+1)); done
    safe="${base}_$i.$ext"
  fi
  ln -s "$(realpath "$f")" "$DB_LINKS/$safe"
done
num_files=$(ls -1 "$DB_LINKS" 2>/dev/null | wc -l || echo 0)
echo "  -> Refreshed symlinks: $num_files files"

# 4) Build lexicon from manifests (preferred)
echo "--> Building manifest-derived lexicon -> $MANIFEST_LEX ..."
if [ -f manifest_to_lexicon_from_manifests.py ]; then
  python3 manifest_to_lexicon_from_manifests.py --root "$MANIFESTS_DIR" --out "$MANIFEST_LEX"
else
  echo "  !! manifest_to_lexicon_from_manifests.py not found; aborting."
  exit 1
fi

# 5) Filter manifest lexicon to tokens present in DB_LINKS -> FILTERED_LEX
echo "--> Filtering manifest lexicon to DB tokens -> $FILTERED_LEX ..."
python3 - <<'PY'
import json, os, re
src = "lexicon_from_manifests.json"
out = "lexicon_filtered_all.json"
dbdir = "syllables_all_links"
have = {os.path.splitext(f)[0] for f in os.listdir(dbdir) if f.lower().endswith('.wav') or os.path.islink(os.path.join(dbdir,f))}
lex = json.load(open(src, encoding='utf-8'))
filtered = {}
for k,v in lex.items():
    kept=[]
    for tok in v:
        tok_norm = re.sub(r'_[0-9]+$','', tok)
        if tok in have:
            kept.append(tok); continue
        if tok_norm in have:
            kept.append(tok_norm); continue
        low = tok.lower()
        low_norm = tok_norm.lower()
        # case-insensitive map
        for h in have:
            if h.lower()==low or h.lower()==low_norm:
                kept.append(h); break
    if kept:
        seen = list(dict.fromkeys(kept))
        filtered[k]=sorted(seen)
json.dump(filtered, open(out,"w",encoding='utf-8'), indent=2, ensure_ascii=False)
print("WROTE", out, "entries:", len(filtered))
PY

# 6) Build canonical lexicon (one preferred token per key)
echo "--> Building canonical lexicon -> $CANONICAL_LEX ..."
if [ -f build_canonical_lexicon.py ]; then
  python3 build_canonical_lexicon.py --db "$DB_LINKS" --lexicon "$FILTERED_LEX" --out "$CANONICAL_LEX"
  if [ -f "$CANONICAL_LEX" ]; then
    echo "  -> canonical lexicon created: $CANONICAL_LEX"
    FILTERED_LEX="$CANONICAL_LEX"
  else
    echo "  -> canonical build failed; using $FILTERED_LEX"
  fi
else
  echo "  -> build_canonical_lexicon.py not found; keeping $FILTERED_LEX"
fi

# 7) Launch GUI with canonical/filtered lexicon
echo "Launching GUI: python3 tonal_tts_full.py --db \"$DB_LINKS\" --lexicon \"$FILTERED_LEX\" --gui"
python3 tonal_tts_full.py --db "$DB_LINKS" --lexicon "$FILTERED_LEX" --gui
