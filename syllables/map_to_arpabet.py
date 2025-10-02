"""
Maps your labeled syllable WAV files (e.g. "Ra3.wav", "Vi3uu2.wav", "mi2.wav")
to ARPABET-style filenames using a user-supplied mapping JSON.

It:
 - walks Sentence folders under the given root (or a range if provided),
 - looks for Sentence X/extracted_labels/*.wav,
 - parses names into orthographic syllable pieces with tone digits, e.g.:
     "Vi3uu2" -> [("Vi","3"), ("uu","2")],
 - maps each orthographic syllable to ARPABET tokens using the mapping JSON (example format below),
 - constructs a new filename where each syllable becomes:
     "<TOK1-TOK2-...><tone>", e.g. ["R","AA"] tone 3 -> "R-AA3"
   and syllables are joined with underscores:
     "R-AA3_UW2.wav"
 - writes/copies the WAV into Sentence X/arpabet_db/
 - produces Sentence X/arpabet_db/manifest.csv listing mappings and warnings

USAGE:
    python map_to_arpabet.py <root_folder> [--map arpabet_map.json] [--from N] [--to M]

Mapping JSON format (example):
{
  "ra": ["R", "AA"],
  "vi": ["V", "IY"],
  "uu": ["UW"],
  ...
}

Notes:
 - The mapping keys are case-insensitive. The script converts orthographic syllables to lowercase for lookup.
 - The script is conservative: if a syllable is not found in the map, the file is prefixed with UNMAPPED_ and the manifest notes missing entries.
 - You should extend the mapping JSON until there are no UNMAPPED entries for your dataset.
"""

import os
import sys
import json
import argparse
import csv
import shutil
import re
from pathlib import Path
import soundfile as sf

# ---------------------------
# Helper / utility functions
# ---------------------------

def load_mapping(map_path):
    """
    Load the mapping JSON from map_path.
    The JSON maps orthographic syllables (lowercase) -> list of ARPABET tokens (uppercase strings).
    Example: "ra": ["R", "AA"]
    """
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize keys to lowercase and tokens to uppercase (defensive)
    normalized = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list):
            raise ValueError("Mapping JSON expects 'key': [list of tokens] pairs.")
        normalized[k.lower()] = [str(tok).upper() for tok in v]
    return normalized

def parse_label_into_syllables(label):
    """
    Parse a filename stem (without .wav) into syllable chunks with optional tone digits.
    The expected pattern is multiple groups of letters possibly followed by a single tone digit [1-3].
    Examples:
      "Vi3uu2" -> [("Vi","3"), ("uu","2")]
      "Ra3"     -> [("Ra","3")]
      "mi2"     -> [("mi","2")]
      "chunk001" -> this will try to match 'chunk' + '' + '001' -> fallback behavior: return None
    Implementation notes:
      - We use a regex finding groups of letters ([A-Za-z]+) followed by optional tone digit ([123]?)
      - After matching all groups we ensure at least one group has letters (otherwise return None)
    """
    # regex finds sequences of letters and an optional single tone digit
    pattern = re.compile(r'([A-Za-z]+)([123]?)')
    matches = pattern.findall(label)
    if not matches:
        return None

    parsed = []
    total_matched_len = 0
    for syl, tone in matches:
        if syl == "":
            continue
        parsed.append((syl, tone if tone != "" else None))
        total_matched_len += len(syl) + (1 if tone else 0)
    # Quick sanity: if nothing matched, return None
    if len(parsed) == 0:
        return None
    return parsed

def make_arpabet_filename_from_parts(parts, mapping):
    """
    Given parsed parts: [ (orth_syllable, tone_digit_or_None), ... ]
    and mapping: dict orth -> [ARPABET tokens], produce an ARPABET filename stem and
    a list of unmapped syllables (if any).

    Rules for filename generation:
      - for each syllable:
         tokens = mapping[orth.lower()]  (list of token strings)
         syll_str = '-'.join(tokens) + (tone_digit or '2')   # default tone 2 if missing
      - full name = '_'.join(syll_str_for_each)
      - We return the constructed filename stem (safe ASCII) and any unmapped list.

    Example:
      parts = [("Vi","3"), ("uu","2")]
      mapping["vi"] = ["V","IY"], mapping["uu"] = ["UW"]
      -> syll1 = "V-IY3", syll2 = "UW2" -> filename stem: "V-IY3_UW2"
    """
    unmapped = []
    syllable_strings = []
    for orth, tone in parts:
        orth_lc = orth.lower()
        tone_digit = tone if (tone and tone in ["1","2","3"]) else "2"  
        if orth_lc not in mapping:
            unmapped.append(orth)
            safe_syl = f"{orth_lc.upper()}{tone_digit}"
            syllable_strings.append(safe_syl)
        else:
            tokens = mapping[orth_lc]  # list of ARPABET tokens e.g. ["R","AA"]
            # join tokens within syllable with hyphen to show multi-token syllables
            joined = "-".join(tokens)
            # append tone digit to the end (attached to the entire syllable)
            concat = f"{joined}{tone_digit}"
            syllable_strings.append(concat)
    # join syllables with underscore for final filename stem
    final_stem = "_".join(syllable_strings)
    return final_stem, unmapped

def safe_write_wav_copy(src_wav_path, dst_wav_path):
    """
    Write (copy or re-write) the WAV file from src to dst.
    We attempt to copy first (fast). If that fails for any reason, we re-load and re-write using soundfile.
    """
    try:
        shutil.copy2(src_wav_path, dst_wav_path)
    except Exception:
        data, sr = sf.read(src_wav_path)
        sf.write(dst_wav_path, data, sr, subtype="PCM_16")


def process_one_sentence(sentence_folder, mapping, out_base="arpabet_db", verbose=True):
    """
    Process a single "Sentence X" folder.
    - looks for SentenceFolder/extracted_labels/*.wav
    - builds target folder SentenceFolder/<out_base>/
    - for each wav: parse stem -> map -> create arpabet filename -> copy to target
    - produce manifest.csv in the target folder
    """
    extracted_dir = os.path.join(sentence_folder, "extracted_labels")
    if not os.path.isdir(extracted_dir):
        if verbose:
            print(f"[WARN] missing extracted_labels in {sentence_folder} -> skipping")
        return

    # prepare output folder
    out_dir = os.path.join(sentence_folder, out_base)
    os.makedirs(out_dir, exist_ok=True)

    # collect WAV files (non-recursive) in extracted_labels
    wavs = [f for f in os.listdir(extracted_dir) if f.lower().endswith(".wav")]
    if len(wavs) == 0:
        if verbose:
            print(f"[WARN] no WAVs found in {extracted_dir} -> skipping")
        return

    manifest_rows = []
    for wav_name in sorted(wavs):
        src = os.path.join(extracted_dir, wav_name)
        stem = os.path.splitext(wav_name)[0]  # remove .wav
        # parse into parts: list of (orth, tone_or_None)
        parts = parse_label_into_syllables(stem)
        if parts is None:
            # cannot parse: preserve original name with UNPARSED_ prefix
            dst_name = f"UNPARSED_{stem}.wav"
            dst = os.path.join(out_dir, dst_name)
            safe_write_wav_copy(src, dst)
            manifest_rows.append({
                "orig": wav_name,
                "parsed": "",
                "arpabet_name": dst_name,
                "unmapped": "PARSE_FAILED"
            })
            if verbose:
                print(f"[WARN] could not parse '{wav_name}' -> copied as {dst_name}")
            continue

        # map to arpabet filename
        arp_stem, unmapped = make_arpabet_filename_from_parts(parts, mapping)
        # sanitize final filename: only keep A-Z,0-9, hyphen, underscore
        arp_stem_sanitized = re.sub(r'[^A-Z0-9_\-]', '_', arp_stem.upper())
        dst_name = arp_stem_sanitized + ".wav"
        dst = os.path.join(out_dir, dst_name)

        # copy the wav (or re-write)
        try:
            safe_write_wav_copy(src, dst)
            status = "OK"
        except Exception as e:
            dst_name = f"ERROR_{stem}.wav"
            dst = os.path.join(out_dir, dst_name)
            try:
                safe_write_wav_copy(src, dst)
                status = f"WRITTEN_WITH_FALLBACK"
            except Exception as e2:
                status = f"FAILED:{e2}"

        manifest_rows.append({
            "orig": wav_name,
            "parsed": ";".join(f"{a}/{b if b else '2'}" for a,b in parts),
            "arpabet_name": dst_name,
            "unmapped": ",".join(unmapped) if unmapped else ""
        })
        if verbose:
            if unmapped:
                print(f"[MAPPED w/ UNMAPPED pieces] {wav_name} -> {dst_name}; missing: {unmapped}")
            else:
                print(f"[MAPPED] {wav_name} -> {dst_name}")

    # write manifest.csv
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline='', encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["orig","parsed","arpabet_name","unmapped"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    if verbose:
        print(f"[DONE] Sentence folder processed. ARPABET copies in: {out_dir}")
        print(f"       Manifest: {manifest_path}")

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Map labeled syllable WAVs to ARPABET filenames using a mapping JSON.")
    parser.add_argument("root", help="Root folder containing Sentence folders (e.g. C:\\...\\syllables)")
    parser.add_argument("--map", help="Path to JSON mapping file (default: arpabet_map.json)", default="arpabet_map.json")
    parser.add_argument("--from", dest="from_idx", type=int, default=None, help="First Sentence index to process (e.g. 1)")
    parser.add_argument("--to", dest="to_idx", type=int, default=None, help="Last Sentence index to process (e.g. 15)")
    parser.add_argument("--all", action="store_true", help="Process all Sentence * folders found under root (overrides --from/--to)")
    args = parser.parse_args()

    root = args.root
    map_path = args.map
    if not os.path.isdir(root):
        print("Root does not exist:", root)
        sys.exit(1)
    if not os.path.isfile(map_path):
        print("Mapping JSON not found:", map_path)
        print("Create a mapping JSON as described in the script comments and try again.")
        sys.exit(1)

    mapping = load_mapping(map_path)

    # enumerate target Sentence folders
    folders = []
    if args.all:
        # find every folder under root that begins with "Sentence "
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and entry.lower().startswith("sentence"):
                folders.append(full)
    else:
        # use range if provided, otherwise default 1..15 (comfortable default)
        from_idx = args.from_idx if args.from_idx is not None else 1
        to_idx = args.to_idx if args.to_idx is not None else 15
        for i in range(from_idx, to_idx + 1):
            candidate = os.path.join(root, f"Sentence {i}")
            if os.path.isdir(candidate):
                folders.append(candidate)
            else:
                print(f"[WARN] folder missing: {candidate} (skipping)")

    if len(folders) == 0:
        print("[ERR] No Sentence folders to process. Exiting.")
        sys.exit(1)

    for f in folders:
        process_one_sentence(f, mapping)

if __name__ == "__main__":
    main()