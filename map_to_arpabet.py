#!/usr/bin/env python3
"""
map_to_arpabet.py (updated to remove .mp3 duplicates)

- Copies / converts ARPABET-named audio files into each Sentence folder (no arpabet_db/).
- Writes SentenceX/manifest.csv (overwrites by default).
- Deletes existing per-sentence manifest.csv files by default (use --no-delete-existing to keep).
- Creates/clears a central manifests folder (--manifests-dir, default ./manifests) and copies each per-sentence manifest into it.
- Converts non-WAV sources (e.g. MP3 or mislabeled files) to valid PCM WAV using ffmpeg if available,
  otherwise falls back to soundfile read/write (best-effort). If conversion not possible, will copy raw data and warn.

NEW: Automatically deletes any .mp3 files found in sentence folders prior to processing
to avoid duplicated entries between .mp3 and .wav sources.
"""
import os
import sys
import json
import argparse
import csv
import shutil
import re
import subprocess
from pathlib import Path
import soundfile as sf

# ---------------------------
# Utilities
# ---------------------------
def load_mapping(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list):
            raise ValueError("Mapping JSON expects 'key': [list of tokens] pairs.")
        normalized[k.lower()] = [str(tok).upper() for tok in v]
    return normalized

def parse_label_into_syllables(label):
    pattern = re.compile(r'([A-Za-z]+)([123]?)')
    matches = pattern.findall(label)
    if not matches:
        return None
    parsed = []
    for syl, tone in matches:
        if syl == "":
            continue
        parsed.append((syl, tone if tone != "" else None))
    if len(parsed) == 0:
        return None
    return parsed

def make_arpabet_filename_from_parts(parts, mapping):
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
            tokens = mapping[orth_lc]
            joined = "-".join(tokens)
            concat = f"{joined}{tone_digit}"
            syllable_strings.append(concat)
    final_stem = "_".join(syllable_strings)
    return final_stem, unmapped

def is_valid_wav(path):
    """
    Quick check of RIFF/WAVE header. Returns True if file looks like a WAV file.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(12)
            if len(hdr) < 12:
                return False
            if hdr[0:4] == b'RIFF' and hdr[8:12] == b'WAVE':
                return True
            return False
    except Exception:
        return False

def safe_write_wav_copy(src_wav_path, dst_wav_path, convert=True):
    """
    Copy or convert src -> dst as a valid PCM WAV file.

    Behavior:
      - if src is already a valid RIFF/WAVE file, do shutil.copy2
      - else if convert True: try to convert using ffmpeg
      - if ffmpeg fails or not available, try to load with soundfile and write PCM_16 WAV (best-effort)
      - if all else fails, copy raw file and warn (may be invalid)
    """
    # if identical paths, skip
    try:
        if os.path.abspath(src_wav_path) == os.path.abspath(dst_wav_path):
            return
    except Exception:
        pass

    # If source already looks like a WAV, just copy
    if os.path.isfile(src_wav_path) and is_valid_wav(src_wav_path):
        try:
            shutil.copy2(src_wav_path, dst_wav_path)
            return
        except Exception:
            # fall through to conversion attempt if copy fails
            pass

    if convert:
        # Try converting with ffmpeg (preserve sample rate/channels defaults to 44100/mono)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", src_wav_path,
            "-ar", "44100", "-ac", "1", "-f", "wav", dst_wav_path
        ]
        try:
            proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode == 0 and os.path.isfile(dst_wav_path) and is_valid_wav(dst_wav_path):
                # success
                return
            else:
                print(f"[WARN] ffmpeg conversion failed for '{src_wav_path}' -> '{dst_wav_path}'; rc={proc.returncode}")
        except FileNotFoundError:
            print("[WARN] ffmpeg not found on PATH; attempting python-based fallback (may not support MP3).")
        except Exception as e:
            print(f"[WARN] ffmpeg conversion exception for '{src_wav_path}': {e}")

    # Fallback: try reading/writing with soundfile (works for some formats)
    try:
        data, sr = sf.read(src_wav_path)
        sf.write(dst_wav_path, data, sr, subtype="PCM_16")
        if is_valid_wav(dst_wav_path):
            return
    except Exception as e:
        print(f"[WARN] soundfile fallback failed for '{src_wav_path}': {e}")

    # Last resort: attempt raw copy (keeps data but may be wrong format)
    try:
        shutil.copy2(src_wav_path, dst_wav_path)
        print(f"[WARN] wrote raw copy {dst_wav_path} (file may not be valid WAV)")
    except Exception as e:
        print(f"[ERROR] failed to copy or convert '{src_wav_path}' -> '{dst_wav_path}': {e}")
        raise

def make_unique_path(dst_base):
    """
    If dst_base exists, append _1, _2, ... before extension to get unique file.
    dst_base e.g. /path/R-AA3.wav -> returns unique path
    """
    if not os.path.exists(dst_base):
        return dst_base
    root, ext = os.path.splitext(dst_base)
    i = 1
    while True:
        cand = f"{root}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

# ---------------------------
# Utilities: delete MP3s
# ---------------------------
def delete_mp3_files_in_tree(path):
    """
    Recursively remove .mp3 files under 'path'. Case-insensitive for .mp3.
    Returns number of files removed.
    """
    removed = 0
    if not os.path.exists(path):
        return removed
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".mp3"):
                full = os.path.join(dirpath, fn)
                try:
                    os.remove(full)
                    removed += 1
                except Exception as e:
                    print(f"[WARN] failed to remove mp3 {full}: {e}")
    if removed > 0:
        print(f"[INFO] Removed {removed} .mp3 file(s) under {path}")
    return removed

# ---------------------------
# Parse ARPABET transcription filenames
# ---------------------------
def sanitize_filename_token(tok):
    return re.sub(r'[^A-Za-z0-9]', '', tok).strip()

def parse_arpabet_transcription_filename(fname):
    # Skip ADS/Zone.Identifier
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
                continue
        else:
            cur_tokens.append(sanitize_filename_token(it_clean).upper())
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

# ---------------------------
# Helpers for manifest aggregation
# ---------------------------
def sanitize_manifest_destname(root, dirpath):
    """
    Produce a filesystem-safe filename for a manifest copy based on the directory path.
    Example: root=./syllables, dirpath=./syllables/Sentence 1 -> 'Sentence_1_manifest.csv'
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

# ---------------------------
# Processing functions
# ---------------------------
def process_from_arpabet_transcription_into_sentence(sentence_folder, verbose=True, convert=True):
    arpa_dir = os.path.join(sentence_folder, "Arpabet Transcription")
    if not os.path.isdir(arpa_dir):
        if verbose:
            print(f"[INFO] No 'Arpabet Transcription' in {sentence_folder}")
        return False

    files = sorted(os.listdir(arpa_dir))
    manifest_rows = []
    for fn in files:
        # skip mp3 explicitly (mp3s should have been deleted, but extra guard here)
        if fn.lower().endswith(".mp3"):
            if verbose:
                print(f"[SKIP] skipping mp3 in Arpabet Transcription: {fn}")
            continue
        parsed = parse_arpabet_transcription_filename(fn)
        if not parsed:
            if verbose:
                print(f"[WARN] Could not parse Arpabet transcription filename: {fn} (skipping)")
            continue
        orig_label, syllables = parsed
        if not orig_label:
            orig_label = os.path.splitext(fn)[0]
        arpabet_stem = arpabet_syllables_to_stem(syllables)
        dst_name = f"{arpabet_stem}.wav"
        src = os.path.join(arpa_dir, fn)
        dst = os.path.join(sentence_folder, dst_name)
        dst = make_unique_path(dst)

        try:
            safe_write_wav_copy(src, dst, convert=convert)
        except Exception as e:
            print(f"[WARN] failed to write '{src}' -> '{dst}': {e}")

        parsed_label_parts = parse_label_into_syllables(orig_label)
        parsed_str = ";".join(f"{a}/{b if b else '2'}" for a,b in parsed_label_parts) if parsed_label_parts else ""

        # attempt to find a matching original wav in sentence_folder (only consider .wav)
        matched_orig = None
        lower_orig = orig_label.lower()
        for cand in os.listdir(sentence_folder):
            if not cand.lower().endswith(".wav"):
                continue
            if os.path.splitext(cand)[0].lower().startswith(lower_orig):
                matched_orig = cand
                break
        if not matched_orig:
            possible = f"{orig_label}.wav"
            if os.path.isfile(os.path.join(sentence_folder, possible)):
                matched_orig = possible
        if not matched_orig:
            matched_orig = orig_label if orig_label.lower().endswith(".wav") else f"{orig_label}.wav"

        manifest_rows.append({
            "orig": matched_orig,
            "parsed": parsed_str,
            "arpabet_name": os.path.basename(dst),
            "unmapped": ""
        })
        if verbose:
            print(f"[ARPABET->SENT] {fn} -> {os.path.basename(dst)} (orig: {matched_orig})")

    # write manifest.csv in sentence root (overwrite)
    manifest_path = os.path.join(sentence_folder, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["orig","parsed","arpabet_name","unmapped"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    if verbose:
        print(f"[DONE] Wrote manifest to {manifest_path}")
    return True

def process_from_extracted_labels_into_sentence(sentence_folder, mapping=None, verbose=True, convert=True):
    extracted_dir = os.path.join(sentence_folder, "extracted_labels")
    if not os.path.isdir(extracted_dir):
        if verbose:
            print(f"[INFO] No extracted_labels in {sentence_folder}")
        return False

    # Only consider .wav files here to prevent duplicates from mp3 presence
    wavs = [f for f in os.listdir(extracted_dir) if f.lower().endswith(".wav")]
    if not wavs:
        if verbose:
            print(f"[INFO] No WAVs in extracted_labels for {sentence_folder}")
        return False

    manifest_rows = []
    for wav_name in sorted(wavs):
        src = os.path.join(extracted_dir, wav_name)
        stem = os.path.splitext(wav_name)[0]
        parts = parse_label_into_syllables(stem)
        if parts is None:
            # write UNPARSED copy
            dst_name = f"UNPARSED_{stem}.wav"
            dst = make_unique_path(os.path.join(sentence_folder, dst_name))
            try:
                safe_write_wav_copy(src, dst, convert=convert)
            except Exception as e:
                print(f"[WARN] failed to write UNPARSED '{src}' -> '{dst}': {e}")
            manifest_rows.append({
                "orig": wav_name,
                "parsed": "",
                "arpabet_name": os.path.basename(dst),
                "unmapped": "PARSE_FAILED"
            })
            if verbose:
                print(f"[WARN] could not parse '{wav_name}' -> copied as {os.path.basename(dst)}")
            continue

        arp_stem, unmapped = make_arpabet_filename_from_parts(parts, mapping or {})
        arp_stem_sanitized = re.sub(r'[^A-Z0-9_\-]', '_', arp_stem.upper())
        dst_name = f"{arp_stem_sanitized}.wav"
        dst = make_unique_path(os.path.join(sentence_folder, dst_name))

        try:
            safe_write_wav_copy(src, dst, convert=convert)
        except Exception as e:
            print(f"[WARN] failed to write '{src}' -> '{dst}': {e}")

        manifest_rows.append({
            "orig": wav_name,
            "parsed": ";".join(f"{a}/{b if b else '2'}" for a,b in parts),
            "arpabet_name": os.path.basename(dst),
            "unmapped": ",".join(unmapped) if unmapped else ""
        })
        if verbose:
            if unmapped:
                print(f"[MAPPED w/ UNMAPPED pieces] {wav_name} -> {os.path.basename(dst)}; missing: {unmapped}")
            else:
                print(f"[MAPPED] {wav_name} -> {os.path.basename(dst)}")

    manifest_path = os.path.join(sentence_folder, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["orig","parsed","arpabet_name","unmapped"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    if verbose:
        print(f"[DONE] Wrote manifest to {manifest_path}")
    return True

# ---------------------------
# Main loop
# ---------------------------
def process_one_sentence(sentence_folder, mapping, use_arpabet=False, verbose=True, convert=True):
    # NEW: Remove .mp3 files under this sentence folder before processing to avoid duplicate tokens.
    try:
        delete_mp3_files_in_tree(sentence_folder)
    except Exception as e:
        if verbose:
            print(f"[WARN] failed to delete mp3s under {sentence_folder}: {e}")

    # 1) try Arpabet Transcription if requested
    if use_arpabet:
        used = process_from_arpabet_transcription_into_sentence(sentence_folder, verbose=verbose, convert=convert)
        if used:
            return
    # 2) fallback to extracted_labels mapping
    used2 = process_from_extracted_labels_into_sentence(sentence_folder, mapping=mapping, verbose=verbose, convert=convert)
    if used2:
        return
    # nothing done
    if verbose:
        print(f"[SKIP] No transcription or extracted_labels to process in {sentence_folder}")

def main():
    parser = argparse.ArgumentParser(description="Map labeled syllable audio to ARPABET filenames and write manifest.csv directly into Sentence folders.")
    parser.add_argument("root", help="Root folder containing Sentence folders (e.g. ./syllables)")
    parser.add_argument("--map", help="Path to JSON mapping file (default: arpabet_map.json)", default="arpabet_map.json")
    parser.add_argument("--from", dest="from_idx", type=int, default=None)
    parser.add_argument("--to", dest="to_idx", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--use-arpabet", action="store_true", help="If Arpabet Transcription exists, use it.")
    parser.add_argument("--manifests-dir", default="./manifests", help="Directory to collect per-sentence manifests into (will be cleared/created)")
    parser.add_argument("--no-delete-existing", action="store_true", help="Do NOT delete existing manifest.csv files in Sentence folders before processing")
    parser.add_argument("--no-convert", action="store_true", help="Do NOT attempt conversion; only copy files as-is (danger: invalid formats may remain)")
    args = parser.parse_args()

    root = args.root
    map_path = args.map
    manifests_dir = args.manifests_dir
    delete_existing = not args.no_delete_existing
    convert = not args.no_convert

    if not os.path.isdir(root):
        print("[ERR] Root does not exist:", root)
        sys.exit(1)

    mapping = {}
    if os.path.isfile(map_path):
        try:
            mapping = load_mapping(map_path)
        except Exception as e:
            print(f"[WARN] failed to load mapping JSON {map_path}: {e}. Continuing with empty mapping.")
            mapping = {}
    else:
        print(f"[INFO] Mapping JSON not found at {map_path}. You can create one, but script will still use Arpabet Transcription if present.")

    folders = []
    if args.all:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and entry.lower().startswith("sentence"):
                folders.append(full)
    else:
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

    # Recursively delete existing per-sentence manifest.csv files if requested
    if delete_existing:
        print("[INFO] Recursively deleting existing manifest.csv files under each Sentence folder (if present)...")
        for f in folders:
            for dirpath, dirnames, filenames in os.walk(f):
                if "manifest.csv" in filenames:
                    mf = os.path.join(dirpath, "manifest.csv")
                    try:
                        os.remove(mf)
                        print(f"  -> removed {mf}")
                    except Exception as e:
                        print(f"  [WARN] failed to remove {mf}: {e}")

    # Prepare manifests_dir: clear/create
    try:
        if os.path.exists(manifests_dir):
            shutil.rmtree(manifests_dir)
        os.makedirs(manifests_dir, exist_ok=True)
        print(f"[INFO] Prepared manifests directory: {manifests_dir}")
    except Exception as e:
        print(f"[ERR] failed to prepare manifests directory {manifests_dir}: {e}")
        sys.exit(1)

    # process each sentence; after processing, copy manifest into manifests_dir
    for f in folders:
        print(f"[INFO] Processing {f} ...")
        process_one_sentence(f, mapping, use_arpabet=args.use_arpabet, verbose=True, convert=convert)

        # copy manifest if it exists (only top-level manifest.csv is copied)
        src_manifest = os.path.join(f, "manifest.csv")
        if os.path.isfile(src_manifest):
            dst_name = sanitize_manifest_destname(root, f)
            dst = os.path.join(manifests_dir, dst_name)
            try:
                shutil.copy2(src_manifest, dst)
                print(f"  -> copied manifest {src_manifest} -> {dst}")
            except Exception as e:
                print(f"  [WARN] failed to copy manifest {src_manifest} -> {dst}: {e}")
        else:
            # if no top-level manifest, search for any manifest.csv under the sentence folder and copy the first one found
            found = False
            for dirpath, dirnames, filenames in os.walk(f):
                if "manifest.csv" in filenames:
                    src_manifest = os.path.join(dirpath, "manifest.csv")
                    dst_name = sanitize_manifest_destname(root, f)
                    dst = os.path.join(manifests_dir, dst_name)
                    try:
                        shutil.copy2(src_manifest, dst)
                        print(f"  -> copied nested manifest {src_manifest} -> {dst}")
                    except Exception as e:
                        print(f"  [WARN] failed to copy nested manifest {src_manifest} -> {dst}: {e}")
                    found = True
                    break
            if not found:
                print(f"  -> no manifest generated in {f} (skipping copy)")

    print(f"[DONE] All done. Per-sentence manifests copied to: {manifests_dir}")

if __name__ == "__main__":
    main()