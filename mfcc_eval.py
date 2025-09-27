"""
mfcc_eval.py - improved MFCC+DTW evaluator for combined vs split tokens.

Usage:
  # Basic (point at a flat folder or folder with subfolders)
  python mfcc_eval.py --db-root ./syllables_all_links --pairs pairs.json --out mfcc_distances.csv

  # If some 'combined' tokens are missing but split parts exist, auto-create combined WAVs
  python mfcc_eval.py --db-root ./syllables_all_links --pairs pairs.json --out mfcc_distances.csv --auto-combine

Notes:
 - The script scans the --db-root recursively and builds a token -> wav-path index.
 - Token lookup first tries exact match, then case-insensitive fallback.
 - Requires: librosa, scipy, numpy, pydub (for optional auto-combine).
   Install: pip install librosa scipy numpy pydub
"""

import os
import argparse
import json
import csv
import sys
from typing import Dict, Optional, List, Tuple

import numpy as np

# try imports and report helpful message if missing
try:
    import librosa
    from scipy.spatial.distance import cdist
    from librosa.sequence import dtw
except Exception as e:
    raise RuntimeError("mfcc_eval requires librosa and scipy. Install with: pip install librosa scipy") from e

# pydub is optional but used when auto-combining split tokens into a synthetic combined WAV
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None  # we'll check at runtime if auto-combine requested


# ------------------------
# Helpers: file index
# ------------------------
def build_token_index(db_root: str) -> Tuple[Dict[str,str], Dict[str,str]]:
    """
    Walk db_root recursively and build:
      - exact_index: token_name -> full_path (token_name = filename without .wav)
      - lower_index: token_name.lower() -> token_name (first seen) for case-insensitive fallback
    We keep the first-seen path for each token name.
    """
    exact_index: Dict[str, str] = {}
    lower_index: Dict[str, str] = {}
    for root, _, files in os.walk(db_root):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            token = os.path.splitext(f)[0]
            full = os.path.join(root, f)
            if token not in exact_index:
                exact_index[token] = full
            lower = token.lower()
            if lower not in lower_index:
                lower_index[lower] = token  # map lowercase -> canonical token name
    return exact_index, lower_index


def find_token_path(token: str, exact_index: Dict[str,str], lower_index: Dict[str,str]) -> Optional[str]:
    """
    Return full path for token:
      1) exact match (case-sensitive)
      2) case-insensitive fallback (if token.lower() matches a known token)
    Returns None if not found.
    """
    if token in exact_index:
        return exact_index[token]
    low = token.lower()
    if low in lower_index:
        canonical = lower_index[low]
        return exact_index.get(canonical)
    return None


# ------------------------
# MFCC / DTW computations
# ------------------------
def load_mfcc(path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Load a WAV and return MFCC matrix with shape (n_mfcc, frames).
    librosa.load -> waveform, then librosa.feature.mfcc.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def dtw_cost(m1: np.ndarray, m2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute DTW on cosine distances between frame-vectors and return:
      - total accumulated cost (sum of matched D values along the warping path)
      - warping path as an array of shape (N,2) with pairs (i_index_in_m1_frames, j_index_in_m2_frames)
    m1, m2 shapes: (n_mfcc, frames)
    """
    # compute frame-to-frame distance matrix D (frames1 x frames2) using cosine distance
    D = cdist(m1.T, m2.T, metric='cosine')
    # librosa.sequence.dtw accepts a cost matrix and returns (acc_cost_matrix, wp)
    acc_cost, wp = dtw(C=D)
    # wp is a list/array of index pairs (i, j) in reversed order (depending on librosa version).
    wp = np.asarray(wp)  # shape (N,2)
    # compute the sum of D over the warping path
    cost = float(D[wp[:, 0], wp[:, 1]].sum())
    return cost, wp


def mean_mfcc_aligned(m1: np.ndarray, m2: np.ndarray, wp: np.ndarray) -> float:
    """
    Compute the mean absolute difference between matched MFCC frames using the DTW path.
    This gives a frame-aligned average MFCC magnitude difference.
    """
    if wp is None or wp.size == 0:
        return float('nan')
    # m1[:, i] and m2[:, j] for each (i,j) in wp
    diffs = []
    for (i, j) in wp:
        # guard: ensure indices in-range (they should be)
        if i < m1.shape[1] and j < m2.shape[1]:
            diffs.append(np.mean(np.abs(m1[:, i] - m2[:, j])))
    if not diffs:
        return float('nan')
    return float(np.mean(diffs))


# ------------------------
# Optional: auto-combine splits into a combined WAV
# ------------------------
def auto_combine_save(split_paths: List[str], out_path: str, crossfade_ms: int = 8) -> bool:
    """
    Concatenate multiple WAV files (split_paths) using pydub with a small crossfade
    and save to out_path. Returns True on success.
    Requires pydub (AudioSegment).
    """
    if AudioSegment is None:
        raise RuntimeError("Auto-combine requires pydub. Install: pip install pydub")
    if not split_paths:
        return False
    seg = None
    for p in split_paths:
        s = AudioSegment.from_file(p)
        if seg is None:
            seg = s
        else:
            # append with short crossfade where possible
            cf = min(crossfade_ms, len(seg), len(s))
            seg = seg.append(s, crossfade=cf) if cf > 0 else seg + s
    # ensure parent directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    seg.export(out_path, format="wav")
    return True


# ------------------------
# Main runner
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="MFCC+DTW evaluator for combined vs split tokens.")
    parser.add_argument("--db-root", required=True,
                        help="Root folder containing .wav tokens (will be scanned recursively).")
    parser.add_argument("--pairs", required=True, help="JSON file listing pairs (combined + split list).")
    parser.add_argument("--out", default="mfcc_distances.csv", help="CSV output path.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for librosa.load (default 22050).")
    parser.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients.")
    parser.add_argument("--auto-combine", action="store_true",
                        help="If combined WAV missing but split parts exist, auto-create combined WAV by concatenation.")
    parser.add_argument("--verbose", action="store_true", help="Print extra info while running.")
    args = parser.parse_args()

    # build file index once
    exact_index, lower_index = build_token_index(args.db_root)
    if args.verbose:
        print(f"Indexed {len(exact_index)} WAV tokens under {args.db_root}")

    # load pairs
    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    rows = []
    for pair in pairs:
        combined = pair.get("combined")
        split = pair.get("split", [])

        # find combined path (exact or case-insensitive)
        combined_path = find_token_path(combined, exact_index, lower_index)
        split_paths = [find_token_path(s, exact_index, lower_index) for s in split]

        # If combined is missing but auto-combine requested and splits are present -> create combined
        auto_generated = False
        if combined_path is None and args.auto_combine:
            if all(p is not None for p in split_paths):
                # choose a combined path location inside db_root (flat), using combined token name
                # if combined contains directories (unlikely), just put it in db_root root.
                out_combined = os.path.join(args.db_root, combined + ".wav")
                if args.verbose:
                    print(f"Auto-combining {split} -> {out_combined}")
                try:
                    auto_combine_save(split_paths, out_combined)
                    auto_generated = True
                    # update the index to include the newly created file
                    exact_index[combined] = out_combined
                    lower_index[combined.lower()] = combined
                    combined_path = out_combined
                except Exception as e:
                    print(f"Auto-combine failed for {combined}: {e}")
            else:
                if args.verbose:
                    missing = [s for s, p in zip(split, split_paths) if p is None]
                    print(f"Cannot auto-combine {combined}: missing split parts {missing}")

        # If still missing either combined or any split -> skip and report
        if combined_path is None or not all(p for p in split_paths):
            print("Skipping pair due to missing files:", {"combined": combined, "split": split})
            # add a CSV row indicating skip (optional) - here we skip writing a numeric evaluation
            rows.append({
                "combined": combined,
                "split": ",".join(split),
                "dtw_cost": "",
                "mean_mfcc_dist": "",
                "combined_path": combined_path or "",
                "split_paths": ";".join(p or "" for p in split_paths),
                "note": "skipped_missing_files"
            })
            continue

        # load MFCCs for combined and stitched split
        try:
            m1 = load_mfcc(combined_path, sr=args.sr, n_mfcc=args.n_mfcc)  # (n_mfcc, frames_c)
        except Exception as e:
            print(f"Failed to load combined MFCC for {combined}: {e}")
            continue

        try:
            m2_parts = [load_mfcc(p, sr=args.sr, n_mfcc=args.n_mfcc) for p in split_paths]
            # horizontally stack frames: result (n_mfcc, frames_summed)
            m2 = np.hstack(m2_parts) if m2_parts else np.empty((args.n_mfcc, 0))
        except Exception as e:
            print(f"Failed to load split MFCCs for {combined} splits {split}: {e}")
            continue

        # compute DTW cost and warping path
        try:
            cost, wp = dtw_cost(m1, m2)
            mean_dist = mean_mfcc_aligned(m1, m2, wp)
        except Exception as e:
            print(f"DTW failed for {combined}: {e}")
            rows.append({
                "combined": combined,
                "split": ",".join(split),
                "dtw_cost": "",
                "mean_mfcc_dist": "",
                "combined_path": combined_path,
                "split_paths": ";".join(split_paths),
                "note": f"dtw_error:{e}"
            })
            continue

        rows.append({
            "combined": combined,
            "split": ",".join(split),
            "dtw_cost": cost,
            "mean_mfcc_dist": mean_dist,
            "combined_path": combined_path,
            "split_paths": ";".join(split_paths),
            "note": "auto_generated" if auto_generated else ""
        })
        if args.verbose:
            print(f"OK: {combined} | dtw_cost={cost:.3f} | mean_mfcc_dist={mean_dist:.4f}")

    # write CSV with extra columns for debugging
    keys = ["combined","split","dtw_cost","mean_mfcc_dist","combined_path","split_paths","note"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved evaluation to", args.out)


if __name__ == "__main__":
    main()