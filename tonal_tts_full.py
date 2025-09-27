#!/usr/bin/env python3
"""
tonal_tts_full.py
Complete concatenative syllable-based tonal TTS reference implementation.

How to run:
  - Demo (CLI):  python tonal_tts_full.py --demo
  - GUI:         python tonal_tts_full.py --db ./syllables --lexicon lexicon.json --gui
  - Play token:  python tonal_tts_full.py --db ./syllables --play "Ra3"

This file is heavily commented so you can explain each line to your professor.
"""

# --- Standard library imports ------------------------------------------------
# We import small, focused modules from the standard library to handle file I/O,
# regular expressions, JSON, threading for background playback, and small helper types.
import os                # for path and filesystem operations
import sys               # for exiting with status codes and script args
import re                # for regular expressions (token & word parsing)
import json              # to load/save lexicons and metadata
import threading         # to play audio without blocking the UI/CLI
import time              # small helper for timing and debug prints
from dataclasses import dataclass   # convenient small classes for data holding
from typing import Optional, List, Callable, Dict  # type hints (helpful for readers)

# --- Third-party imports -----------------------------------------------------
# pydub is used for simple audio segment manipulation and playback. It is not
# super-optimized, but it's easy to understand and good for a teaching project.
from pydub import AudioSegment           # core audio segment object
from pydub.playback import play          # simple synchronous playback
from pydub.generators import Sine        # used for highlighting beeps in GUI

# numpy for numeric arrays (used during optional pitch shifting)
import numpy as np

# librosa is optional but used for higher quality pitch shifting if available.
# We attempt to import it; if it's unavailable we gracefully degrade.
try:
    import librosa
except Exception:
    librosa = None  # mark as unavailable; code will check this before using it.

# --- Regular expressions and small helpers -----------------------------------
# TOKEN_TONE_RE matches a token like "Ra3" -> ("Ra", "3"). We use a non-greedy
# token name then trailing digits for tone. This is general enough for your naming.
TOKEN_TONE_RE = re.compile(r"^(.+?)(\d+)$")

def split_token_and_tone(token: str):
    """
    Split a token like 'Ra3' into (base, tone) e.g. ('Ra', '3').
    If no trailing digits, returns (token, None).
    """
    m = TOKEN_TONE_RE.match(token)        # check if token ends with digits
    if m:
        return m.group(1), m.group(2)    # base token name, tone digits
    return token, None                   # no tone found

# --- Small data container ----------------------------------------------------
# SyllableUnit holds metadata and cached AudioSegment for a single token.
@dataclass
class SyllableUnit:
    token: str                             # the token name, e.g., "Ra3"
    path: str                              # full path to the WAV file
    audio: Optional[AudioSegment] = None   # cached loaded audio (lazy load)
    duration_ms: Optional[int] = None      # duration in ms (set on load)

    def load(self):
        """
        Load the audio from disk if not already loaded. Using lazy-loading
        avoids reading all WAV files at startup (helpful for large DBs).
        """
        if self.audio is None:
            self.audio = AudioSegment.from_file(self.path)  # read file into memory
            self.duration_ms = len(self.audio)             # audio length in ms

# --- Syllable database (collection of tokens) --------------------------------
class SyllableDB:
    """
    Simple object representing a folder of WAV tokens. It loads a list of
    .wav files into SyllableUnit objects and supports simple aliasing via
    a meta.json file placed in the DB folder.
    """
    def __init__(self, folder_path: str):
        self.folder_path = folder_path    # store path for later uses
        self.units: Dict[str, SyllableUnit] = {}  # token_name -> SyllableUnit
        self.meta = {}                    # optional meta.json contents
        self._load()                      # perform initial loading

    def _load(self):
        """
        Iterate over the folder and create SyllableUnit entries for each WAV.
        Also parse meta.json for optional aliases mapping (alias -> canonical).
        """
        meta_path = os.path.join(self.folder_path, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)   # load alias metadata if present
            except Exception:
                print("Warning: could not parse meta.json; ignoring.")  # non-fatal

        # iterate over files in folder and collect .wav files
        for fname in sorted(os.listdir(self.folder_path)):
            if not fname.lower().endswith(".wav"):
                continue                     # skip non-wav files
            token = os.path.splitext(fname)[0]  # token is filename without extension
            full = os.path.join(self.folder_path, fname)
            self.units[token] = SyllableUnit(token=token, path=full)  # create unit

        # apply aliases (if meta.json contained {"aliases": {"aliasName":"Canonical"}})
        aliases = self.meta.get("aliases", {})
        for alias, canonical in aliases.items():
            if canonical in self.units:
                self.units[alias] = self.units[canonical]  # alias points to same SyllableUnit

    def get(self, token: str) -> Optional[SyllableUnit]:
        """Return the SyllableUnit for token, or None if not found."""
        return self.units.get(token)

    def available_tokens(self) -> List[str]:
        """Return a list of token names available in the DB."""
        return list(self.units.keys())

# --- Text -> syllable/token mapping ------------------------------------------
class TextToSyllableMapper:
    """
    Maps orthographic words (plain text) or token sequences to a list of tokens.
    It uses:
      - an optional lexicon (word -> token list) if provided,
      - a set of diphthongs/triphthongs (multi-character vowel combos),
      - a list of onomatopoeia (words that should be treated as single tokens),
      - fallback matching based on available tokens in the DB (longest-first).
    The mapping is intentionally greedy (longest match first) so multi-character
    combinations are prioritized.
    """
    def __init__(self, db: SyllableDB, lexicon_path: Optional[str]=None):
        self.db = db                                # database to check token existence
        self.lexicon = {}                           # loaded lexicon mapping (optional)
        if lexicon_path and os.path.exists(lexicon_path):
            try:
                with open(lexicon_path, "r", encoding="utf-8") as f:
                    self.lexicon = json.load(f)     # load lexicon if present
            except Exception:
                print("Warning: could not parse lexicon.json; ignoring.")

        # A non-exhaustive list of diphthongs that we want to consider as atomic units.
        # You can expand these to match the phonology of your language.
        self.diphthongs = [
            "ai", "au", "ei", "oi", "ou", "ia", "ie", "io", "ua", "ue", "uo", "ei", "eu"
        ]

        # A short list of common triphthongs (expand for your language as needed).
        self.triphthongs = ["iau", "uai", "eau", "iou"]

        # Onomatopoeia are words like "boom", "bang", "meow" that often should
        # be treated as single units or mapped to special tokens. Keep this list
        # in the lexicon for better control, but we also include a small builtin
        # set as reasonable defaults for demonstration.
        self.onomatopoeia = set([
            "boom", "bang", "crash", "meow", "woof", "beep", "ding", "buzz", "hiss", "clack"
        ])

        # Build the candidate token list used for greedy matching:
        # 1) tokens present in DB (these are actual WAV token names like 'Ra3')
        # 2) lexicon keys (words mapped to sequences)
        tokens_from_db = sorted(db.available_tokens(), key=len, reverse=True)
        lex_keys = sorted(self.lexicon.keys(), key=len, reverse=True)

        # Combine all interesting items and sort longest-first for greedy matching.
        # We include diphthongs/triphthongs explicitly so they can be matched even
        # if they don't exist as tokens in the DB.
        combined = list(tokens_from_db) + lex_keys + self.diphthongs + self.triphthongs + list(self.onomatopoeia)
        self.sorted_tokens = sorted(set(combined), key=len, reverse=True)

    def map_word(self, word: str):
        """
        Map a single orthographic word to a list of tokens.
        The algorithm:
          - if the whole word is in the lexicon, return the lexicon mapping
          - otherwise perform longest-first greedy matching using sorted_tokens
            so multi-character sequences (diphthongs/triphthongs/onomatopoeia)
            are chosen before single letters
          - if nothing matches a prefix, fall back to consuming one character
            (this is a safety net so the mapping never loops forever)
        """
        word = word.strip().lower()  # normalize to lowercase (lexicon should be lowercase too)
        if not word:
            return []                  # empty word -> empty token list

        # if the whole word is in the lexicon, return that mapping (prefer exact entries)
        if word in self.lexicon:
            return list(self.lexicon[word])

        # If the word is a known onomatopoeic whole-word, treat it as a single unit.
        # This allows "boom" to be preserved rather than split into 'b','oo','m'
        if word in self.onomatopoeia:
            return [word]

        remaining = word
        result = []

        # Greedy loop: take the longest matching token/sequence at each step.
        while remaining:
            matched = None
            for tok in self.sorted_tokens:
                # We purposely compare lowercase substrings, so ensure tok is lowercase for matching:
                if remaining.startswith(tok.lower()):
                    matched = tok
                    # If the matched candidate is a lexicon key (e.g., "thorough"),
                    # expand it into the token sequence from the lexicon, otherwise
                    # append the token name itself (e.g., "Ra3" or "ai").
                    if tok in self.lexicon:
                        result.extend(self.lexicon[tok])
                    else:
                        result.append(tok)
                    remaining = remaining[len(tok):]  # consume matched portion
                    break
            if not matched:
                # Fallback: consume one character (safe default). We use lowercased char.
                result.append(remaining[0])
                remaining = remaining[1:]
        return result

    def map_sentence(self, sentence: str):
        """
        Map an entire sentence (string) to token sequence.
        We extract words with a regular expression to skip punctuation, then
        map each word individually and concatenate results.
        """
        # Accept letters, numbers, hyphens and many accented letters (basic Unicode range).
        words = re.findall(r"[A-Za-z0-9\-\u00C0-\u024F]+", sentence)
        tokens = []
        for w in words:
            tokens.extend(self.map_word(w))
        return tokens

# --- Utilities to convert between pydub AudioSegment and numpy arrays ----------
def audiosegment_to_numpy(audioseg: AudioSegment):
    """
    Convert a pydub AudioSegment to a numpy float32 array (range -1..1) and sample rate.
    We average channels to mono if necessary. This representation is needed for librosa.
    """
    sr = audioseg.frame_rate                      # get sample rate (e.g., 22050 or 44100)
    samples = np.array(audioseg.get_array_of_samples())  # raw integer samples
    if audioseg.channels > 1:
        # For multi-channel audio average channels into mono (simple, safe approach)
        samples = samples.reshape((-1, audioseg.channels)).mean(axis=1)
    sample_width = audioseg.sample_width           # bytes per sample (often 2 for 16-bit)
    max_int = float(2 ** (8 * sample_width - 1))   # maximum absolute integer value
    y = samples.astype(np.float32) / max_int       # normalize to -1..1 in float32
    return y, sr

def numpy_to_audiosegment(y: np.ndarray, sr: int):
    """
    Convert a numpy float32 array (-1..1) into a pydub AudioSegment (mono, 16-bit).
    We clip values to avoid overflow. This is used to return processed audio back
    to pydub after using librosa effects.
    """
    y = np.clip(y, -1.0, 1.0)                      # ensure no sample is outside range
    int16 = (y * 32767).astype(np.int16)           # convert to 16-bit PCM integers
    seg = AudioSegment(data=int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    return seg

# --- Synthesizer --------------------------------------------------------------
class Synthesizer:
    """
    The synthesizer is responsible for:
      - ensuring individual tokens are loaded and normalized
      - concatenating tokens with a small crossfade
      - optionally applying pitch/tone adjustments per token
      - optionally highlighting tokens (beep) for GUI visibility
    """
    def __init__(self, db: SyllableDB, crossfade_ms: int = 10, normalize: bool = True):
        self.db = db                                # reference to token DB
        self.crossfade_ms = int(crossfade_ms)       # default crossfade length (ms)
        self.normalize = normalize                  # whether to normalize loudness
        self.target_dBFS = -20.0                    # target loudness (dBFS) after normalization

    def _ensure_audio(self, unit: SyllableUnit):
        """
        Ensure the SyllableUnit has its AudioSegment loaded, and optionally
        normalize its loudness to a consistent target. This keeps perceived
        volume similar across differently-recorded tokens.
        """
        unit.load()                                 # load the WAV if necessary
        if self.normalize and unit.audio is not None:
            try:
                current = unit.audio.dBFS          # measure current loudness (dBFS)
            except Exception:
                current = None
            if current is not None and np.isfinite(current):
                change_in_dBFS = self.target_dBFS - current   # compute adjustment
                unit.audio = unit.audio.apply_gain(change_in_dBFS)  # apply dB gain

    def synthesize(self, token_sequence: list, apply_pitch_fn: Optional[Callable]=None,
                   highlight_tokens: Optional[list] = None) -> Optional[AudioSegment]:
        """
        Concatenate a sequence of tokens into a single AudioSegment.
        - token_sequence: list of token names or lexicon-expanded strings
        - apply_pitch_fn: optional function(seg, token) -> seg to alter pitch/tones
        - highlight_tokens: list of tokens to visually/audibly mark (beep before them)
        Returns an AudioSegment or None if nothing was produced.
        """
        output = None               # will hold the growing output AudioSegment

        # iterate over each token requested
        for tok in token_sequence:
            unit = self.db.get(tok)             # find the SyllableUnit in DB
            if unit is None:
                # if token isn't available, use a short silent placeholder
                seg = AudioSegment.silent(duration=50)
            else:
                self._ensure_audio(unit)       # lazy-load + normalize the token audio
                seg = unit.audio or AudioSegment.silent(duration=50)

                # If a pitch/tone function is provided, apply it.
                if apply_pitch_fn:
                    try:
                        seg = apply_pitch_fn(seg, tok)
                    except Exception as e:
                        # non-fatal: print and continue using original segment
                        print(f"Pitch adjust failed for {tok}: {e}")

            # If this token is requested to be highlighted, prepend a tiny beep.
            # This makes diphthongs/triphthongs visible when the GUI plays the sequence.
            if highlight_tokens and tok in highlight_tokens:
                beep = Sine(800).to_audio_segment(duration=50)  # short 800Hz beep
                seg = AudioSegment.silent(duration=50) + beep + seg + AudioSegment.silent(duration=50)

            # Concatenate: if output empty, set to this segment; otherwise append with crossfade
            if output is None:
                output = seg
            else:
                # choose crossfade not longer than either segment
                cf = min(self.crossfade_ms, len(output), len(seg))
                # if cf > 0 we append with short crossfade to smooth edges
                output = output.append(seg, crossfade=cf) if cf > 0 else output + seg

        return output

# --- Pitch adjustment helpers -------------------------------------------------
def pitch_shift_librosa(audio_segment: AudioSegment, n_steps: float):
    """
    Shift pitch by n_steps (semitones) using librosa if available.
    We convert AudioSegment -> numpy -> librosa -> numpy -> AudioSegment.
    """
    if librosa is None:
        raise RuntimeError("librosa required for pitch shifting")  # clearly inform user
    y, sr = audiosegment_to_numpy(audio_segment)                 # convert to numpy
    # librosa expects float32 mono array
    y_shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr, n_steps=n_steps)
    return numpy_to_audiosegment(y_shifted, sr)                 # convert back to AudioSegment

def simple_tone_pitch_adjust(seg: AudioSegment, token: str):
    """
    Simple tone mapping that interprets trailing digits as tones and maps them
    to approximate pitch shift in semitones. This is intentionally conservative:
    - '3' -> raise slightly (e.g., 0.5 semitone)
    - '2' -> neutral (0 semitone)
    - '1' -> lower slightly (-0.6 semitone)
    This function is simple and for demonstration. For higher quality tonal
    manipulation you'd implement time-preserving formant-aware transforms.
    """
    tone_digit = None
    # check last character of token for tone digit
    if token and token[-1] in "123":
        tone_digit = token[-1]
    if tone_digit is None:
        return seg  # no tone info -> nothing to do

    # mapping chosen to be subtle (small semitone shifts). You can tune these.
    mapping = {'3': 0.5, '2': 0.0, '1': -0.6}
    n_steps = mapping.get(tone_digit, 0.0)
    if abs(n_steps) < 1e-6:
        return seg  # effectively no shift required

    try:
        return pitch_shift_librosa(seg, n_steps)
    except Exception:
        # If librosa is not installed or an error occurs, return original segment.
        return seg

# --- Graphical (Tkinter) live demo ------------------------------------------
def start_live_gui(synth: Synthesizer, mapper: TextToSyllableMapper):
    """
    Start a simple Tkinter GUI for live typing and playback.
    The GUI highlights diphthongs/triphthongs in red and beeps before highlighted tokens.
    """
    import tkinter as tk  # local import so script can run in CLI-only environments
    import threading       # used to play audio off the main thread
    import re

    # Create main window
    root = tk.Tk()
    root.title("Tonal TTS - Live Typing Demo")
    root.geometry("750x500")
    root.configure(bg="#F0F0F0")

    # Instruction label at top
    instr_frame = tk.Frame(root, bg="#F0F0F0")
    instr_frame.pack(fill="x", padx=8, pady=6)
    instr = tk.Label(
        instr_frame,
        text="Type text or tokenized tokens (e.g., 'Ra3 a1 Vi3'). Diphthongs/triphthongs highlighted in red.",
        anchor="w",
        justify="left",
        bg="#F0F0F0",
        fg="#333333"
    )
    instr.pack(fill='x')

    # Small example button area (click to insert examples and play)
    example_frame = tk.Frame(root, bg="#F0F0F0")
    example_frame.pack(fill="x", padx=8, pady=(0,6))
    tk.Label(example_frame, text="Examples (click to insert):", bg="#F0F0F0").pack(anchor='w')
    wrap = tk.Frame(example_frame, bg="#F0F0F0")
    wrap.pack(fill='x', pady=2)

    # Choose a few example keys from lexicon or tokens to populate quick buttons
    example_keys = sorted(list(mapper.lexicon.keys()))[:20] if mapper.lexicon else sorted(mapper.db.available_tokens(), key=len)[:20]

    # helper to insert text into the box and play
    def insert_and_play_phrase(phrase_text: str):
        text.delete('1.0', 'end')          # clear previous text
        text.insert('1.0', phrase_text)    # insert example phrase
        highlight_diphthongs_triphthongs() # highlight multi-vowel combos
        play_text_later()                  # play the resulting sequence

    # create colored buttons for examples
    colors = ["#8A2BE2", "#5F9EA0", "#FF7F50", "#3CB371"]
    for i, key in enumerate(example_keys):
        b = tk.Button(
            wrap,
            text=key,
            width=12,
            bg=colors[i % len(colors)],
            fg="white",
            activebackground="#D8BFD8",
            activeforeground="black",
            command=lambda k=key: insert_and_play_phrase(k)
        )
        b.grid(row=i//4, column=i%4, padx=3, pady=3)

    # Multi-line text box for input
    text = tk.Text(root, height=6, width=80)
    text.pack(padx=8, pady=6)
    text.focus_set()

    # status label at the bottom
    status = tk.Label(root, text="Ready", anchor="w", bg="#F0F0F0", fg="#333333")
    status.pack(fill="x", padx=8)

    # configure a text tag that will color diphthongs/triphthongs in red
    text.tag_configure("diphthong", foreground="red")

    # Highlighting function: find occurrences of diphthongs & triphthongs in text and tag them
    def highlight_diphthongs_triphthongs():
        content = text.get("1.0", "end-1c").lower()  # get current content and lowercase
        text.tag_remove("diphthong", "1.0", "end")    # clear previous tags
        # iterate all diphthongs and triphthongs from the mapper and add tag for each found occurrence
        for dt in mapper.diphthongs + mapper.triphthongs:
            start = 0
            while True:
                idx = content.find(dt, start)           # find next occurrence
                if idx == -1:
                    break
                start_idx = f"1.0 + {idx} chars"        # Tkinter text indices: "line.char"
                end_idx = f"1.0 + {idx + len(dt)} chars"
                text.tag_add("diphthong", start_idx, end_idx)
                start = idx + 1                         # continue searching after this occurrence

    # Playback function (called after typing debounce or Return)
    def play_text_later():
        txt = text.get("1.0", "end").strip()           # full text
        if not txt:
            status.config(text="Empty input")
            return
        # decide if the input is tokenized (contains digits like "Ra3") or plain text
        if re.search(r"\d", txt):
            tokens = txt.split()                      # split on whitespace into explicit tokens
        else:
            tokens = mapper.map_sentence(txt)        # map orthographic sentence into tokens via mapper

        unknowns = [t for t in tokens if synth.db.get(t) is None]  # tokens not present in DB
        if unknowns:
            status.config(text=f"Unknown tokens: {unknowns} (playing known tokens)")
            # filter out unknown tokens - continue playing known ones
            tokens = [t for t in tokens if synth.db.get(t)]
            if not tokens:
                return

        # tokens to highlight are those which are diphthongs/triphthongs (present in mapper lists)
        highlighted = [t for t in tokens if t in mapper.diphthongs + mapper.triphthongs]

        # Synthesize audio for token list, applying tone pitch adjustment
        audio = synth.synthesize(tokens, apply_pitch_fn=simple_tone_pitch_adjust, highlight_tokens=highlighted)
        if audio:
            # Play on a background thread so GUI stays responsive
            threading.Thread(target=lambda: play(audio), daemon=True).start()
            status.config(text=f"Playing {len(tokens)} tokens — {len(audio)/1000:.2f}s")
        else:
            status.config(text="No audio produced")

        highlight_diphthongs_triphthongs()  # refresh highlight tags after play

    # Small debounce: call play_text_later 300ms after typing stops
    debounce = {'job': None}
    def on_key_release(event):
        if debounce['job'] is not None:
            root.after_cancel(debounce['job'])
        debounce['job'] = root.after(300, play_text_later)

    # Bind events: call debounce on key release; Enter triggers immediate play
    text.bind("<KeyRelease>", on_key_release)
    text.bind("<Return>", lambda e: play_text_later())

    # start Tk main loop (blocking until GUI closed)
    root.mainloop()

# --- Helpers for file output and bulk renaming --------------------------------
def save_audio_segment_to_wav(segment: AudioSegment, outpath: str):
    """Export a pydub AudioSegment to a WAV on disk."""
    segment.export(outpath, format="wav")

def batch_rename_to_arpabet(folder_in: str, mapping_json: str, folder_out: str):
    """
    Convert your raw filenames into ARPAbet (or other orthography) using a mapping JSON.
    - folder_in: directory with original WAV files
    - mapping_json: JSON mapping from orthographic base -> arpabet (without tone)
    - folder_out: destination directory where new files will be written
    This is useful to standardize token names based on a mapping file.
    """
    if not os.path.exists(folder_out):
        os.makedirs(folder_out, exist_ok=True)
    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    for fname in os.listdir(folder_in):
        if not fname.lower().endswith(".wav"):
            continue
        base = os.path.splitext(fname)[0]          # name without extension
        tone = ""
        if base and base[-1] in "123":
            orth = base[:-1]                       # orthographic portion (without tone digit)
            tone = base[-1]                        # tone digit preserved
        else:
            orth = base
        arp = mapping.get(orth, None)              # look up mapping
        new_name = f"{arp}{tone}.wav" if arp else fname
        # copy file bytes to new destination name
        with open(os.path.join(folder_in, fname), "rb") as src, open(os.path.join(folder_out, new_name), "wb") as dst:
            dst.write(src.read())
    meta = {"source": folder_in, "mapping_used": mapping_json}
    with open(os.path.join(folder_out, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

# --- CLI / main entrypoint ---------------------------------------------------
def main():
    """
    Parse arguments and run as CLI, GUI, or demo. The CLI also supports playing
    single tokens passed with --play.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Tonal concatenative TTS demo.")
    parser.add_argument("--db", type=str, default="syllables")          # DB folder
    parser.add_argument("--lexicon", type=str, default="lexicon.json")  # lexicon path
    parser.add_argument("--gui", action="store_true")                   # start GUI
    parser.add_argument("--play", type=str, default=None)               # play this token/word
    parser.add_argument("--demo", action="store_true")                  # run one-shot demo
    args = parser.parse_args()

    # Ensure the DB folder exists to avoid confusing errors later
    if not os.path.exists(args.db):
        print(f'Error: DB folder "{args.db}" not found. Create "{args.db}" with .wav files first.')
        sys.exit(1)

    # instantiate DB, mapper and synthesizer
    db = SyllableDB(args.db)
    mapper = TextToSyllableMapper(db, lexicon_path=args.lexicon if os.path.exists(args.lexicon) else None)
    synth = Synthesizer(db, crossfade_ms=12, normalize=True)

    # If --play is provided, synthesize and play/save once then exit
    if args.play:
        s = args.play.strip()
        # If input contains digits, assume tokenized input (like "Ra3 a1 Vi3")
        if re.search(r"\d", s):
            tokens = s.split()
        else:
            tokens = mapper.map_sentence(s)   # map orthographic to tokens
        unknowns = [t for t in tokens if db.get(t) is None]
        if unknowns:
            print("Warning: unknown tokens (they'll be skipped):", unknowns)
            tokens = [t for t in tokens if db.get(t)]
        if not tokens:
            print("No playable tokens found.")
            sys.exit(1)
        audio = synth.synthesize(tokens, apply_pitch_fn=simple_tone_pitch_adjust)
        if audio:
            outpath = "demo_output.wav"
            save_audio_segment_to_wav(audio, outpath)
            print(f"Synthesized saved to {outpath}; playing now...")
            play(audio)  # synchronous play
        sys.exit(0)

    # Launch GUI if requested
    if args.gui:
        start_live_gui(synth, mapper)
        sys.exit(0)

    # One-shot interactive demo
    if args.demo:
        print("Interactive demo: enter tokenized tokens (e.g., 'Ra3 a1 Vi3') or orthographic text.")
        txt = input("Enter sentence: ").strip()
        if not txt:
            print("Empty input; exiting.")
            sys.exit(0)
        if re.search(r"\d", txt):
            tokens = txt.split()
        else:
            tokens = mapper.map_sentence(txt)
        print("Mapped tokens:", tokens)
        audio = synth.synthesize(tokens, apply_pitch_fn=simple_tone_pitch_adjust)
        if audio:
            outpath = "demo_output.wav"
            save_audio_segment_to_wav(audio, outpath)
            print(f"Synthesized saved to {outpath}; playing now...")
            play(audio)
        sys.exit(0)

    # Otherwise run interactive CLI REPL
    print("Tonal TTS CLI. Type sentence or tokenized tokens (e.g., 'Ra3 a1 Vi3'). 'exit' to quit.")
    while True:
        s = input(">>> ").strip()
        if s.lower() in ("exit","quit"):
            break
        if not s:
            continue
        if re.search(r"\d", s):
            tokens = s.split()
        else:
            tokens = mapper.map_sentence(s)
        print("Tokens:", tokens)
        unknowns = [t for t in tokens if db.get(t) is None]
        if unknowns:
            print("Warning: unknown tokens:", unknowns)
            tokens = [t for t in tokens if db.get(t)]
            if not tokens:
                print("No playable tokens.")
                continue
        audio = synth.synthesize(tokens, apply_pitch_fn=simple_tone_pitch_adjust)
        if audio:
            # Play in background thread to avoid blocking the prompt
            threading.Thread(target=lambda: play(audio), daemon=True).start()
            print(f"Playing {len(tokens)} tokens — {len(audio)/1000.0:.2f}s")
        else:
            print("No audio to play.")

# standard Python idiom to allow import or direct execution
if __name__ == "__main__":
    main()