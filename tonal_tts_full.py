#!/usr/bin/env python3
"""
tonal_tts_full.py
Complete concatenative syllable-based tonal TTS reference implementation.
(Updated: PlaybackManager + safe queueing + inter-syllable silence to avoid overlapping echo)
"""

# --- Standard library imports ------------------------------------------------
import os
import sys
import re
import json
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict

# --- Third-party imports -----------------------------------------------------
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine

import numpy as np
try:
    import librosa
except Exception:
    librosa = None

# --- Helpers -----------------------------------------------------------------
import re
def normalize_key(k):
    return re.sub(r'[^a-z0-9\-]', '', k.lower())

def map_word_to_tokens(word, lexicon):
    key = word.strip()
    if not key:
        return []

    kl = key.lower()
    if kl in lexicon and lexicon[kl]:
        return lexicon[kl]

    kn = normalize_key(kl)
    if kn in lexicon and lexicon[kn]:
        return lexicon[kn]

    kn2 = re.sub(r'_[0-9]+$','', kn)
    if kn2 in lexicon and lexicon[kn2]:
        return lexicon[kn2]

    for L in range(len(kn), 0, -1):
        sub = kn[:L]
        if sub in lexicon and lexicon[sub]:
            return lexicon[sub]

    parts = re.split(r'[\s_\-]+', key)
    tokens = []
    for p in parts:
        pkn = normalize_key(p)
        if pkn in lexicon and lexicon[pkn]:
            tokens.extend(lexicon[pkn])
        else:
            pass
    return tokens

TOKEN_TONE_RE = re.compile(r"^(.+?)(\d+)$")
def split_token_and_tone(token: str):
    m = TOKEN_TONE_RE.match(token)
    if m:
        return m.group(1), m.group(2)
    return token, None

# --- Data container ----------------------------------------------------------
@dataclass
class SyllableUnit:
    token: str
    path: str
    audio: Optional[AudioSegment] = None
    duration_ms: Optional[int] = None

    def load(self):
        if self.audio is None:
            self.audio = AudioSegment.from_file(self.path, format="wav")
            self.duration_ms = len(self.audio)

# --- DB ---------------------------------------------------------------------
class SyllableDB:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.units: Dict[str, SyllableUnit] = {}
        self.meta = {}
        self._load()

    def _load(self):
        meta_path = os.path.join(self.folder_path, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                print("Warning: could not parse meta.json; ignoring.")

        for dirpath, dirs, files in os.walk(self.folder_path):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                token = os.path.splitext(fname)[0]
                full = os.path.join(dirpath, fname)
                if token in self.units:
                    continue
                self.units[token] = SyllableUnit(token=token, path=full)
                lower = token.lower()
                if lower not in self.units:
                    self.units[lower] = self.units[token]

        aliases = self.meta.get("aliases", {})
        for alias, canonical in aliases.items():
            if canonical in self.units:
                self.units[alias] = self.units[canonical]

    def get(self, token: str) -> Optional[SyllableUnit]:
        if token is None:
            return None
        unit = self.units.get(token)
        if unit:
            return unit
        return self.units.get(token.lower())

    def available_tokens(self) -> List[str]:
        return list({u.token for u in self.units.values()})

# --- Text -> syllable mapping ------------------------------------------------
class TextToSyllableMapper:
    def __init__(self, db: SyllableDB, lexicon_path: Optional[str]=None):
        self.db = db
        self.lexicon = {}
        if lexicon_path and os.path.exists(lexicon_path):
            try:
                with open(lexicon_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.lexicon = {k.lower(): v for k, v in raw.items()}
            except Exception:
                print("Warning: could not parse lexicon.json; ignoring.")

        self.diphthongs = [
            "ai", "au", "ei", "oi", "ou", "ia", "ie", "io", "ua", "ue", "uo", "eu"
        ]
        self.triphthongs = ["iau", "uai", "eau", "iou"]
        self.onomatopoeia = set([
            "boom", "bang", "crash", "meow", "woof", "beep", "ding", "buzz", "hiss", "clack"
        ])

        lex_keys = sorted(self.lexicon.keys(), key=len, reverse=True)
        combined = lex_keys + self.diphthongs + self.triphthongs + list(self.onomatopoeia)
        self.sorted_tokens = sorted(set(combined), key=len, reverse=True)

    def map_word(self, word: str):
        word = word.strip().lower()
        if not word:
            return []

        if word in self.lexicon:
            return list(self.lexicon[word])
        if word in self.onomatopoeia:
            return [word]

        remaining = word
        result = []
        while remaining:
            matched = None
            for tok in self.sorted_tokens:
                if remaining.startswith(tok.lower()):
                    matched = tok.lower()
                    if matched in self.lexicon:
                        result.extend(self.lexicon[matched])
                    else:
                        result.append(tok.lower())
                    remaining = remaining[len(tok):]
                    break
            if not matched:
                result.append(remaining[0])
                remaining = remaining[1:]
        return result

    def map_sentence(self, sentence: str):
        words = re.findall(r"[A-Za-z0-9\-\u00C0-\u024F]+", sentence)
        tokens = []
        for w in words:
            mapped = map_word_to_tokens(w, self.lexicon)
            if not mapped:
                print(f"[WARN] Could not map word: '{w}'")
            tokens.extend(mapped)
        return tokens

# --- Num / audio conversion helpers -----------------------------------------
def audiosegment_to_numpy(audioseg: AudioSegment):
    sr = audioseg.frame_rate
    samples = np.array(audioseg.get_array_of_samples())
    if audioseg.channels > 1:
        samples = samples.reshape((-1, audioseg.channels)).mean(axis=1)
    sample_width = audioseg.sample_width
    max_int = float(2 ** (8 * sample_width - 1))
    y = samples.astype(np.float32) / max_int
    return y, sr

def numpy_to_audiosegment(y: np.ndarray, sr: int):
    y = np.clip(y, -1.0, 1.0)
    int16 = (y * 32767).astype(np.int16)
    seg = AudioSegment(data=int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    return seg

# --- PlaybackManager: single worker queue (prevents overlap) -----------------
class PlaybackManager:
    """
    Single background worker that plays AudioSegment jobs sequentially from a queue.
    play(audio, clear_queue=True) empties queue (optional) then enqueues new audio.
    """
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get()
                if item is None:
                    # sentinel: stop worker
                    break
                audio = item
                try:
                    play(audio)  # blocking call until audio finishes
                except Exception as e:
                    print("[PLAYBACK ERROR]", e)
                finally:
                    self._queue.task_done()
            except Exception as e:
                print("[PLAYBACK WORKER ERROR]", e)
                time.sleep(0.1)

    def play(self, audio: AudioSegment, clear_queue: bool = True):
        """
        Enqueue audio for playback.
        If clear_queue=True, previously queued items are discarded.
        """
        if audio is None:
            return
        if clear_queue:
            # drain existing items
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
        self._queue.put(audio)

    def stop(self):
        self._stop_event.set()
        # enqueue sentinel to unblock worker
        self._queue.put(None)
        self._thread.join(timeout=1.0)

# --- Synthesizer -------------------------------------------------------------
class Synthesizer:
    """
    Synthesizer concatenates tokens adding short silence between tokens to avoid overlap
    and optionally applies pitch adjustments. Default inter_silence_ms reduces smear.
    """
    def __init__(self, db: SyllableDB, crossfade_ms: int = 10, inter_silence_ms: int = 12, normalize: bool = True):
        self.db = db
        self.crossfade_ms = int(crossfade_ms)
        self.inter_silence_ms = int(inter_silence_ms)
        self.normalize = normalize
        self.target_dBFS = -20.0

    def _ensure_audio(self, unit: SyllableUnit):
        unit.load()
        if self.normalize and unit.audio is not None:
            try:
                current = unit.audio.dBFS
            except Exception:
                current = None
            if current is not None and np.isfinite(current):
                change_in_dBFS = self.target_dBFS - current
                unit.audio = unit.audio.apply_gain(change_in_dBFS)

    def synthesize(self, token_sequence: list, apply_pitch_fn: Optional[Callable]=None,
                   highlight_tokens: Optional[list] = None) -> Optional[AudioSegment]:
        output = None
        for idx, tok in enumerate(token_sequence):
            unit = self.db.get(tok)
            if unit is None:
                seg = AudioSegment.silent(duration=50)
            else:
                self._ensure_audio(unit)
                seg = unit.audio or AudioSegment.silent(duration=50)
                if apply_pitch_fn:
                    try:
                        seg = apply_pitch_fn(seg, tok)
                    except Exception as e:
                        print(f"Pitch adjust failed for {tok}: {e}")

            if highlight_tokens and tok in highlight_tokens:
                beep = Sine(800).to_audio_segment(duration=50)
                seg = AudioSegment.silent(duration=20) + beep + AudioSegment.silent(duration=20) + seg

            if output is None:
                output = seg
            else:
                # Insert a short silence between tokens rather than heavy crossfade to avoid smearing/echo
                inter = AudioSegment.silent(duration=self.inter_silence_ms)
                output = output + inter + seg

        return output

# --- Pitch helpers -----------------------------------------------------------
def pitch_shift_librosa(audio_segment: AudioSegment, n_steps: float):
    if librosa is None:
        raise RuntimeError("librosa required for pitch shifting")
    y, sr = audiosegment_to_numpy(audio_segment)
    y_shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr, n_steps=n_steps)
    return numpy_to_audiosegment(y_shifted, sr)

def simple_tone_pitch_adjust(seg: AudioSegment, token: str):
    tone_digit = None
    if token and token[-1] in "123":
        tone_digit = token[-1]
    if tone_digit is None:
        return seg
    mapping = {'3': 0.5, '2': 0.0, '1': -0.6}
    n_steps = mapping.get(tone_digit, 0.0)
    if abs(n_steps) < 1e-6:
        return seg
    try:
        return pitch_shift_librosa(seg, n_steps)
    except Exception:
        return seg

# --- GUI (Tkinter) live demo -----------------------------------------------
def start_live_gui(synth: Synthesizer, mapper: TextToSyllableMapper, playback_manager: PlaybackManager):
    import tkinter as tk
    import re

    root = tk.Tk()
    root.title("Tonal TTS - Live Typing Demo")
    root.geometry("750x500")
    root.configure(bg="#F0F0F0")

    instr_frame = tk.Frame(root, bg="#F0F0F0")
    instr_frame.pack(fill="x", padx=8, pady=6)
    instr = tk.Label(
        instr_frame,
        text="Type text or tokenized tokens (e.g., 'Ra3 a1 Vi3'). Diphthongs/triphthongs highlighted in red.",
        anchor="w", justify="left", bg="#F0F0F0", fg="#333333"
    )
    instr.pack(fill='x')

    example_frame = tk.Frame(root, bg="#F0F0F0")
    example_frame.pack(fill="x", padx=8, pady=(0,6))
    tk.Label(example_frame, text="Examples (click to insert):", bg="#F0F0F0").pack(anchor='w')
    wrap = tk.Frame(example_frame, bg="#F0F0F0")
    wrap.pack(fill='x', pady=2)

    example_keys = sorted(list(mapper.lexicon.keys()))[:20] if mapper.lexicon else sorted(mapper.db.available_tokens(), key=len)[:20]

    def insert_and_play_phrase(phrase_text: str):
        text.delete('1.0', 'end')
        key = phrase_text.lower()
        if key in mapper.lexicon and mapper.lexicon[key]:
            token_to_insert = mapper.lexicon[key][0]
            text.insert('1.0', token_to_insert)
        else:
            text.insert('1.0', phrase_text)
        highlight_diphthongs_triphthongs()
        play_text_later(clear_queue=True)

    colors = ["#8A2BE2", "#5F9EA0", "#FF7F50", "#3CB371"]
    for i, key in enumerate(example_keys):
        b = tk.Button(wrap, text=key, width=12, bg=colors[i % len(colors)], fg="white",
                      activebackground="#D8BFD8", activeforeground="black",
                      command=lambda k=key: insert_and_play_phrase(k))
        b.grid(row=i//4, column=i%4, padx=3, pady=3)

    text = tk.Text(root, height=6, width=80)
    text.pack(padx=8, pady=6)
    text.focus_set()

    status = tk.Label(root, text="Ready", anchor="w", bg="#F0F0F0", fg="#333333")
    status.pack(fill="x", padx=8)

    text.tag_configure("diphthong", foreground="red")

    def highlight_diphthongs_triphthongs():
        content = text.get("1.0", "end-1c").lower()
        text.tag_remove("diphthong", "1.0", "end")
        for dt in mapper.diphthongs + mapper.triphthongs:
            start = 0
            while True:
                idx = content.find(dt, start)
                if idx == -1:
                    break
                start_idx = f"1.0 + {idx} chars"
                end_idx = f"1.0 + {idx + len(dt)} chars"
                text.tag_add("diphthong", start_idx, end_idx)
                start = idx + 1

    def play_text_later(clear_queue: bool = True):
        txt = text.get("1.0", "end").strip()
        if not txt:
            status.config(text="Empty input")
            return
        if re.search(r"\d", txt):
            tokens = txt.split()
        else:
            tokens = mapper.map_sentence(txt)

        unknowns = [t for t in tokens if synth.db.get(t) is None]
        if unknowns:
            status.config(text=f"Unknown tokens: {unknowns} (playing known tokens)")
            tokens = [t for t in tokens if synth.db.get(t)]
            if not tokens:
                return

        highlighted = [t for t in tokens if t in mapper.diphthongs + mapper.triphthongs]

        audio = synth.synthesize(tokens, apply_pitch_fn=simple_tone_pitch_adjust, highlight_tokens=highlighted)
        if audio:
            # Use playback manager: clear_queue=True for immediate play
            playback_manager.play(audio, clear_queue=clear_queue)
            status.config(text=f"Queued {len(tokens)} tokens — {len(audio)/1000:.2f}s")
        else:
            status.config(text="No audio produced")

        highlight_diphthongs_triphthongs()

    debounce = {'job': None}
    def on_key_release(event):
        if debounce['job'] is not None:
            root.after_cancel(debounce['job'])
        debounce['job'] = root.after(300, lambda: play_text_later(clear_queue=True))

    text.bind("<KeyRelease>", on_key_release)
    text.bind("<Return>", lambda e: play_text_later(clear_queue=True))

    root.mainloop()

# --- File helpers ------------------------------------------------------------
def save_audio_segment_to_wav(segment: AudioSegment, outpath: str):
    segment.export(outpath, format="wav")

def batch_rename_to_arpabet(folder_in: str, mapping_json: str, folder_out: str):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out, exist_ok=True)
    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    for fname in os.listdir(folder_in):
        if not fname.lower().endswith(".wav"):
            continue
        base = os.path.splitext(fname)[0]
        tone = ""
        if base and base[-1] in "123":
            orth = base[:-1]
            tone = base[-1]
        else:
            orth = base
        arp = mapping.get(orth, None)
        new_name = f"{arp}{tone}.wav" if arp else fname
        with open(os.path.join(folder_in, fname), "rb") as src, open(os.path.join(folder_out, new_name), "wb") as dst:
            dst.write(src.read())
    meta = {"source": folder_in, "mapping_used": mapping_json}
    with open(os.path.join(folder_out, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

# --- CLI / main --------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tonal concatenative TTS demo.")
    parser.add_argument("--db", type=str, default="syllables")
    parser.add_argument("--lexicon", type=str, default="lexicon.json")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--play", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f'Error: DB folder "{args.db}" not found. Create "{args.db}" with .wav files first.')
        sys.exit(1)

    db = SyllableDB(args.db)
    mapper = TextToSyllableMapper(db, lexicon_path=args.lexicon if os.path.exists(args.lexicon) else None)
    synth = Synthesizer(db, crossfade_ms=12, inter_silence_ms=12, normalize=True)

    # create global playback manager for this run
    playback_manager = PlaybackManager()

    if args.play:
        s = args.play.strip()
        if re.search(r"\d", s):
            tokens = s.split()
        else:
            tokens = mapper.map_sentence(s)
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
            # synchronous play for --play mode
            play(audio)
        sys.exit(0)

    if args.gui:
        start_live_gui(synth, mapper, playback_manager)
        playback_manager.stop()
        sys.exit(0)

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

    print("Tonal TTS CLI. Type sentence or tokenized tokens (e.g., 'Ra3 a1 Vi3'). 'exit' to quit.")
    try:
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
                # enqueue for playback - keep queued plays (clear_queue=False)
                playback_manager.play(audio, clear_queue=False)
                print(f"Queued {len(tokens)} tokens — {len(audio)/1000.0:.2f}s")
            else:
                print("No audio to play.")
    finally:
        playback_manager.stop()

if __name__ == "__main__":
    main()