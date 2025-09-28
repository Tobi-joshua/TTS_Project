# TTS Project — README

A beginner-friendly guide for the hand-cut syllable TTS system. This README is a concise, step-by-step reference so anyone can get the project running quickly.

---

## What this project is

A simple text-to-speech (TTS) system built from **hand-cut syllable WAV files**. The system stitches short WAV tokens (syllables) together to synthesize words and short sentences.

Think of each syllable as a Lego brick — the synthesizer picks bricks, snaps them together, and plays the result.

---

## Repo layout (important files)

```
projects/TTS_Project/
  tonal_tts_full.py                # Main synthesizer / demo script
  run_on_typing.py                 # Simple interactive CLI demo (type -> speak)
  prepare_and_run_gui.sh           # Helper to launch GUI (if present)
  mfcc_eval.py                     # MFCC distance / evaluation utilities
  manifest_to_lexicon_from_manifests.py  # Auto-generate lexicon from manifests
  lexicon_from_filenames.json      # auto-generated lexicon stub
  lexicon_from_manifests.json      # auto-generated lexicon stub
  lexicon_filtered_all.json        # filtered lexicon
  pairs.json                       # combined-token / parts pairs used by MFCC eval
  syllables/                       # hand-cut syllable WAVs (organized by sentence)
  syllables_s1/                    # small test DB (Sentence 1)
  syllables_all_links/             # (optional) symlink collection for all syllables
  mfcc_distances.csv               # MFCC evaluation outputs
  README.md                        # this file
  requirements.txt                 # Python dependencies
  venv/                            # optional local virtualenv
```

> If your repo differs slightly, adjust the commands below to match filenames.

---

## Quickstart (minimal)

Run all commands from the project root (the folder that contains `tonal_tts_full.py`).

### 1. Create & activate a virtualenv

**Linux / macOS / WSL**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

If you have a `requirements.txt` file, prefer that:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If not, the common packages used in this project are:

```bash
pip install pydub soundfile numpy librosa scipy python-docx
```

### 3. Make a quick Sentence-1 test DB

```bash
rm -rf ./syllables_s1
mkdir -p ./syllables_s1
cp "./syllables/Sentence 1/"*.wav ./syllables_s1/ 2>/dev/null || true
ls ./syllables_s1
```

### 4. Generate (or inspect) a lexicon

Auto-generate a starter lexicon (if the script exists in your repo):

```bash
python manifest_to_lexicon_from_manifests.py --out lexicon_from_manifests.json
# or
python lexicon_generator.py --syll-root ./syllables_s1 --out lexicon_sentence1.json
```

A lexicon is a JSON mapping: `{ "word": ["Token1","Token2"] }`.

### 5. Play/Render audio

**Play a single token by name:**

```bash
python tonal_tts_full.py --db ./syllables_s1 --play "Ra3"
```

**Play a word using a lexicon:**

```bash
python tonal_tts_full.py --db ./syllables --lexicon lexicon_from_manifests.json --play "ra"
```

**Run the demo:**

```bash
python tonal_tts_full.py --db ./syllables --lexicon lexicon.json --demo
```

> If audio doesn't play under WSL, open the output WAV in Windows Explorer:

```bash
explorer.exe "$(wslpath -w demo_output.wav)"
```

---

## Evaluation (MFCC distances)

Compare combined tokens vs stitched parts with `mfcc_eval.py`:

```bash
# simple run (no auto-combine)
python mfcc_eval.py --db-root ./syllables_all_links --pairs pairs.json --out mfcc_distances.csv

# run with verbose to see progress
python mfcc_eval.py --db-root ./syllables_all_links --pairs pairs.json --out mfcc_distances.csv --verbose

# if combined tokens are missing but you want to create them automatically:
python mfcc_eval.py --db-root ./syllables_all_links --pairs pairs.json --out mfcc_distances.csv --auto-combine --verbose
```

`pairs.json` entries look like:

```json
{ "combined": "Vi3uu2", "parts": ["Vi3","uu2","gu1"] }
```

Smaller MFCC distance = closer spectral match.

---

## Troubleshooting

* **Missing packages**: install the missing package using `pip install <package>` inside the venv.
* **`ModuleNotFoundError`**: ensure `venv` is activated and `pip install -r requirements.txt` succeeded.
* **WAV/audio playback errors**: install `ffmpeg` on your system and/or open the output file with a local player.
* **Script has no `--help`**: open the top lines to read usage: `sed -n '1,160p' tonal_tts_full.py`.
* **Shell scripts on Windows**: run inside WSL or convert to PowerShell.

---

## Tips & conventions

* Token names are case-sensitive (`Ra3` ≠ `ra3`).
* Start with `syllables_s1` (Sentence 1) to test before scaling to the full dataset.
* Keep original WAVs untouched; use copies or a dedicated test folder.

---

## Contributing

If you want this README shortened, expanded into separate HOWTOs, or converted into a PDF/Word, open an issue or send a PR with edits.

---

## License

Add a LICENSE file if you want to make the project open-source.

---

If you want, I can also:

* produce a short `QUICKSTART.md` containing only the minimal commands, or
* convert this README into a downloadable `README.md` file (I already created it in the document panel).

---

## GUI launcher & keyboard-run helper

To make running the GUI and preparing a sanitized DB easy, this repo includes two helper scripts:

### `prepare_and_run_gui.sh` (shell)

* Purpose: creates a sanitized symlink DB (`./syllables_all_links`), generates/filters lexicons, and launches the GUI by running `tonal_tts_full.py --gui`.
* Location: project root (make executable if needed).
* Typical usage:

```bash
# make it executable once
chmod +x prepare_and_run_gui.sh
# prepare DB and launch GUI
./prepare_and_run_gui.sh
```

Notes:

* The script checks for `python3` and warns if no virtualenv is active.
* It prefers a manifest-derived lexicon (`lexicon_from_manifests.json`) and will filter it to tokens that exist in the sanitized DB. If not present, it generates a filename-derived fallback `lexicon_from_filenames.json`.
* If the GUI requires `tkinter`, install it (e.g. `sudo apt install python3-tk` on Debian/Ubuntu).

### `run_on_typing.py` (Python)

* Purpose: a tiny terminal helper that waits for you to press Enter and then runs the shell script. Useful when you want a simple keypress-to-launch experience from a terminal.
* Usage:

```bash
python3 run_on_typing.py
# press Enter to run the prepare_and_run_gui.sh script
```

---