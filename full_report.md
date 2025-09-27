# Tonal Syllable-Concatenation Text-to-Speech — Full Report

## Abstract
(Write a concise abstract summarizing goals, approach, and results.)

## 1. Introduction
Explain motivation (tonal African language TTS), goals, and scope.

## 2. Dataset and Preprocessing
- Recording details (100 carrier sentences; sample rate; mono).
- Praat workflow: annotate TextGrid -> export syllable WAVs.
- File naming conventions: `SyllableTone.wav` (e.g., Ra1.wav, Raa3.wav).
- Placeholders:
  - Screenshot: [FIGURE_PLACEHOLDER_1.png] (Praat showing a TextGrid)
  - Screenshot: [FIGURE_PLACEHOLDER_2.png] (syllables folder listing)

## 3. ARPABET and Tone Encoding
Describe mapping to ARPABET and tone numbering (1 low, 2 mid, 3 high). Explain long vowels and falling/rising tone representation (e.g., Ra3a1).

## 4. System Design
### 4.1 Architecture
Pipeline: Input → Tokenization (lexicon) → Unit selection (syllable WAVs) → Concatenation (crossfade + normalize) → Playback.

### 4.2 Key modules
- `SyllableDB` — loads WAV units
- `TextToSyllableMapper` — maps orthography → token sequence
- `Synthesizer` — concatenates with crossfade; optional pitch adjust

## 5. Implementation
- Python code files included:
  - `tonal_tts_full.py`
  - `lexicon_generator.py`
  - `mfcc_eval.py`
- How to run (commands are in README section)

## 6. Handling Diphthongs & Triphthongs
- Explain split vs combined units; list examples (from your instructions).
- Mention `pairs.json` used to select evaluation pairs.

## 7. Pitch & Tone Handling
- Concatenation preserves recorded tone.
- Optional tiny pitch shifts via librosa (experimental).
- Limitations: artifacts, coarticulation.

## 8. Evaluation
- MFCC+DTW comparisons between combined vs split units. Output file: `mfcc_distances.csv`.
- Placeholders:
  - Screenshot/plot: [FIGURE_PLACEHOLDER_3.png] (distance table or plot)
  - Audio examples: (attach `demo_output.wav` sample)

## 9. Results & Achievements
- Describe what you produced: syllable DB, GUI, evaluation CSV, starter lexicon.

## 10. Discussion
- Challenges: recording consistency, coarticulation, tone mapping.
- How addressed: normalization, crossfade, careful annotation.

## 11. Conclusion and Future Work
- Summarize outcomes and propose PSOLA / Praat pitch-tier editing and neural TTS future work.

## References
- Include "A Tutorial on Acoustic Phonetic Feature Extraction for Automatic" and any Praat / ARPABET references.

## Appendices
- Appendix A: Code listings (paste `tonal_tts_full.py`).
- Appendix B: `pairs.json` and sample `lexicon.json`.
- Appendix C: Timeline and defense checklist.

