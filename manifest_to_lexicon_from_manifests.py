"""
Build lexicon.json by scanning manifest.csv files under ./syllables.
Produces a JSON mapping: orthographic_key -> [tokenName, ...]
Usage:
  python3 manifest_to_lexicon_from_manifests.py --root ./syllables --out lexicon_from_manifests.json
"""
import os, csv, json, argparse, re
from collections import defaultdict

def orth_key_from_parsed(parsed):
    # parsed is like "Ra/3" or "aad/1" -> return "ra" or "aad"
    if not parsed:
        return None
    return parsed.split('/')[0].strip().lower()

def token_from_arpabet(arp):
    if not arp:
        return None
    return os.path.splitext(arp.strip())[0]

def build_from_manifests(root):
    lex = defaultdict(list)
    for dirpath, dirs, files in os.walk(root):
        if "manifest.csv" in files:
            path = os.path.join(dirpath, "manifest.csv")
            try:
                with open(path, newline='', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        parsed = row.get("parsed") or row.get("Parsed") or row.get("parsed_text") or ""
                        arp = row.get("arpabet_name") or row.get("arpabet") or row.get("arpabet_name ")
                        orig = row.get("orig") or row.get("Orig") or ""
                        key = orth_key_from_parsed(parsed)
                        token = token_from_arpabet(arp)
                        # fallback: if parsed missing, use the orig filename (without digits/spaces)
                        if not key and orig:
                            key = re.sub(r'[^0-9a-z]', '', os.path.splitext(orig)[0].lower())
                        if key and token:
                            lex[key].append(token)
            except Exception as e:
                print("Failed to read", path, ":", e)
    # dedupe and sort token lists
    return {k: sorted(list(dict.fromkeys(v))) for k, v in lex.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./syllables")
    p.add_argument("--out", default="lexicon_from_manifests.json")
    args = p.parse_args()
    lex = build_from_manifests(args.root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(lex, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(lex)} lexicon entries to {args.out}")

if __name__ == "__main__":
    main()