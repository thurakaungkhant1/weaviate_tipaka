# -*- coding: utf-8 -*-
"""
Generate a single CSV with 6 columns:
  chunk_id, chunk_text, subchunk_id, subchunk_text, sentence_id, sentence_text

Sentence rules:
- Ends with '.', '?', '!' followed by a space
- Next sentence must start with a capital letter (Unicode .isupper())
- Ignore terminators inside (), [], {}

Tokenization:
- Whitespace-based (no external deps)
- Chunk = 8000 tokens, Sub-chunk = 200 tokens

Usage:
  python scripts/make_chunk_subchunk_sentence_v2.py "data_clean/your_text.txt" \
      --out "outputs_v2/chunk_subchunk_sentence_v2.csv"
"""

import argparse
from pathlib import Path
import re
import csv
from typing import List, Tuple

# ===================== Config =====================
CHUNK_TOKEN_SIZE = 8000
SUBCHUNK_TOKEN_SIZE = 200

CHUNK_PAD    = 6   # chunk_000001
SUBCHUNK_PAD = 3   # sc_000001_001
SENTENCE_PAD = 3   # s_000001_001_001
# ==================================================

# -------- Tokenizer (whitespace) --------
def encode_tokens(text: str) -> List[str]:
    return text.split()

def decode_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)

# -------- Helpers --------
def chunk_spans(n: int, size: int) -> List[Tuple[int, int]]:
    """Return [(start, end_exclusive), ...] windows of length 'size' (last may be shorter)."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]

def build_char_offsets(tokens: List[str]) -> List[int]:
    """
    Char-start offsets of tokens in ' '.join(tokens).
    Example: ["ab","c"] -> "ab c" -> [0, 3]
    """
    offsets = []
    pos = 0
    for i, tk in enumerate(tokens):
        offsets.append(pos)
        pos += len(tk)
        if i < len(tokens) - 1:
            pos += 1  # single space between tokens
    return offsets

# -------- Strict sentence splitter --------
def split_sentences_strict(text: str) -> List[str]:
    """
    Ends with one of .?! followed by a space; next sentence starts with capital.
    Ignores .?! inside (), [], {}.
    Preserves original spacing inside each sentence.
    """
    openers = "([{"
    closers = ")]}"

    stack = []
    out: List[str] = []
    n = len(text)
    start = 0
    i = 0

    while i < n:
        ch = text[i]

        # Track bracket nesting
        if ch in openers:
            stack.append(ch)
            i += 1
            continue
        if ch in closers and stack:
            stack.pop()
            i += 1
            continue

        # Potential terminators
        if ch in ".?!":
            if not stack:
                # must be followed by a space
                if i + 1 < n and text[i + 1] == " ":
                    # next non-space should be capital
                    j = i + 2
                    while j < n and text[j].isspace():
                        j += 1
                    if j < n and text[j].isupper():
                        sent = text[start:i + 1].strip()
                        if sent:
                            out.append(sent)
                        start = j
                        i = j
                        continue
        i += 1

    tail = text[start:].strip()
    if tail:
        out.append(tail)

    return out

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Make chunk/subchunk/sentence CSV (v2 rules).")
    parser.add_argument("input", help="Path to input .txt file")
    parser.add_argument("--out", default="outputs_v2/chunk_subchunk_sentence_v2.csv",
                        help="Path to output CSV (default: outputs_v2/chunk_subchunk_sentence_v2.csv)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Read input
    text = in_path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Tokenize whole corpus
    all_tokens = encode_tokens(text)

    rows: List[List[str]] = []

    # Iterate chunks
    for c_idx, (c_start, c_end) in enumerate(chunk_spans(len(all_tokens), CHUNK_TOKEN_SIZE), start=1):
        chunk_id = f"chunk_{c_idx:0{CHUNK_PAD}d}"
        chunk_tokens = all_tokens[c_start:c_end]
        chunk_text = decode_tokens(chunk_tokens)

        # Build subchunks (by token count)
        sub_spans = chunk_spans(len(chunk_tokens), SUBCHUNK_TOKEN_SIZE)
        sub_ids: List[str] = []
        sub_texts: List[str] = []
        sub_char_ranges: List[Tuple[str, int, int]] = []  # (sub_id, char_start, char_end) within chunk_text

        chunk_token_char = build_char_offsets(chunk_tokens)

        for s_idx, (sc_start, sc_end) in enumerate(sub_spans, start=1):
            sub_id = f"sc_{c_idx:0{CHUNK_PAD}d}_{s_idx:0{SUBCHUNK_PAD}d}"
            sub_tokens = chunk_tokens[sc_start:sc_end]
            sub_text = decode_tokens(sub_tokens)

            # char range within chunk_text
            if sc_start < len(chunk_token_char):
                cstart = chunk_token_char[sc_start]
            else:
                cstart = len(chunk_text)

            last_tok = sc_end - 1
            if last_tok >= sc_start and last_tok < len(chunk_tokens):
                cend = chunk_token_char[last_tok] + len(chunk_tokens[last_tok]) - 1
                # add spaces between tokens inside the subchunk
                cend += (last_tok - sc_start)
            else:
                cend = cstart - 1  # empty subchunk case

            sub_ids.append(sub_id)
            sub_texts.append(sub_text)
            sub_char_ranges.append((sub_id, cstart, cend))

        # Split sentences in this chunk
        sentences = split_sentences_strict(chunk_text)

        # Map sentences → subchunk where they START (by char start)
        cursor = 0
        for s_order, s_text in enumerate(sentences, start=1):
            if not s_text:
                continue

            start_idx = chunk_text.find(s_text, cursor)
            if start_idx == -1:
                start_idx = chunk_text.find(s_text)  # fallback

            sub_id = ""
            sub_text_val = ""
            sub_ord = 0
            for k, (sid, cstart, cend) in enumerate(sub_char_ranges, start=1):
                if cstart <= start_idx <= cend:
                    sub_id = sid
                    sub_ord = k
                    sub_text_val = sub_texts[k - 1] if 0 <= k - 1 < len(sub_texts) else ""
                    break

            if sub_ord > 0:
                sentence_id = f"s_{c_idx:0{CHUNK_PAD}d}_{sub_ord:0{SUBCHUNK_PAD}d}_{s_order:0{SENTENCE_PAD}d}"
            else:
                sentence_id = f"s_{c_idx:0{CHUNK_PAD}d}_000_{s_order:0{SENTENCE_PAD}d}"

            rows.append([
                chunk_id, chunk_text,
                sub_id, sub_text_val,
                sentence_id, s_text
            ])

            cursor = start_idx + len(s_text)

    # Write CSV
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "chunk_text", "subchunk_id", "subchunk_text", "sentence_id", "sentence_text"])
        w.writerows(rows)

    print(f"[✓] Done: {out_path}")

if __name__ == "__main__":
    main()
