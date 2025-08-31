# -*- coding: utf-8 -*-
"""
Make a single CSV file containing:
chunk_id, chunk_text, subchunk_id, subchunk_text, sentence_id, sentence_text
from one input text file.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple
import csv

# -------- CONFIG --------
CHUNK_TOKEN_SIZE = 8000
SUBCHUNK_TOKEN_SIZE = 200
TOKENIZER_NAME = "cl100k_base"  # if using tiktoken

CHUNK_PAD = 6
SUBCHUNK_PAD = 3
SENTENCE_PAD = 3

OUTPUT_FILE = Path("outputs") / "chunk_subchunk_sentence.csv"

# -------- Tokenizer --------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding(TOKENIZER_NAME)
    def encode_txt(x: str) -> List[int]:
        return _ENC.encode(x)
    def decode_tokens(toks: List[int]) -> str:
        return _ENC.decode(toks)
    print(f"[i] Using tiktoken: {TOKENIZER_NAME}")
except Exception:
    print("[!] tiktoken not available; using whitespace fallback.")
    def encode_txt(x: str) -> List[str]:
        return x.split()
    def decode_tokens(toks: List[str]) -> str:
        return " ".join(toks)

# -------- Sentence Splitter --------
def split_into_sentences(text: str) -> List[str]:
    text = text.replace('\n', ' ')
    sentences, current = [], []
    paren_level = 0
    parts = re.split(r'(\.|\(|\))', text)
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == '(':
            paren_level += 1
            if current:
                current[-1] += part
            else:
                current.append(part)
        elif part == ')':
            paren_level = max(paren_level - 1, 0)
            current.append(part)
        elif part == '.' and paren_level == 0:
            current.append(part)
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
        else:
            current.append(part)
        i += 1
    tail = ''.join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences

# -------- Helpers --------
def chunk_spans(n_tokens: int, size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + size, n_tokens)) for i in range(0, n_tokens, size)]

# -------- Main --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/text_to_single_csv.py data_clean/yourfile.txt")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"[x] File not found: {in_path}")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    all_tokens = encode_txt(text)
    total_tokens = len(all_tokens)

    rows = []

    # Chunk loop
    for c_idx, (c_start, c_end) in enumerate(chunk_spans(total_tokens, CHUNK_TOKEN_SIZE), start=1):
        chunk_id = f"chunk_{c_idx:0{CHUNK_PAD}d}"
        chunk_tokens = all_tokens[c_start:c_end]
        chunk_text = decode_tokens(chunk_tokens)

        # Sub-chunk loop
        sub_spans = chunk_spans(len(chunk_tokens), SUBCHUNK_TOKEN_SIZE)
        sub_ids = []
        sub_texts = []
        for s_idx, (sc_start, sc_end) in enumerate(sub_spans, start=1):
            sub_id = f"sc_{c_idx:0{CHUNK_PAD}d}_{s_idx:0{SUBCHUNK_PAD}d}"
            sub_tokens = chunk_tokens[sc_start:sc_end]
            sub_text = decode_tokens(sub_tokens)
            sub_ids.append(sub_id)
            sub_texts.append(sub_text)

        # Sentence loop (within chunk)
        sentences = split_into_sentences(chunk_text)
        token_ptr_in_chunk = 0
        for sent_idx, sent_text in enumerate(sentences, start=1):
            sent_tokens = encode_txt(sent_text)
            sub_idx0 = token_ptr_in_chunk // SUBCHUNK_TOKEN_SIZE
            if sub_ids:
                sub_idx0 = max(0, min(len(sub_ids)-1, sub_idx0))
                subchunk_id = sub_ids[sub_idx0]
                subchunk_text = sub_texts[sub_idx0]
                subchunk_ord = sub_idx0 + 1
            else:
                subchunk_id = ""
                subchunk_text = ""
                subchunk_ord = 0

            sentence_id = (f"s_{c_idx:0{CHUNK_PAD}d}_{subchunk_ord:0{SUBCHUNK_PAD}d}_{sent_idx:0{SENTENCE_PAD}d}"
                           if subchunk_ord > 0 else
                           f"s_{c_idx:0{CHUNK_PAD}d}_000_{sent_idx:0{SENTENCE_PAD}d}")

            rows.append([
                chunk_id, chunk_text,
                subchunk_id, subchunk_text,
                sentence_id, sent_text
            ])
            token_ptr_in_chunk += len(sent_tokens)

    # Write single CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "chunk_text", "subchunk_id", "subchunk_text", "sentence_id", "sentence_text"])
        w.writerows(rows)

    print(f"[✓] Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
