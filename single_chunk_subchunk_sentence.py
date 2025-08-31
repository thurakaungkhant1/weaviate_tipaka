# -*- coding: utf-8 -*-
# Usage:
#   python single_chunk_subchunk_sentence.py "pali chunk.txt"
# Output:
#   outputs/chunk_subchunk_sentence.csv
#
# Columns:
#   chunk_id, chunk_text, subchunk_id, subchunk_text, sentence_id, sentence_text

import sys, re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import tiktoken

# -------- CONFIG --------
TOKENIZER_NAME = "cl100k_base"   # GPT-3.5/4 compatible
MAIN_CHUNK_TOKENS = 8000
SUBCHUNK_TOKENS = 200
OUTPUT_DIR = Path("outputs")
OUT_FILE = OUTPUT_DIR / "chunk_subchunk_sentence.csv"
# ------------------------

def split_into_sentences(text: str) -> List[str]:
    """User-provided splitter: split by '.' except inside parentheses."""
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

def chunk_spans(n_tokens: int, size: int) -> List[Tuple[int, int]]:
    """Return [(start, end_exclusive), ...] spans of length `size` (last may be shorter)."""
    return [(i, min(i+size, n_tokens)) for i in range(0, n_tokens, size)]

def main():
    if len(sys.argv) < 2:
        print('Usage: python single_chunk_subchunk_sentence.py "pali chunk.txt"')
        sys.exit(1)

    inp = Path(sys.argv[1])
    assert inp.exists(), f"Input not found: {inp}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    # Read full text (normalize newlines)
    full_text = inp.read_text(encoding="utf-8", errors="replace").replace("\r\n","\n").replace("\r","\n")
    all_tokens = enc.encode(full_text)
    total_tokens = len(all_tokens)

    # 1) 8k-token CHUNKS on full corpus
    c_spans = chunk_spans(total_tokens, MAIN_CHUNK_TOKENS)

    rows = []
    for c_idx, (c_abs_start, c_abs_end) in enumerate(c_spans, start=1):
        chunk_id = f"chunk_{c_idx:06d}"
        chunk_tokens = all_tokens[c_abs_start:c_abs_end]
        chunk_text = enc.decode(chunk_tokens)

        # 2) 200-token SUBCHUNKS within this chunk (relative → absolute)
        rel_spans = chunk_spans(len(chunk_tokens), SUBCHUNK_TOKENS)
        sub_ids = []
        sub_texts = []
        for s_idx, (r_start, r_end) in enumerate(rel_spans, start=1):
            sub_id = f"sc_{c_idx:06d}_{s_idx:03d}"
            abs_start = c_abs_start + r_start
            abs_end = c_abs_start + r_end
            sub_text = enc.decode(all_tokens[abs_start:abs_end])
            sub_ids.append(sub_id)
            sub_texts.append(sub_text)

        # 3) SENTENCES (use user's splitter)
        sentences = split_into_sentences(chunk_text)

        # Map sentences → subchunks by sentence **start token** (within chunk)
        token_ptr_in_chunk = 0
        for o_idx, sent_text in enumerate(sentences, start=1):
            sent_tokens = enc.encode(sent_text)

            # Which subchunk does this sentence START in?
            sub_idx0 = token_ptr_in_chunk // SUBCHUNK_TOKENS
            if sub_ids:
                sub_idx0 = max(0, min(len(sub_ids)-1, sub_idx0))
                subchunk_id = sub_ids[sub_idx0]
                subchunk_text = sub_texts[sub_idx0]
            else:
                subchunk_id = None
                subchunk_text = ""

            sentence_id = f"s_{c_idx:06d}_{o_idx:03d}"
            rows.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "subchunk_id": subchunk_id,
                "subchunk_text": subchunk_text,
                "sentence_id": sentence_id,
                "sentence_text": sent_text
            })

            token_ptr_in_chunk += len(sent_tokens)

    df = pd.DataFrame(rows, columns=[
        "chunk_id", "chunk_text",
        "subchunk_id", "subchunk_text",
        "sentence_id", "sentence_text"
    ])
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Done: {OUT_FILE}")

if __name__ == "__main__":
    main()
