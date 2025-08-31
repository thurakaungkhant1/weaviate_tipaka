# -*- coding: utf-8 -*-
# Run:
#   python make_wide_table.py "pali chunk.txt"
# Output:
#   outputs/wide_chunks.csv  (chunk_id, chunk_text, subchunk_id, subchunk_text, sentence_id, sentence_text)

import sys, re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import tiktoken

# ---------------- CONFIG ----------------
TOKENIZER_NAME = "cl100k_base"    # OpenAI GPT-3.5/4 compatible tokenizer
MAIN_CHUNK_TOKENS = 8000
SUBCHUNK_TOKENS = 200
OUTPUT_DIR = Path("outputs")
OUT_FILE = OUTPUT_DIR / "wide_chunks.csv"
# ----------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """Your splitter: split by '.' except inside parentheses."""
    text = text.replace('\n', ' ')
    sentences, current = [], []
    paren_level = 0
    parts = re.split(r'(\.|\(|\))', text)
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == '(':
            paren_level += 1
            if current: current[-1] += part
            else: current.append(part)
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

def chunk_token_spans(n_tokens: int, size: int) -> List[Tuple[int, int]]:
    spans = []
    for start in range(0, n_tokens, size):
        end = min(start + size, n_tokens)
        spans.append((start, end))  # end is exclusive
    return spans

def main():
    if len(sys.argv) < 2:
        print('Usage: python make_wide_table.py "pali chunk.txt"')
        sys.exit(1)

    INPUT = Path(sys.argv[1])
    assert INPUT.exists(), f"Input not found: {INPUT}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    # Read file (UTF-8)
    full_text = INPUT.read_text(encoding="utf-8", errors="replace")
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")

    # Tokenize whole corpus
    all_tokens = enc.encode(full_text)
    total_tokens = len(all_tokens)

    # 1) 8k-token chunks (absolute token spans on full corpus)
    chunk_spans = chunk_token_spans(total_tokens, MAIN_CHUNK_TOKENS)

    rows = []  # for wide table
    for c_idx, (c_abs_start, c_abs_end) in enumerate(chunk_spans, start=1):
        chunk_id = f"chunk_{c_idx:06d}"
        chunk_tokens = all_tokens[c_abs_start:c_abs_end]
        chunk_text = enc.decode(chunk_tokens)

        # 2) 200-token sub-chunks (relative spans within this chunk)
        rel_spans = chunk_token_spans(len(chunk_tokens), SUBCHUNK_TOKENS)
        sub_ids = []
        sub_texts = []
        for s_idx, (r_start, r_end) in enumerate(rel_spans, start=1):
            subchunk_id = f"sc_{c_idx:06d}_{s_idx:03d}"
            abs_start = c_abs_start + r_start
            abs_end = c_abs_start + r_end
            sub_tokens = all_tokens[abs_start:abs_end]
            sub_text = enc.decode(sub_tokens)
            sub_ids.append(subchunk_id)
            sub_texts.append(sub_text)

        # 3) Sentences inside this chunk (your splitter)
        sentences = split_into_sentences(chunk_text)

        # Running token pointer from start of chunk to map sentences → sub-chunks
        token_pointer_in_chunk = 0
        for o_idx, sent_text in enumerate(sentences, start=1):
            sent_tokens = enc.encode(sent_text)
            # sentence absolute token start offset in corpus
            sent_abs_token_start = c_abs_start + token_pointer_in_chunk

            # sub-chunk index by start token (0-based)
            sub_idx0 = token_pointer_in_chunk // SUBCHUNK_TOKENS
            # clamp
            if sub_idx0 < 0: sub_idx0 = 0
            if sub_idx0 >= len(sub_ids): sub_idx0 = len(sub_ids) - 1 if sub_ids else 0

            # lookup subchunk id/text
            subchunk_id = sub_ids[sub_idx0] if sub_ids else None
            subchunk_text = sub_texts[sub_idx0] if sub_texts else ""

            sentence_id = f"s_{c_idx:06d}_{o_idx:03d}"

            rows.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "subchunk_id": subchunk_id,
                "subchunk_text": subchunk_text,
                "sentence_id": sentence_id,
                "sentence_text": sent_text
            })

            token_pointer_in_chunk += len(sent_tokens)

    df = pd.DataFrame(rows, columns=[
        "chunk_id", "chunk_text",
        "subchunk_id", "subchunk_text",
        "sentence_id", "sentence_text"
    ])
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Done: {OUT_FILE}")

if __name__ == "__main__":
    main()
