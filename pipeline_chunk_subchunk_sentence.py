# -*- coding: utf-8 -*-
# Run:  python pipeline_chunk_subchunk_sentence.py  "pali chunk.txt"
# Output CSVs: outputs/chunks.csv, outputs/subchunks.csv, outputs/sentences.csv, outputs/windows.csv

import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# pip install tiktoken pandas
import tiktoken

# ================== CONFIG ==================
TOKENIZER_NAME = "cl100k_base"   # OpenAI GPT-4/3.5 compatible tokenizer
MAIN_CHUNK_TOKENS = 8000
SUBCHUNK_TOKENS = 200
OUTPUT_DIR = Path("outputs")
# ============================================

def split_into_sentences(text: str) -> List[str]:
    text = text.replace('\n', ' ')
    sentences = []
    current = []
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

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_text_file(path: Path) -> str:
    # Normalize to UTF-8 and normalize newlines
    data = path.read_text(encoding="utf-8", errors="replace")
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    return data

def chunk_tokens(tokens: List[int], chunk_size: int) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx_exclusive) token spans of size chunk_size
    """
    spans = []
    n = len(tokens)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        spans.append((start, end))
    return spans

def token_count(enc, text: str) -> int:
    return len(enc.encode(text))

def sentence_char_spans_in_chunk(chunk_text: str, sentences: List[str]) -> List[Tuple[int, int]]:
    """
    Compute (char_start, char_end_exclusive) spans of each sentence within chunk_text
    by sequential accumulation (robust to repeated substrings).
    """
    spans = []
    pos = 0
    for s in sentences:
        s = s.strip()
        if not s:
            spans.append((pos, pos))
            continue
        # Find next occurrence from current pos
        idx = chunk_text.find(s, pos)
        if idx == -1:
            # Fallback: assume contiguous; use length from current pos
            idx = pos
        start = idx
        end = start + len(s)
        spans.append((start, end))
        pos = end
    return spans

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_chunk_subchunk_sentence.py \"pali chunk.txt\"")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    assert input_path.exists(), f"Input file not found: {input_path}"

    ensure_dirs()
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    # Read & tokenize whole corpus (single file)
    full_text = read_text_file(input_path)
    all_tokens = enc.encode(full_text)

    # 1) MAIN CHUNKS (8k tokens)
    chunk_spans = chunk_tokens(all_tokens, MAIN_CHUNK_TOKENS)

    chunk_rows = []
    chunk_texts: List[str] = []
    for idx, (start, end) in enumerate(chunk_spans, start=1):
        chunk_tokens_list = all_tokens[start:end]
        chunk_text = enc.decode(chunk_tokens_list)
        chunk_texts.append(chunk_text)
        chunk_rows.append({
            "chunk_id": f"chunk_{idx:06d}",
            "order_idx": idx,
            "token_start": start,
            "token_end": end,  # exclusive
            "pali_text": chunk_text
        })

    pd.DataFrame(chunk_rows).to_csv(OUTPUT_DIR / "chunks.csv", index=False, encoding="utf-8-sig")

    # 2) SUBCHUNKS (200 tokens) per chunk
    sub_rows = []
    sub_id_counter = 1
    subchunk_texts_per_chunk: List[List[str]] = []

    for cidx, (start, end) in enumerate(chunk_spans, start=1):
        chunk_id = f"chunk_{cidx:06d}"
        rel_tokens = all_tokens[start:end]
        rel_spans = chunk_tokens(rel_tokens, SUBCHUNK_TOKENS)

        sub_texts = []
        for oidx, (rstart, rend) in enumerate(rel_spans, start=1):
            # absolute spans
            abs_start = start + rstart
            abs_end = start + rend
            sub_tokens = all_tokens[abs_start:abs_end]
            sub_text = enc.decode(sub_tokens)
            sub_texts.append(sub_text)
            sub_rows.append({
                "subchunk_id": f"sc_{cidx:06d}_{oidx:03d}",
                "chunk_id": chunk_id,
                "order_idx": oidx,
                "abs_token_start": abs_start,
                "abs_token_end": abs_end,  # exclusive
                "rel_token_start": rstart,
                "rel_token_end": rend,     # exclusive
                "pali_text": sub_text
            })
        subchunk_texts_per_chunk.append(sub_texts)

    pd.DataFrame(sub_rows).to_csv(OUTPUT_DIR / "subchunks.csv", index=False, encoding="utf-8-sig")

    # 3) SENTENCES per chunk (using your splitter), with char spans
    sent_rows = []
    sentence_id_counter = 1
    # To optionally attach subchunk_id: map by running token pointer (approx)
    for cidx, chunk_text in enumerate(chunk_texts, start=1):
        chunk_id = f"chunk_{cidx:06d}"

        # sentences
        sentences = split_into_sentences(chunk_text)
        char_spans = sentence_char_spans_in_chunk(chunk_text, sentences)

        # For approximate subchunk mapping, count tokens cumulatively within the chunk
        # Using tokenizer over each sentence in order
        token_pointer = 0
        # Number of subchunks in this chunk
        n_sub = len(subchunk_texts_per_chunk[cidx-1])

        for oidx, (sent_text, (cs, ce)) in enumerate(zip(sentences, char_spans), start=1):
            sent_tokens = enc.encode(sent_text)
            # subchunk index guess
            sub_index_zero_based = token_pointer // SUBCHUNK_TOKENS
            token_pointer += len(sent_tokens)

            subchunk_id = None
            if n_sub > 0:
                # clamp to [0, n_sub-1]
                sub_index_zero_based = max(0, min(n_sub-1, sub_index_zero_based))
                subchunk_id = f"sc_{cidx:06d}_{sub_index_zero_based+1:03d}"

            sent_rows.append({
                "sentence_id": f"s_{cidx:06d}_{oidx:03d}",
                "chunk_id": chunk_id,
                "subchunk_id": subchunk_id,     # optional
                "order_idx": oidx,
                "char_start": cs,
                "char_end": ce,                 # exclusive
                "pali_text": sent_text
            })
            sentence_id_counter += 1

    pd.DataFrame(sent_rows).to_csv(OUTPUT_DIR / "sentences.csv", index=False, encoding="utf-8-sig")

    # 4) WINDOWS (size 2 & 3) per chunk
    windows = []
    window_id_counter = 1

    # Group sentences by chunk_id
    df_sent = pd.DataFrame(sent_rows)
    df_sent['order_idx'] = df_sent['order_idx'].astype(int)

    for cidx in range(1, len(chunk_spans)+1):
        chunk_id = f"chunk_{cidx:06d}"
        sents = df_sent[df_sent['chunk_id'] == chunk_id].sort_values('order_idx')
        sents_list = list(sents.to_dict('records'))

        # rolling size=2
        for i in range(len(sents_list)-1):
            left = sents_list[i]
            right = sents_list[i+1]
            text = (left["pali_text"] + " " + right["pali_text"]).strip()
            windows.append({
                "window_id": f"w2_{cidx:06d}_{i+1:03d}",
                "chunk_id": chunk_id,
                "size": 2,
                "left_sentence_id": left["sentence_id"],
                "right_sentence_id": right["sentence_id"],
                "order_idx": i+1,
                "text": text
            })
        # rolling size=3
        for i in range(len(sents_list)-2):
            a = sents_list[i]
            b = sents_list[i+1]
            c = sents_list[i+2]
            text = (a["pali_text"] + " " + b["pali_text"] + " " + c["pali_text"]).strip()
            windows.append({
                "window_id": f"w3_{cidx:06d}_{i+1:03d}",
                "chunk_id": chunk_id,
                "size": 3,
                "left_sentence_id": a["sentence_id"],
                "right_sentence_id": c["sentence_id"],
                "order_idx": i+1,
                "text": text
            })

    pd.DataFrame(windows).to_csv(OUTPUT_DIR / "windows.csv", index=False, encoding="utf-8-sig")

    print("✅ Done.")
    print(f"- {OUTPUT_DIR/'chunks.csv'}")
    print(f"- {OUTPUT_DIR/'subchunks.csv'}")
    print(f"- {OUTPUT_DIR/'sentences.csv'}")
    print(f"- {OUTPUT_DIR/'windows.csv'}")

if __name__ == "__main__":
    main()
