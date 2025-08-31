# -*- coding: utf-8 -*-
"""
Build 8k-token chunks, 200-token sub-chunks, and sentence splits
Outputs: outputs/chunks.csv, outputs/subchunks.csv, outputs/sentences.csv

Folder layout (already set by you):
  tipitaka-project/
    data_clean/   -> cleaned .txt files (input)
    outputs/      -> CSV files will be written here
    scripts/      -> put this script here

Run:
  python scripts/build_chunks_subchunks_sentences.py
"""

import os
import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN   = PROJECT_ROOT / "data_clean"
OUTPUTS      = PROJECT_ROOT / "outputs"

CHUNK_TOKEN_SIZE    = 8000   # ~8k tokens per chunk
SUBCHUNK_TOKEN_SIZE = 200    # ~200 tokens per sub-chunk

# Zero-padding rules
CHUNK_PAD    = 6   # chunk_000001
SUBCHUNK_PAD = 3   # sc_000001_001
SENTENCE_PAD = 3   # s_000001_001_001

# =========================
# Sentence split (your method)
# =========================
def split_into_sentences(text: str) -> List[str]:
    """
    Your custom splitter:
    - keeps parentheses pairs together
    - ends sentence on '.' only if not inside parentheses
    """
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

    # tail (no trailing '.')
    tail = ''.join(current).strip()
    if tail:
        sentences.append(tail)

    return sentences

# =========================
# Helpers
# =========================
def ensure_dirs():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

def list_txt_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.txt") if p.is_file()])

def tokenize_whitespace(text: str) -> List[str]:
    # simple, deterministic tokenizer (keeps punctuation attached)
    return text.split()

def slice_ranges(n: int, size: int) -> List[Tuple[int, int]]:
    """
    Return list of (start, end_inclusive) token index ranges
    for slicing a list of length n into windows of length 'size'.
    """
    ranges = []
    start = 0
    while start < n:
        end = min(start + size, n) - 1
        ranges.append((start, end))
        start += size
    return ranges

def join_tokens(tokens: List[str]) -> str:
    # mirror the way we counted tokens: tokenized by whitespace, join with single space
    return " ".join(tokens).strip()

def build_token_char_offsets(tokens: List[str]) -> List[int]:
    """
    Return list of char_start offsets for each token after joining with single spaces.
    Example:
      tokens = ["abc", "de"]
      joined = "abc de"
      -> offsets = [0, 4]
    """
    offsets = []
    pos = 0
    for i, tk in enumerate(tokens):
        offsets.append(pos)
        pos += len(tk)
        if i < len(tokens) - 1:
            pos += 1  # the single space between tokens
    return offsets

def find_subchunk_id_by_char(
    char_pos: int,
    subchunks_char_ranges: List[Tuple[str, int, int]]
) -> Optional[str]:
    """
    subchunks_char_ranges: list of (subchunk_id, char_start, char_end_inclusive) within the chunk_text
    Return the subchunk_id whose char range contains char_pos (first match).
    """
    for sid, cstart, cend in subchunks_char_ranges:
        if cstart <= char_pos <= cend:
            return sid
    return None  # may span boundaries; we at least anchor to starting subchunk


# =========================
# CSV Writers
# =========================
def open_csv_writers():
    chunks_fp     = open(OUTPUTS / "chunks.csv",     "w", encoding="utf-8", newline="")
    subchunks_fp  = open(OUTPUTS / "subchunks.csv",  "w", encoding="utf-8", newline="")
    sentences_fp  = open(OUTPUTS / "sentences.csv",  "w", encoding="utf-8", newline="")

    chunks_w = csv.writer(chunks_fp)
    sub_w    = csv.writer(subchunks_fp)
    sent_w   = csv.writer(sentences_fp)

    # headers (as agreed)
    chunks_w.writerow(["chunk_id", "file_name", "token_start", "token_end", "pali_text"])
    sub_w.writerow(["subchunk_id", "chunk_id", "order_idx", "token_start", "token_end", "pali_text"])
    sent_w.writerow(["sentence_id", "chunk_id", "subchunk_id", "order_idx", "char_start", "char_end", "pali_text"])

    return (chunks_fp, subchunks_fp, sentences_fp), (chunks_w, sub_w, sent_w)

# =========================
# Main processing
# =========================
def process_file(file_path: Path, chunk_base_index: int,
                 chunks_w, sub_w, sent_w) -> int:
    """
    Returns: next_chunk_index to continue numbering across files
    """
    text = file_path.read_text(encoding="utf-8")
    tokens = tokenize_whitespace(text)
    total_tokens = len(tokens)

    chunk_ranges = slice_ranges(total_tokens, CHUNK_TOKEN_SIZE)

    # For ID continuity across multiple files:
    next_chunk_idx = chunk_base_index

    for c_idx_within_file, (g_start, g_end) in enumerate(chunk_ranges, start=1):
        # Build IDs
        chunk_idx = next_chunk_idx
        chunk_id = f"chunk_{chunk_idx:0{CHUNK_PAD}d}"

        # Extract chunk tokens and text
        chunk_tokens = tokens[g_start: g_end + 1]
        chunk_text   = join_tokens(chunk_tokens)

        # Write chunk row
        chunks_w.writerow([chunk_id, file_path.name, g_start, g_end, chunk_text])

        # ---- Subchunks (200 tokens) within this chunk ----
        sub_ranges_local = slice_ranges(len(chunk_tokens), SUBCHUNK_TOKEN_SIZE)

        # Precompute token->char offsets for the chunk (to later map subchunks to char ranges)
        chunk_token_char_offsets = build_token_char_offsets(chunk_tokens)
        # For each subchunk, compute char_start/end within the chunk_text
        subchunks_char_ranges: List[Tuple[str, int, int]] = []

        for sc_order, (sc_start_local, sc_end_local) in enumerate(sub_ranges_local, start=1):
            subchunk_id = f"sc_{chunk_idx:0{CHUNK_PAD}d}_{sc_order:0{SUBCHUNK_PAD}d}"

            # Token index in global file space
            sc_global_start = g_start + sc_start_local
            sc_global_end   = g_start + sc_end_local

            # Subchunk tokens/text
            sc_tokens = chunk_tokens[sc_start_local: sc_end_local + 1]
            sc_text   = join_tokens(sc_tokens)

            # Order index is within the chunk
            order_idx = sc_order

            # Char range inside chunk_text
            sc_char_start = chunk_token_char_offsets[sc_start_local]
            last_token_idx = sc_end_local
            sc_char_end = chunk_token_char_offsets[last_token_idx] + len(chunk_tokens[last_token_idx]) - 1
            # plus spaces between tokens inside this subchunk:
            if sc_end_local > sc_start_local:
                # add number of spaces between tokens in this subchunk
                space_count = (sc_end_local - sc_start_local)
                sc_char_end += space_count

            # store for sentence->subchunk mapping
            subchunks_char_ranges.append((subchunk_id, sc_char_start, sc_char_end))

            sub_w.writerow([subchunk_id, chunk_id, order_idx, sc_global_start, sc_global_end, sc_text])

        # ---- Sentences within this chunk (using your splitter) ----
        sentences = split_into_sentences(chunk_text)
        # build char offsets for sentences by scanning chunk_text once
        sent_char_positions: List[Tuple[int, int, str]] = []
        cursor = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # find s starting from current cursor to keep order robust
            start = chunk_text.find(s, cursor)
            if start == -1:
                # fallback search from 0 if something odd (should be rare)
                start = chunk_text.find(s)
            end = start + len(s) - 1
            sent_char_positions.append((start, end, s))
            cursor = end + 1

        for s_order, (cstart, cend, s_text) in enumerate(sent_char_positions, start=1):
            # anchor sentence to subchunk where it STARTS (may span across)
            s_subchunk_id = find_subchunk_id_by_char(cstart, subchunks_char_ranges)

            sentence_id = f"s_{chunk_idx:0{CHUNK_PAD}d}_{int(s_subchunk_id.split('_')[-1]) if s_subchunk_id else 1:0{SUBCHUNK_PAD}d}_{s_order:0{SENTENCE_PAD}d}" \
                          if s_subchunk_id else f"s_{chunk_idx:0{CHUNK_PAD}d}_000_{s_order:0{SENTENCE_PAD}d}"

            sent_w.writerow([
                sentence_id,
                chunk_id,
                s_subchunk_id if s_subchunk_id else "",
                s_order,
                cstart,
                cend,
                s_text
            ])

        next_chunk_idx += 1

    return next_chunk_idx


def main():
    ensure_dirs()
    files = list_txt_files(DATA_CLEAN)
    if not files:
        print(f"[!] No .txt files found in {DATA_CLEAN}")
        return

    (chunks_fp, subchunks_fp, sentences_fp), (chunks_w, sub_w, sent_w) = open_csv_writers()

    try:
        next_chunk_index = 1  # global chunk sequence across all files
        for f in files:
            print(f"[+] Processing {f.name} ...")
            next_chunk_index = process_file(
                f, chunk_base_index=next_chunk_index,
                chunks_w=chunks_w, sub_w=sub_w, sent_w=sent_w
            )
        print("[✓] Done. CSVs written to 'outputs/'")
    finally:
        chunks_fp.close()
        subchunks_fp.close()
        sentences_fp.close()


if __name__ == "__main__":
    main()
