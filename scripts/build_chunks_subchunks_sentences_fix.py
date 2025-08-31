# -*- coding: utf-8 -*-
"""
build_chunks_subchunks_sentences_fix.py
--------------------------------------
- Accurate token counts with (start, end_exclusive) slicing
- Sentence splitter (strict):
    • Sentence ends with one of . ? ! followed by a space
    • Next sentence starts with a capital letter (Unicode .isupper())
    • Ignore .?! inside (), [], {}
- Outputs 3 CSVs:
    1) chunks.csv      -> chunk_id, file_name, token_start, token_end_excl, token_count, pali_text
    2) subchunks.csv   -> subchunk_id, chunk_id, order_idx, token_start, token_end_excl, token_count, pali_text
    3) sentences.csv   -> sentence_id, chunk_id, subchunk_id, order_idx, char_start, char_end_incl, pali_text

Usage examples:
  # Single file, whitespace tokenizer (word-like 8k/200 counts)
  python scripts/build_chunks_subchunks_sentences_fix.py --only "data_clean/pali chunk.txt" --tokenizer whitespace

  # Process all .txt under data_clean with tiktoken
  python scripts/build_chunks_subchunks_sentences_fix.py --tokenizer tiktoken
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# =========================
# CONFIG (defaults)
# =========================
CHUNK_TOKEN_SIZE    = 8000   # ~8k tokens per chunk
SUBCHUNK_TOKEN_SIZE = 200    # ~200 tokens per sub-chunk

# Zero-padding rules
CHUNK_PAD    = 6   # chunk_000001
SUBCHUNK_PAD = 3   # sc_000001_001
SENTENCE_PAD = 3   # s_000001_001_001

# Project structure (can be overridden by CLI)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN   = PROJECT_ROOT / "data_clean"

# =========================
# Tokenizer toggle
# =========================
USE_TIKTOKEN = False  # default = whitespace
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

def encode_tokens(text: str):
    """Return tokens list (tiktoken ids or whitespace tokens)."""
    if USE_TIKTOKEN and _enc:
        return _enc.encode(text)
    return text.split()

def decode_tokens(tok):
    """Return text from tokens list (tiktoken ids or whitespace tokens)."""
    if USE_TIKTOKEN and _enc:
        return _enc.decode(tok)
    return " ".join(tok)

# =========================
# Sentence split (strict)
# - Enders: . ? ! + space
# - Next sentence must start with capital
# - Ignore inside (), [], {}
# =========================
def split_into_sentences(text: str) -> List[str]:
    text = text.replace('\n', ' ')
    openers, closers = "([{", ")]}"
    stack = []
    out: List[str] = []
    start = 0
    i, n = 0, len(text)

    while i < n:
        ch = text[i]
        if ch in openers:
            stack.append(ch); i += 1; continue
        if ch in closers and stack:
            stack.pop(); i += 1; continue
        if ch in ".?!":
            if not stack and i + 1 < n and text[i+1] == " ":
                j = i + 2
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j].isupper():
                    sent = text[start:i+1].strip()
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

# =========================
# Helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_txt_files(folder: Path) -> List[Path]:
    return sorted([x for x in folder.glob("*.txt") if x.is_file()])

def slice_ranges(n: int, size: int) -> List[Tuple[int, int]]:
    """Return (start, end_exclusive) ranges across n tokens."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]

def build_char_offsets_from_tokens(tokens: List[str]) -> List[int]:
    """
    For char anchoring inside a whitespace-joined text:
      joined = " ".join(tokens)
      offsets[i] is the starting char index of tokens[i].
    """
    offsets = []
    pos = 0
    for i, tk in enumerate(tokens):
        offsets.append(pos)
        pos += len(tk)
        if i < len(tokens) - 1:
            pos += 1  # the single space between tokens
    return offsets

def find_subchunk_id_by_char(char_pos: int, subchunks_char_ranges: List[Tuple[str, int, int]]) -> Optional[str]:
    for sid, cstart, cend in subchunks_char_ranges:
        if cstart <= char_pos <= cend:
            return sid
    return None

# ---- safe open (avoid PermissionError if file is locked) ----
def _safe_open(path: Path, mode: str):
    try:
        return open(path, mode, encoding="utf-8", newline="")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_stem(path.stem + f"_{ts}")
        print(f"[!] Locked: {path.name} → writing to {alt.name} instead")
        return open(alt, mode, encoding="utf-8", newline="")

# =========================
# CSV Writers
# =========================
def open_csv_writers(outdir: Path):
    ensure_dir(outdir)
    chunks_fp     = _safe_open(outdir / "chunks.csv", "w")
    subchunks_fp  = _safe_open(outdir / "subchunks.csv", "w")
    sentences_fp  = _safe_open(outdir / "sentences.csv", "w")

    chunks_w = csv.writer(chunks_fp)
    sub_w    = csv.writer(subchunks_fp)
    sent_w   = csv.writer(sentences_fp)

    # headers (token_end is exclusive; token_count present for verification)
    chunks_w.writerow(["chunk_id", "file_name", "token_start", "token_end_excl", "token_count", "pali_text"])
    sub_w.writerow(["subchunk_id", "chunk_id", "order_idx", "token_start", "token_end_excl", "token_count", "pali_text"])
    sent_w.writerow(["sentence_id", "chunk_id", "subchunk_id", "order_idx", "char_start", "char_end_incl", "pali_text"])

    return (chunks_fp, subchunks_fp, sentences_fp), (chunks_w, sub_w, sent_w)

# =========================
# Core processing
# =========================
def process_file(file_path: Path, chunk_base_index: int, outdir: Path,
                 chunks_w, sub_w, sent_w) -> int:
    text_raw = file_path.read_text(encoding="utf-8", errors="replace").replace("\r\n","\n").replace("\r","\n")

    # 0) tokenize full text
    tokens_all = encode_tokens(text_raw)
    N = len(tokens_all)

    # 1) chunk ranges (start, end_exclusive)
    chunk_ranges = slice_ranges(N, CHUNK_TOKEN_SIZE)

    # for summary
    total_tokens_this_file = 0

    next_chunk_idx = chunk_base_index
    for (g_start, g_end) in chunk_ranges:
        chunk_idx = next_chunk_idx
        chunk_id = f"chunk_{chunk_idx:0{CHUNK_PAD}d}"

        # chunk token/text
        chunk_tokens = tokens_all[g_start:g_end]
        token_count = g_end - g_start
        total_tokens_this_file += token_count
        chunk_text = decode_tokens(chunk_tokens)

        # write chunk
        chunks_w.writerow([chunk_id, file_path.name, g_start, g_end, token_count, chunk_text])

        # for subchunk char anchoring, we need word-like tokens to compute offsets on the *visible* text
        if USE_TIKTOKEN and _enc:
            chunk_words = chunk_text.split()
        else:
            # whitespace tokenizer already word-like
            chunk_words = chunk_tokens

        char_offsets = build_char_offsets_from_tokens(chunk_words)

        # 2) subchunk local ranges relative to chunk_tokens
        sub_local = slice_ranges(len(chunk_tokens), SUBCHUNK_TOKEN_SIZE)

        subchunks_char_ranges: List[Tuple[str, int, int]] = []
        for order_idx, (sc_start, sc_end) in enumerate(sub_local, start=1):
            subchunk_id = f"sc_{chunk_idx:0{CHUNK_PAD}d}_{order_idx:0{SUBCHUNK_PAD}d}"
            sc_count = sc_end - sc_start

            # subchunk visible text
            if USE_TIKTOKEN and _enc:
                sub_text = " ".join(chunk_words[sc_start:sc_end])
            else:
                sub_text = " ".join(chunk_tokens[sc_start:sc_end])

            # char_start / char_end_incl inside chunk_text
            if sc_start < len(char_offsets):
                cstart = char_offsets[sc_start]
            else:
                cstart = len(chunk_text)

            if sc_end == 0:
                cend = -1
            elif sc_end < len(char_offsets):
                last_tok_start = char_offsets[sc_end - 1]
                last_tok_len = len(chunk_words[sc_end - 1])
                cend = last_tok_start + last_tok_len - 1
            else:
                cend = len(chunk_text) - 1 if len(chunk_text) else -1

            subchunks_char_ranges.append((subchunk_id, cstart, cend))

            # global token indices for subchunk start/end_excl
            sc_global_start = g_start + sc_start
            sc_global_end_excl = g_start + sc_end

            sub_w.writerow([subchunk_id, chunk_id, order_idx, sc_global_start, sc_global_end_excl, sc_count, sub_text])

        # 3) sentences within this chunk (strict splitter)
        sentences = split_into_sentences(chunk_text)

        # sentence char positions
        cursor = 0
        sent_positions: List[Tuple[int,int,str]] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            st = chunk_text.find(s, cursor)
            if st == -1:
                st = chunk_text.find(s)
            en = st + len(s) - 1  # inclusive
            sent_positions.append((st, en, s))
            cursor = en + 1

        for s_order, (cstart, cend, s_text) in enumerate(sent_positions, start=1):
            s_subchunk_id = find_subchunk_id_by_char(cstart, subchunks_char_ranges)
            if s_subchunk_id:
                sub_order = int(s_subchunk_id.split("_")[-1])
                sentence_id = f"s_{chunk_idx:0{CHUNK_PAD}d}_{sub_order:0{SUBCHUNK_PAD}d}_{s_order:0{SENTENCE_PAD}d}"
            else:
                sentence_id = f"s_{chunk_idx:0{CHUNK_PAD}d}_000_{s_order:0{SENTENCE_PAD}d}"

            sent_w.writerow([
                sentence_id,        # sentence_id
                chunk_id,           # chunk_id
                s_subchunk_id if s_subchunk_id else "",  # subchunk_id
                s_order,            # order_idx (within chunk)
                cstart,             # char_start
                cend,               # char_end_incl
                s_text              # pali_text (sentence)
            ])

        next_chunk_idx += 1

    return next_chunk_idx, len(chunk_ranges), total_tokens_this_file

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build chunks/subchunks/sentences with accurate token counts.")
    parser.add_argument("--only", help="Process only this txt file path (default: process all under data_clean)")
    parser.add_argument("--tokenizer", choices=["whitespace", "tiktoken"], default="whitespace",
                        help="Tokenizer to use (default: whitespace)")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "outputs"),
                        help="Output directory (default: ./outputs)")
    args = parser.parse_args()

    global USE_TIKTOKEN
    USE_TIKTOKEN = (args.tokenizer == "tiktoken")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.only:
        files = [Path(args.only)]
    else:
        files = list_txt_files(DATA_CLEAN)

    if not files:
        print(f"[!] No .txt files found in {DATA_CLEAN} (or --only path not found).")
        return

    (chunks_fp, subchunks_fp, sentences_fp), (chunks_w, sub_w, sent_w) = open_csv_writers(outdir)

    try:
        next_chunk_index = 1
        grand_chunks = 0
        grand_tokens = 0

        for f in files:
            if not f.exists():
                print(f"[!] Skip missing: {f}")
                continue
            print(f"[+] Processing {f.name} ...")
            next_chunk_index, n_chunks, n_tokens = process_file(
                f, chunk_base_index=next_chunk_index, outdir=outdir,
                chunks_w=chunks_w, sub_w=sub_w, sent_w=sent_w
            )
            grand_chunks += n_chunks
            grand_tokens += n_tokens

        print("[✓] Done. CSVs written to", outdir.as_posix())
        print(f"[i] Tokenizer: {'tiktoken cl100k_base' if USE_TIKTOKEN and _enc else 'whitespace'}")
        print(f"[summary] total_chunks={grand_chunks}, total_tokens={grand_tokens}")

    finally:
        chunks_fp.close()
        subchunks_fp.close()
        sentences_fp.close()

if __name__ == "__main__":
    main()
