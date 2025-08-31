# -*- coding: utf-8 -*-
"""
Make a compact 4-column CSV (ordered):
  chunk_id, subchunk_id, sentence_id, sentence_text

Input CSV is the v2 single-file (6 columns):
  chunk_id, chunk_text, subchunk_id, subchunk_text, sentence_id, sentence_text

Sort order:
  by chunk number -> subchunk number (from sentence_id) -> sentence number
Usage:
  python scripts/make_sentences_compact4_v2.py \
      --in "outputs_v3/chunk_subchunk_sentence_v2_fromcsv.csv" \
      --out "outputs_v3/sentences_compact4_v2.csv"
"""

import argparse
import re
import pandas as pd
from pathlib import Path

def chunk_no(cid: str) -> int:
    # e.g., "chunk_000012" -> 12
    m = re.search(r"(\d+)$", str(cid))
    return int(m.group(1)) if m else 0

def sub_no_from_sid(sid: str) -> int:
    # e.g., "s_000012_007_005" -> 7 (second last block)
    parts = str(sid).split("_")
    return int(parts[-2]) if len(parts) >= 3 and parts[-2].isdigit() else 0

def sent_no_from_sid(sid: str) -> int:
    # e.g., "s_000012_007_005" -> 5 (last block)
    parts = str(sid).split("_")
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0

def main():
    ap = argparse.ArgumentParser(description="Build compact 4-column sentences CSV (v2).")
    ap.add_argument("--in", dest="inp", required=True, help="Path to v2 6-col CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    required = ["chunk_id","chunk_text","subchunk_id","subchunk_text","sentence_id","sentence_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    # Natural ordering: chunk -> subchunk -> sentence
    df_sorted = (
        df.assign(
            _c = df["chunk_id"].apply(chunk_no),
            _sc = df["sentence_id"].apply(sub_no_from_sid),
            _s = df["sentence_id"].apply(sent_no_from_sid),
        )
        .sort_values(by=["_c","_sc","_s"], kind="stable")
        .drop(columns=["_c","_sc","_s"])
        .reset_index(drop=True)
    )

    compact = df_sorted[["chunk_id","subchunk_id","sentence_id","sentence_text"]]
    compact.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"[✓] Done: {outp}  (rows={len(compact)})")

if __name__ == "__main__":
    main()
