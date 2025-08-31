# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

CHUNKS = Path("outputs/chunks.csv")
SUBS   = Path("outputs/subchunks.csv")
SENTS  = Path("outputs/sentences.csv")

def ok(b, msg):
    print(("✅ " if b else "❌ ") + msg)
    return b

def main():
    dfc = pd.read_csv(CHUNKS)
    dfs = pd.read_csv(SUBS)
    dft = pd.read_csv(SENTS)

    # 1) chunk token_count = end_excl - start
    ok((dfc["token_count"] == (dfc["token_end_excl"] - dfc["token_start"])).all(),
       "chunks: token_count == end_excl - start")

    # 2) sum(subchunks token_count) per chunk == chunk token_count (last one may be <200 but sum must match)
    sub_sum = dfs.groupby("chunk_id")["token_count"].sum().rename("subs_sum")
    merged  = dfc.merge(sub_sum, on="chunk_id", how="left")
    ok((merged["token_count"] == merged["subs_sum"]).all(),
       "subchunks: sum(token_count) per chunk == chunk token_count")

    # 3) sentences anchored into some subchunk (allow empty for cross-gap edge, but expect >99% anchored)
    anchored = dft["subchunk_id"].ne("").mean()
    print(f"📌 sentences anchored ratio: {anchored:.4f}")
    ok(anchored > 0.99, "≥99% sentences have a subchunk_id")

    # 4) sentence char range valid (start <= end, within chunk text length)
    #    load chunk text length map
    chunk_len = dfc.set_index("chunk_id")["pali_text"].str.len().to_dict()
    dft["len_ok"] = dft.apply(lambda r: (r["char_start"] <= r["char_end_incl"]) and 
                                         (0 <= r["char_start"] <= chunk_len.get(r["chunk_id"], 10**12)) and
                                         (0 <= r["char_end_incl"] < chunk_len.get(r["chunk_id"], 10**12)), axis=1)
    ok(dft["len_ok"].all(), "sentence char positions are within chunk_text bounds")

if __name__ == "__main__":
    main()
