# search.py
import argparse
import sys
from typing import List, Dict, Any, Tuple
import weaviate
from weaviate.classes.query import Filter

# ---- Config ----
CHUNK_COLL = "PaliChunk"
SUB_COLL   = "PaliSubChunk"
SENT_COLL  = "PaliSentence"

DEFAULT_LIMIT = 10

# ---- Helpers ----
def connect():
    return weaviate.connect_to_local()

def ptrim(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + " …"

def split_terms(q: str) -> List[str]:
    # very simple splitter: space-separated tokens, drop empties
    return [t for t in (q or "").strip().split() if t]

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_obj_list(objs: List[Any], props: List[str]):
    if not objs:
        print("∅ No results.")
        return
    for i, o in enumerate(objs, 1):
        p = o.properties or {}
        fields = " | ".join(f"{k}={ptrim(str(p.get(k,'')))}" for k in props)
        print(f"{i:>2}. {fields}")

# ---- Query primitives ----
def near_text(coll, query: str, limit: int, return_props: List[str]):
    return coll.query.near_text(
        query=query,
        limit=limit,
        return_properties=return_props
    )

def hybrid(coll, query: str, limit: int, return_props: List[str], alpha: float = 0.75):
    # alpha: 0 → pure keyword, 1 → pure vector
    return coll.query.hybrid(
        query=query,
        limit=limit,
        alpha=alpha,
        return_properties=return_props
    )

# ---- Level searchers ----
def search_chunks(client, query: str, limit: int, mode: str):
    coll = client.collections.get(CHUNK_COLL)
    ret = ["chunk_id", "pali_text"]
    res = near_text(coll, query, limit, ret) if mode == "vector" else hybrid(coll, query, limit, ret)
    print_header(f"[Chunk] top {limit} — mode={mode}")
    print_obj_list(res.objects or [], ret)

def search_subchunks(client, query: str, limit: int, mode: str):
    coll = client.collections.get(SUB_COLL)
    ret = ["sub_chunk_id", "chunk_id", "pali_text", "order_idx"]
    res = near_text(coll, query, limit, ret) if mode == "vector" else hybrid(coll, query, limit, ret)
    print_header(f"[SubChunk] top {limit} — mode={mode}")
    print_obj_list(res.objects or [], ret)

def search_sentences(client, query: str, limit: int, mode: str):
    coll = client.collections.get(SENT_COLL)
    ret = ["sentence_id", "sub_chunk_id", "pali_text", "order_idx"]
    res = near_text(coll, query, limit, ret) if mode == "vector" else hybrid(coll, query, limit, ret)
    print_header(f"[Sentence] top {limit} — mode={mode}")
    print_obj_list(res.objects or [], ret)

# ---- Adjacent sentence search (A and B in neighboring sentences) ----
def adjacent_sentence_search(client, term_a: str, term_b: str, per_term_fetch: int = 300, window: int = 1):
    """
    Strategy:
      1) Fetch sentences likely containing term_a and term_b (hybrid for recall)
      2) Index them by (sub_chunk_id, order_idx)
      3) Report pairs where |order_idx_a - order_idx_b| <= window and same sub_chunk_id
    """
    coll = client.collections.get(SENT_COLL)
    ret = ["sentence_id", "sub_chunk_id", "pali_text", "order_idx"]

    ra = hybrid(coll, term_a, per_term_fetch, ret, alpha=0.25)
    rb = hybrid(coll, term_b, per_term_fetch, ret, alpha=0.25)
    A = [(o.properties["sub_chunk_id"], int(o.properties["order_idx"]), o) for o in (ra.objects or []) if o.properties]
    B = [(o.properties["sub_chunk_id"], int(o.properties["order_idx"]), o) for o in (rb.objects or []) if o.properties]

    # Build index for B by sub_chunk
    idxB: Dict[str, List[Tuple[int, Any]]] = {}
    for sid, ord_idx, obj in B:
        idxB.setdefault(sid, []).append((ord_idx, obj))

    # Find neighbors
    pairs = []
    for sid, ord_idx, objA in A:
        if sid not in idxB:
            continue
        for ordB, objB in idxB[sid]:
            if abs(ord_idx - ordB) <= window:
                pairs.append((objA, objB))

    # Deduplicate by (A,B) ids
    seen = set()
    uniq_pairs = []
    for a, b in pairs:
        key = (a.uuid, b.uuid) if a.uuid <= b.uuid else (b.uuid, a.uuid)
        if key not in seen:
            seen.add(key)
            uniq_pairs.append((a, b))

    # Print
    print_header(f"[Sentence-Adjacent] termA='{term_a}' termB='{term_b}' | window={window} | pairs={len(uniq_pairs)}")
    if not uniq_pairs:
        print("∅ No adjacent pairs found.")
        return

    for i, (a, b) in enumerate(uniq_pairs, 1):
        pa = a.properties; pb = b.properties
        print(f"{i:>2}. sub_chunk={pa['sub_chunk_id']} | orders=({pa['order_idx']},{pb['order_idx']})")
        print(f"    A[{pa['sentence_id']}]: {ptrim(pa['pali_text'])}")
        print(f"    B[{pb['sentence_id']}]: {ptrim(pb['pali_text'])}")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Weaviate Tipitaka Search (Chunk/Sub/Sentence + Adjacent mode)")
    ap.add_argument("--q", "--query", dest="query", required=True, help="Search query text (Pali terms)")
    ap.add_argument("--level", choices=["chunk", "sub", "sentence", "all"], default="all", help="Search level")
    ap.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid", help="hybrid or pure vector")
    ap.add_argument("--k", "--limit", dest="limit", type=int, default=DEFAULT_LIMIT, help="Top K per level")
    ap.add_argument("--adjacent", action="store_true", help="Sentence-adjacent search for two terms (A B)")
    ap.add_argument("--window", type=int, default=1, help="Adjacency window in sentence order (default 1 = prev/next)")
    ap.add_argument("--fetch", type=int, default=300, help="Per-term fetch size for adjacent mode")
    args = ap.parse_args()

    terms = split_terms(args.query)
    if args.adjacent and len(terms) < 2:
        print("⚠️  --adjacent mode requires at least two terms, e.g. --q 'wordA wordB'")
        sys.exit(1)

    client = connect()
    try:
        if args.adjacent:
            # only sentence-level, find adjacent pairs for first two tokens
            term_a, term_b = terms[0], terms[1]
            adjacent_sentence_search(client, term_a, term_b, per_term_fetch=args.fetch, window=args.window)
            return

        # normal search
        if args.level in ("chunk", "all"):
            search_chunks(client, args.query, args.limit, args.mode)
        if args.level in ("sub", "all"):
            search_subchunks(client, args.query, args.limit, args.mode)
        if args.level in ("sentence", "all"):
            search_sentences(client, args.query, args.limit, args.mode)

    finally:
        client.close()

if __name__ == "__main__":
    main()
