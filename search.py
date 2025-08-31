# search.py — hybrid→bm25→nearText fallback, skip empty sentences client-side
import weaviate
from weaviate.exceptions import WeaviateQueryError

QUERY_TEXT = "bhikkhave"  # ဥပမာ query ကို ဒီလိုထားပြီး စမ်းကြည့်ပါ
LIMIT = 10

def print_results(res):
    objs = (getattr(res, "objects", []) or [])
    filtered = []
    for obj in objs:
        p = (getattr(obj, "properties", None) or {})
        sent_txt = (p.get("sentence_text") or "").strip()
        if sent_txt:
            filtered.append(p)
    print(f"📄 Results ({len(filtered)} hits with sentence_text):")
    if not filtered:
        print("   (No non-empty sentences matched.)")
    else:
        for i, p in enumerate(filtered, start=1):
            chunk_label = str(p.get("chunk_id", "?"))
            sub_label   = str(p.get("subchunk_id", "?"))
            sent_label  = str(p.get("sentence_id", "?"))
            preview     = p["sentence_text"].replace("\n", " ")[:200]
            print(f"{i}. [{chunk_label}-{sub_label}-{sent_label}] {preview}")

print("📡 Connecting to Weaviate...")
client = weaviate.connect_to_local()
print("✅ Connected!")
try:
    coll = client.collections.get("PaliText")

    # 1) HYBRID (BM25 + vector)
    #    alpha=0.3 -> BM25 ကို ပိုတော်တော်အလေးပေး (short phrase တွေအတွက် အထူးထိ)
    print(f"🔍 hybrid search: {QUERY_TEXT}")
    try:
        res = coll.query.hybrid(
            query=QUERY_TEXT,
            limit=LIMIT,
            alpha=0.3,
            # vector=...  # normally not needed
            return_properties=["chunk_id","subchunk_id","sentence_id","sentence_text"],
            # query_properties narrowed to sentence_text to avoid noise
            query_properties=["sentence_text"],
        )
        print_results(res)
        # If nothing found, keep falling back
        if len(getattr(res, "objects", []) or []) > 0:
            raise SystemExit
    except WeaviateQueryError as e:
        print(f"⚠️ hybrid failed ({e}); trying BM25...")

    # 2) BM25 keyword only
    print(f"🔍 bm25 search: {QUERY_TEXT}")
    try:
        res = coll.query.bm25(
            query=QUERY_TEXT,
            limit=LIMIT,
            return_properties=["chunk_id","subchunk_id","sentence_id","sentence_text"],
            query_properties=["sentence_text"],
        )
        print_results(res)
        if len(getattr(res, "objects", []) or []) > 0:
            raise SystemExit
    except WeaviateQueryError as e:
        print(f"⚠️ bm25 failed ({e}); trying nearText...")

    # 3) nearText (vector only)
    print(f"🔍 nearText search: {QUERY_TEXT}")
    try:
        res = coll.query.near_text(
            query=QUERY_TEXT,
            limit=LIMIT,
            return_properties=["chunk_id","subchunk_id","sentence_id","sentence_text"],
        )
        print_results(res)
    except WeaviateQueryError as e:
        print(f"❌ nearText failed as well: {e}")

finally:
    client.close()
    print("🔌 Connection closed.")
