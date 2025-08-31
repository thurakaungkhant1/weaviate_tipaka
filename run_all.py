# run_all.py — One-shot runner: (re)create schema -> insert CSV -> search
# Usage:
#   python run_all.py --csv wide_chunks.csv --query "anicca" --limit 10 --drop
#
# Notes:
# - Works with your INT schema (chunk_id/subchunk_id/sentence_id).
# - Vectorizer config is version-agnostic (vector_config fallback to vectorizer_config).
# - Batch insert (insert_many), idempotent UUID.
# - nearText → BM25 fallback; empty sentences are skipped client-side.

import argparse
import csv
import os
import sys
import time
import weaviate
from weaviate.exceptions import WeaviateInvalidInputError, WeaviateQueryError
from weaviate.classes.config import Property, DataType, Configure

COLLECTION = "PaliText"

def list_collection_names(client):
    cols = client.collections.list_all()
    names = []
    for c in cols:
        if isinstance(c, str):
            names.append(c)
        else:
            names.append(getattr(c, "name", str(c)))
    return names

def recreate_schema(client, drop_existing: bool):
    names = list_collection_names(client)
    if drop_existing and COLLECTION in names:
        print(f"🗑 Dropping existing collection: {COLLECTION}")
        client.collections.delete(COLLECTION)

    if COLLECTION in list_collection_names(client):
        print(f"ℹ️ Collection '{COLLECTION}' already exists. Skipping creation.")
        return

    print("🛠 Creating collection (try vector_config first)")
    try:
        client.collections.create(
            name=COLLECTION,
            properties=[
                Property(name="chunk_id", data_type=DataType.INT),
                Property(name="subchunk_id", data_type=DataType.INT),
                Property(name="sentence_id", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="subchunk_text", data_type=DataType.TEXT),
                Property(name="sentence_text", data_type=DataType.TEXT),
            ],
            # Preferred on newer weaviate-client
            vector_config=Configure.Vectorizer.text2vec_transformers(),
        )
        print("✅ Created with vector_config")
    except WeaviateInvalidInputError:
        print("↩️  Falling back to deprecated vectorizer_config ...")
        client.collections.create(
            name=COLLECTION,
            properties=[
                Property(name="chunk_id", data_type=DataType.INT),
                Property(name="subchunk_id", data_type=DataType.INT),
                Property(name="sentence_id", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="subchunk_text", data_type=DataType.TEXT),
                Property(name="sentence_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        )
        print("✅ Created with vectorizer_config (deprecated)")

    print(f"🎉 Schema ready: {COLLECTION}")

def to_int(x, name):
    try:
        return int(str(x).strip())
    except Exception:
        raise ValueError(f"Row has non-integer {name}: {x!r}")

def insert_csv(client, csv_path: str, batch_size: int = 1000):
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    names = list_collection_names(client)
    if COLLECTION not in names:
        print(f"❌ Collection '{COLLECTION}' not found. Create schema first.")
        sys.exit(1)

    coll = client.collections.get(COLLECTION)
    required = ["chunk_id","chunk_text","subchunk_id","subchunk_text","sentence_id","sentence_text"]

    print(f"📂 Reading CSV: {csv_path}")
    start = time.time()
    buf = []
    total = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for col in required:
            if col not in reader.fieldnames:
                print(f"❌ Missing column in CSV: {col}")
                sys.exit(1)

        for row in reader:
            chunk_id    = to_int(row["chunk_id"], "chunk_id")
            subchunk_id = to_int(row["subchunk_id"], "subchunk_id")
            sentence_id = to_int(row["sentence_id"], "sentence_id")

            props = {
                "chunk_id": chunk_id,
                "subchunk_id": subchunk_id,
                "sentence_id": sentence_id,
                "chunk_text": row.get("chunk_text", "") or "",
                "subchunk_text": row.get("subchunk_text", "") or "",
                "sentence_text": row.get("sentence_text", "") or "",
            }

            # Stable UUID (idempotent)
            uuid = f"{chunk_id:06d}-{subchunk_id:06d}-{sentence_id:06d}"

            buf.append({"properties": props, "uuid": uuid})
            total += 1

            if len(buf) >= batch_size:
                coll.data.insert_many(buf)
                print(f"✅ Inserted {total}")
                buf.clear()

        if buf:
            coll.data.insert_many(buf)
            print(f"✅ Inserted {total}")

    dur = time.time() - start
    print(f"🎯 Done! Inserted total {total} records in {dur:.1f}s.")

def search(client, query_text: str, limit: int):
    coll = client.collections.get(COLLECTION)

    print(f"🔍 nearText search for: {query_text}")
    try:
        res = coll.query.near_text(
            query=query_text,
            limit=limit,
            return_properties=["chunk_id", "subchunk_id", "sentence_id", "sentence_text"],
        )
    except WeaviateQueryError as e:
        print(f"⚠️ nearText failed ({e}); falling back to BM25 keyword search...")
        res = coll.query.bm25(
            query=query_text,
            limit=limit,
            return_properties=["chunk_id", "subchunk_id", "sentence_id", "sentence_text"],
        )

    objs = (getattr(res, "objects", []) or [])
    # Client-side filter: keep only non-empty sentence_text
    filtered = []
    for obj in objs:
        p = (getattr(obj, "properties", None) or {})
        if (p.get("sentence_text") or "").strip():
            filtered.append(p)

    print(f"📄 Results ({len(filtered)} hits with sentence_text):")
    if not filtered:
        print("   (No non-empty sentences matched. Try a different query.)")
    else:
        for i, p in enumerate(filtered, start=1):
            chunk_label = str(p.get("chunk_id", "?"))
            sub_label   = str(p.get("subchunk_id", "?"))
            sent_label  = str(p.get("sentence_id", "?"))
            preview     = p["sentence_text"].replace("\n", " ")[:200]
            print(f"{i}. [{chunk_label}-{sub_label}-{sent_label}] {preview}")

def main():
    ap = argparse.ArgumentParser(description="Run schema -> insert -> search in one go")
    ap.add_argument("--csv",   default="wide_chunks.csv", help="CSV path")
    ap.add_argument("--query", default="anicca",          help="Query text")
    ap.add_argument("--limit", type=int, default=10,      help="Number of results to return")
    ap.add_argument("--batch", type=int, default=1000,    help="Insert batch size")
    ap.add_argument("--drop",  action="store_true",       help="Drop & recreate collection before insert")
    args = ap.parse_args()

    print("📡 Connecting to Weaviate...")
    client = weaviate.connect_to_local()
    print("✅ Connected!")

    try:
        recreate_schema(client, drop_existing=args.drop)
        insert_csv(client, args.csv, batch_size=args.batch)
        search(client, args.query, args.limit)
    finally:
        client.close()
        print("🔌 Connection closed.")

if __name__ == "__main__":
    main()
