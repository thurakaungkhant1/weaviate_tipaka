import csv
import os
import sys
import weaviate

CSV_FILE = "wide_chunks.csv"
BATCH_SIZE = 1000

def to_int(x, name):
    try:
        return int(str(x).strip())
    except Exception:
        raise ValueError(f"Row has non-integer {name}: {x!r}")

print("📡 Connecting to Weaviate...")
client = weaviate.connect_to_local()
print("✅ Connected!")

try:
    # Ensure collection exists
    names = [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in client.collections.list_all()]
    if "PaliText" not in names:
        print("❌ Collection 'PaliText' not found. Run schema.py first.")
        sys.exit(1)

    coll = client.collections.get("PaliText")

    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV not found: {CSV_FILE}")
        sys.exit(1)

    print(f"📂 Reading CSV: {CSV_FILE}")
    required = ["chunk_id","chunk_text","subchunk_id","subchunk_text","sentence_id","sentence_text"]

    buf = []
    total = 0
    with open(CSV_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # check headers
        for col in required:
            if col not in reader.fieldnames:
                print(f"❌ Missing column in CSV: {col}")
                sys.exit(1)

        for row in reader:
            # cast to INT where schema requires
            chunk_id    = to_int(row["chunk_id"], "chunk_id")
            subchunk_id = to_int(row["subchunk_id"], "subchunk_id")
            sentence_id = to_int(row["sentence_id"], "sentence_id")

            props = {
                "chunk_id": chunk_id,                         # INT
                "subchunk_id": subchunk_id,                   # INT
                "sentence_id": sentence_id,                   # INT
                "chunk_text": row.get("chunk_text", "") or "",
                "subchunk_text": row.get("subchunk_text", "") or "",
                "sentence_text": row.get("sentence_text", "") or "",
            }

            # Stable, idempotent UUID (optional but recommended)
            uuid = f"{chunk_id:06d}-{subchunk_id:06d}-{sentence_id:06d}"

            buf.append({"properties": props, "uuid": uuid})
            total += 1

            if len(buf) >= BATCH_SIZE:
                coll.data.insert_many(buf)
                print(f"✅ Inserted {total}")
                buf.clear()

        if buf:
            coll.data.insert_many(buf)
            print(f"✅ Inserted {total}")

    print(f"🎯 Done! Inserted total {total} records.")
finally:
    client.close()
    print("🔌 Connection closed.")
