# insert_from_sqlite.py
import sqlite3
import weaviate
from weaviate.classes.query import Filter

BATCH_SZ = 500
DB_PATH = "tipitaka.db"   # pipeline_build_sqlite.py ထုတ်ထားတဲ့ DB

client = weaviate.connect_to_local()
coll_chunk = client.collections.get("PaliChunk")
coll_sub   = client.collections.get("PaliSubChunk")
coll_sent  = client.collections.get("PaliSentence")

def build_uuid_map(collection, key_prop: str):
    """Return dict: {key_prop_value: uuid} for the whole collection."""
    uuid_map = {}
    cursor = None
    while True:
        res = collection.query.fetch_objects(
            limit=1000,
            after=cursor,
            return_properties=[key_prop]
        )
        for obj in res.objects or []:
            k = obj.properties.get(key_prop)
            if k:
                uuid_map[str(k)] = obj.uuid
        cursor = res.page_info.end_cursor if res.page_info else None
        if not res.objects or len(res.objects) == 0 or not cursor:
            break
    return uuid_map

def main():
    # 1) Load parent uuid map for PaliChunk (key = chunk_id)
    print("Building chunk UUID map…")
    chunk_uuid = build_uuid_map(coll_chunk, "chunk_id")
    print(f" - {len(chunk_uuid)} chunks in Weaviate")

    # 2) Insert SubChunk (with reference to PaliChunk)
    #    SQLite table: SubChunk(sub_chunk_id, chunk_id, pali_text, token_count, order_idx)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("Inserting PaliSubChunk…")
    batch = []
    inserted = 0
    sub_uuid = {}

    for (sub_id, chunk_id, pali_text, token_count, order_idx) in c.execute("""
        SELECT sub_chunk_id, chunk_id, pali_text, token_count, order_idx
        FROM SubChunk
        ORDER BY chunk_id, order_idx
    """):
        props = {
            "sub_chunk_id": str(sub_id),
            "pali_text": str(pali_text),
            "order_idx": int(order_idx),
            "chunk_id": str(chunk_id),  # convenience
        }

        # reference payload (if we have parent uuid; if not found, skip ref to avoid errors)
        refs = None
        parent_uuid = chunk_uuid.get(str(chunk_id))
        if parent_uuid:
            refs = { "belongsToChunk": [ coll_chunk.reference.to(parent_uuid) ] }

        batch.append({"properties": props, "references": refs})

        if len(batch) >= BATCH_SZ:
            resp = coll_sub.data.insert_many(batch)
            # collect uuids back for mapping subchunks to sentences
            for o in resp.uuids or []:
                # we don't get properties back in insert_many; rebuild map later
                pass
            inserted += len(batch)
            print(f" - inserted {inserted} sub-chunks")
            batch = []

    if batch:
        coll_sub.data.insert_many(batch)
        inserted += len(batch)
        print(f" - inserted {inserted} sub-chunks (final)")

    # 3) Build uuid map for PaliSubChunk (key = sub_chunk_id)
    print("Building sub_chunk UUID map…")
    sub_uuid = build_uuid_map(coll_sub, "sub_chunk_id")
    print(f" - {len(sub_uuid)} sub-chunks in Weaviate")

    # 4) Insert Sentences (with reference to PaliSubChunk)
    #    SQLite table: Sentence(sentence_id, sub_chunk_id, pali_text, order_idx)
    print("Inserting PaliSentence…")
    batch = []
    inserted = 0

    for (sent_id, sub_id, pali_text, order_idx) in c.execute("""
        SELECT sentence_id, sub_chunk_id, pali_text, order_idx
        FROM Sentence
        ORDER BY sub_chunk_id, order_idx
    """):
        props = {
            "sentence_id": str(sent_id),
            "pali_text": str(pali_text),
            "order_idx": int(order_idx),
            "sub_chunk_id": str(sub_id),  # convenience
        }

        refs = None
        parent_uuid = sub_uuid.get(str(sub_id))
        if parent_uuid:
            refs = { "belongsToSubChunk": [ coll_sub.reference.to(parent_uuid) ] }

        batch.append({"properties": props, "references": refs})

        if len(batch) >= BATCH_SZ:
            coll_sent.data.insert_many(batch)
            inserted += len(batch)
            print(f" - inserted {inserted} sentences")
            batch = []

    if batch:
        coll_sent.data.insert_many(batch)
        inserted += len(batch)
        print(f" - inserted {inserted} sentences (final)")

    conn.close()
    client.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()
