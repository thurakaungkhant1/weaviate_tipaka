# sync_to_weaviate.py  (conceptual – adjust to your v4 client)
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Property, DataType, Configure

client = weaviate.WeaviateClient(ConnectionParams.from_url("http://localhost:8080", grpc_port=50051))
client.connect()

# 1) create collections (idempotent create)
def create_collections():
    try:
        client.collections.create(
            name="Chunk",
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="pali_text", data_type=DataType.TEXT),
                Property(name="order_idx", data_type=DataType.INT)
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
    except Exception: pass

    try:
        client.collections.create(
            name="SubChunk",
            properties=[
                Property(name="sub_chunk_id", data_type=DataType.TEXT),
                Property(name="pali_text", data_type=DataType.TEXT),
                Property(name="order_idx", data_type=DataType.INT)
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
    except Exception: pass

    try:
        client.collections.create(
            name="Sentence",
            properties=[
                Property(name="sentence_id", data_type=DataType.TEXT),
                Property(name="pali_text", data_type=DataType.TEXT),
                Property(name="order_idx", data_type=DataType.INT)
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
    except Exception: pass

create_collections()

# 2) read from SQLite and insert with external vectors (LaBSE)
import sqlite3
conn = sqlite3.connect("tipitaka.db")
c = conn.cursor()

# helper: insert + keep uuids for references
from uuid import uuid4
chunk_uuid = {}
sub_uuid = {}

# insert Chunks
for (chunk_id, pali_text, _toks, order_idx) in c.execute("SELECT chunk_id, pali_text, token_count, order_idx FROM Chunk ORDER BY order_idx"):
    uuid = str(uuid4())
    chunk_uuid[chunk_id] = uuid
    client.collections.get("Chunk").data.insert(
        properties={"chunk_id": chunk_id, "pali_text": pali_text, "order_idx": order_idx},
        vector=None  # or your LaBSE vector here
    )

# insert SubChunks + add ref to parent Chunk
for (sub_id, chunk_id, pali_text, _toks, order_idx) in c.execute("""
  SELECT sub_chunk_id, chunk_id, pali_text, token_count, order_idx
  FROM SubChunk ORDER BY chunk_id, order_idx
"""):
    uuid = str(uuid4())
    sub_uuid[sub_id] = uuid
    client.collections.get("SubChunk").data.insert(
        properties={"sub_chunk_id": sub_id, "pali_text": pali_text, "order_idx": order_idx},
        vector=None  # or vector
    )
    # reference: SubChunk.belongsToChunk → Chunk
    client.collections.get("SubChunk").references.add(
        from_uuid=uuid,
        from_property="belongsToChunk",
        to=client.collections.get("Chunk").generate.combine_with_id(chunk_uuid[chunk_id])
    )

# insert Sentences + add ref to parent SubChunk
for (sent_id, sub_id, pali_text, order_idx) in c.execute("""
  SELECT sentence_id, sub_chunk_id, pali_text, order_idx
  FROM Sentence ORDER BY sub_chunk_id, order_idx
"""):
    uuid = str(uuid4())
    client.collections.get("Sentence").data.insert(
        properties={"sentence_id": sent_id, "pali_text": pali_text, "order_idx": order_idx},
        vector=None  # or vector
    )
    client.collections.get("Sentence").references.add(
        from_uuid=uuid,
        from_property="belongsToSubChunk",
        to=client.collections.get("SubChunk").generate.combine_with_id(sub_uuid[sub_id])
    )

conn.close()
client.close()
print("Synced to Weaviate.")
