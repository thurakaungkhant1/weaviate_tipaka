import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.exceptions import WeaviateInvalidInputError

print("📡 Connecting...")
client = weaviate.connect_to_local()
print("✅ Connected")

try:
    # Drop if exists
    names = [c if isinstance(c, str) else c.name for c in client.collections.list_all()]
    if "PaliText" in names:
        print("🗑 Dropping existing collection: PaliText")
        client.collections.delete("PaliText")

    print("🛠 Creating collection (try vector_config first)")
    try:
        # New-style (some client versions need this)
        client.collections.create(
            name="PaliText",
            properties=[
                Property(name="chunk_id", data_type=DataType.INT),
                Property(name="subchunk_id", data_type=DataType.INT),
                Property(name="sentence_id", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="subchunk_text", data_type=DataType.TEXT),
                Property(name="sentence_text", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectorizer.text2vec_transformers(),  # NEW param
        )
        print("✅ Created with vector_config")
    except WeaviateInvalidInputError:
        print("↩️  Falling back to deprecated vectorizer_config ...")
        client.collections.create(
            name="PaliText",
            properties=[
                Property(name="chunk_id", data_type=DataType.INT),
                Property(name="subchunk_id", data_type=DataType.INT),
                Property(name="sentence_id", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="subchunk_text", data_type=DataType.TEXT),
                Property(name="sentence_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),  # OLD param (works)
        )
        print("✅ Created with vectorizer_config (deprecated)")

    print("🎉 Schema ready: PaliText")

finally:
    client.close()
    print("🔌 Closed.")
