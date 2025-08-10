# schema_weaviate.py  (Pali* names)
import weaviate
from weaviate.classes.config import Property, DataType, Configure
try:
    from weaviate.classes.config import ReferenceProperty
except Exception:
    ReferenceProperty = None

client = weaviate.connect_to_local()

def ensure_collection(name: str, create_fn):
    try:
        client.collections.get(name)
        print(f"ℹ️  {name} already exists — skipping create.")
    except Exception:
        create_fn()
        print(f"✅ {name} created.")

def create_chunk():
    client.collections.create(
        name="PaliChunk",
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="pali_text", data_type=DataType.TEXT),
            Property(name="order_idx", data_type=DataType.INT),
        ],
        vector_config=[
            Configure.Vectors.text2vec_transformers(
                name="pali_vec",
                source_properties=["pali_text"],
            )
        ],
    )

def create_subchunk():
    cfg = dict(
        name="PaliSubChunk",
        properties=[
            Property(name="sub_chunk_id", data_type=DataType.TEXT),
            Property(name="pali_text", data_type=DataType.TEXT),
            Property(name="order_idx", data_type=DataType.INT),
            Property(name="chunk_id", data_type=DataType.TEXT),  # convenience filter
        ],
        vector_config=[
            Configure.Vectors.text2vec_transformers(
                name="pali_vec",
                source_properties=["pali_text"],
            )
        ],
    )
    if ReferenceProperty:
        cfg["references"] = [
            ReferenceProperty(name="belongsToChunk", target_collection="PaliChunk")
        ]
    client.collections.create(**cfg)

def create_sentence():
    cfg = dict(
        name="PaliSentence",
        properties=[
            Property(name="sentence_id", data_type=DataType.TEXT),
            Property(name="pali_text", data_type=DataType.TEXT),
            Property(name="order_idx", data_type=DataType.INT),
            Property(name="sub_chunk_id", data_type=DataType.TEXT),  # convenience filter
        ],
        vector_config=[
            Configure.Vectors.text2vec_transformers(
                name="pali_vec",
                source_properties=["pali_text"],
            )
        ],
    )
    if ReferenceProperty:
        cfg["references"] = [
            ReferenceProperty(name="belongsToSubChunk", target_collection="PaliSubChunk")
        ]
    client.collections.create(**cfg)

def main():
    ensure_collection("PaliChunk", create_chunk)         # ← data တင်ပြီးသားနဲ့ကိုက်
    ensure_collection("PaliSubChunk", create_subchunk)
    ensure_collection("PaliSentence", create_sentence)
    client.close()
    print("Done.")

if __name__ == "__main__":
    main()
