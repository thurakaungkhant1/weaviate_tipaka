import weaviate
import pandas as pd
import uuid

CSV_PATH = "chunks_8000token_ids.csv"   # <- လက်ရှိဖိုင်လမ်းကြောင်းနဲ့ ကိုက်ညီအောင်
COLL = "PaliChunk"

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"📄 Loaded {len(df)} rows from {CSV_PATH}")

    # text2vec-transformers module အသင့် run ဖြစ်နေသတ်မှတ်ချက်: ENABLE_MODULES, INFERENCE API
    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(COLL)

        # vector မပို့ပါ — Weaviate ကိုယ်တိုင် pali_text ကနေ vector တွက်စေမယ်
        objs = []
        for idx, row in df.iterrows():
            props = {
                "chunk_id": str(row["chunk_id"]),
                "pali_text": str(row["pali_text"]),
            }
            objs.append({"properties": props, "uuid": str(uuid.uuid4())})

            if len(objs) >= 500:  # mini-batch
                col.data.insert_many(objs)
                print(f"✅ Inserted {idx+1} rows...")
                objs = []

        if objs:
            col.data.insert_many(objs)
            print(f"✅ Inserted {len(df)} rows.")

        print("🎉 Done (auto-vectorized by text2vec).")
    finally:
        client.close()

if __name__ == "__main__":
    main()
