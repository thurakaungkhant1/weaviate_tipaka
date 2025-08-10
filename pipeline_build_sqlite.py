# pipeline_build_sqlite.py
import csv, re, sqlite3, math
from pathlib import Path

DB_PATH = "tipitaka.db"
CSV_PATH = "chunks_8000token_ids.csv"

MAIN_CHUNK_LIMIT = 8000    # info only (already applied)
SUB_CHUNK_LIMIT  = 200     # tokens per sub-chunk

# very simple whitespace tokenizer
def tokenize(text: str):
    return text.split()

def detokenize(tokens):
    return " ".join(tokens)

# Sentence split rule per your spec:
#   - starts with Capital (A-Z Ā ... if you have extended caps, add ranges)
#   - ends with ". "  (dot + space)
# Note: Keep it pragmatic; you can refine for Pali punctuation later.
SENTENCE_PATTERN = re.compile(r'([A-Z][\s\S]*?\. )')

def split_sentences(pali_text: str):
    # ensure trailing space so regex can catch last sentence with ". "
    t = pali_text if pali_text.endswith(" ") else pali_text + " "
    parts = SENTENCE_PATTERN.findall(t)
    # fallback: if nothing matched, treat as one sentence
    return [p.strip() for p in parts] or [pali_text.strip()]

def ensure_parent_dirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def main():
    ensure_parent_dirs(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    # create schema
    with open("tipitaka.sql", "w", encoding="utf-8") as f:
        f.write("""PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS Chunk (
  chunk_id      TEXT PRIMARY KEY,
  pali_text     TEXT NOT NULL,
  token_count   INTEGER NOT NULL,
  order_idx     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS SubChunk (
  sub_chunk_id  TEXT PRIMARY KEY,
  chunk_id      TEXT NOT NULL,
  pali_text     TEXT NOT NULL,
  token_count   INTEGER NOT NULL,
  order_idx     INTEGER NOT NULL,
  FOREIGN KEY (chunk_id) REFERENCES Chunk(chunk_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS Sentence (
  sentence_id   TEXT PRIMARY KEY,
  sub_chunk_id  TEXT NOT NULL,
  pali_text     TEXT NOT NULL,
  order_idx     INTEGER NOT NULL,
  FOREIGN KEY (sub_chunk_id) REFERENCES SubChunk(sub_chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_subchunk_chunk ON SubChunk(chunk_id, order_idx);
CREATE INDEX IF NOT EXISTS idx_sentence_subchunk ON Sentence(sub_chunk_id, order_idx);
""")

    with open("tipitaka.sql", "r", encoding="utf-8") as f:
        cur.executescript(f.read())

    # ingest chunks.csv
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for ci, row in enumerate(reader, start=1):
            chunk_id   = str(row["chunk_id"]).strip()
            pali_text  = str(row["pali_text"]).strip()
            tokens     = tokenize(pali_text)
            cur.execute(
                "INSERT OR REPLACE INTO Chunk(chunk_id,pali_text,token_count,order_idx) VALUES (?,?,?,?)",
                (chunk_id, pali_text, len(tokens), ci)
            )

            # split into sub-chunks (200 tokens)
            sub_count = math.ceil(max(1, len(tokens)) / SUB_CHUNK_LIMIT)
            for si in range(sub_count):
                start = si * SUB_CHUNK_LIMIT
                end   = min((si+1)*SUB_CHUNK_LIMIT, len(tokens))
                sub_tokens = tokens[start:end]
                sub_text   = detokenize(sub_tokens)
                sub_id     = f"{chunk_id}.{si+1:03d}"   # e.g., 12 → 12.001, 12.002

                cur.execute(
                    "INSERT OR REPLACE INTO SubChunk(sub_chunk_id,chunk_id,pali_text,token_count,order_idx) VALUES (?,?,?,?,?)",
                    (sub_id, chunk_id, sub_text, len(sub_tokens), si+1)
                )

                # split sentences within sub-chunk
                sents = split_sentences(sub_text)
                for pi, sent in enumerate(sents, start=1):
                    sent_id = f"{sub_id}.{pi:03d}"       # e.g., 12.001.001
                    cur.execute(
                        "INSERT OR REPLACE INTO Sentence(sentence_id,sub_chunk_id,pali_text,order_idx) VALUES (?,?,?,?)",
                        (sent_id, sub_id, sent, pi)
                    )

    conn.commit()
    conn.close()
    print("SQLite build complete →", DB_PATH)

if __name__ == "__main__":
    main()
