PRAGMA foreign_keys = ON;
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
