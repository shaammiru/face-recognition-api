CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding vector(2622), -- atau 2622 kalau pakai VGG-Face
    created_at TIMESTAMP DEFAULT NOW()
);