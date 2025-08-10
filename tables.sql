CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    image_base TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);