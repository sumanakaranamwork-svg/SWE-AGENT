import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_model():
    """Load the embedding model once and reuse it afterwards."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=32):
    """Turn a list of strings into an array of shape (len(texts), 384)."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # every vector gets length 1
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def chunk_to_embedding_text(chunk):
    """The text we embed for a chunk: a short header (file + name) plus the code."""
    symbol = chunk.get("symbol") or "<file>"
    return "File: " + chunk["path"] + "\nName: " + symbol + "\n" + chunk["text"]


if __name__ == "__main__":
    snippets = [
        "def get_connection():\n    return sqlite3.connect(DB_PATH)",
        "def send_email(to, subject, body):\n    smtp.send_message(to, subject, body)",
        "def parse_date(text):\n    return datetime.strptime(text, '%Y-%m-%d')",
    ]
    # Each query is paired with the snippet that SHOULD be the best match.
    queries = [
        ("open a database connection", 0),
        ("send a message to a user by email", 1),
        ("convert a string into a date", 2),
    ]

    print("Loading the model (the first run downloads it, please wait)...")
    snippet_vectors = embed_texts(snippets)
    print("Embedding shape:", snippet_vectors.shape)
    print("Length of the first vector:", round(float(np.linalg.norm(snippet_vectors[0])), 3))

    query_vectors = embed_texts([q for q, _ in queries])

    # Vectors have length 1, so a dot product equals cosine similarity.
    scores = query_vectors @ snippet_vectors.T

    print()
    for i, (query, expected) in enumerate(queries):
        best = int(np.argmax(scores[i]))
        status = "PASS" if best == expected else "FAIL"
        print(status, "-", repr(query), "-> best match is snippet", best)
        print("      scores:", [round(float(s), 2) for s in scores[i]])