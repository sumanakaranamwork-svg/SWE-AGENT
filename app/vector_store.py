import faiss
import numpy as np

from app.embedder import chunk_to_embedding_text, embed_texts


class VectorStore:
    """Holds the chunks plus a FAISS index, and finds the chunks closest to a question."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks to index.")

        self.chunks = chunks

        # One vector per chunk, in the same order as the chunks list.
        texts = [chunk_to_embedding_text(c) for c in chunks]
        vectors = embed_texts(texts)

        # Exact search, inner product (= cosine similarity for unit-length vectors).
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, k=5):
        """Return the k most relevant chunks as [{"chunk": ..., "score": ...}, ...]."""
        k = min(k, len(self.chunks))
        query_vector = embed_texts([query])

        scores, positions = self.index.search(query_vector, k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position == -1:  # FAISS uses -1 for "no result"
                continue
            results.append({"chunk": self.chunks[int(position)], "score": float(score)})
        return results


if __name__ == "__main__":
    from pathlib import Path

    from app.chunker import chunk_documents

    # Index this project's own files. We skip vector_store.py itself, because
    # it contains the test questions below and would "find" its own text.
    documents = []
    for file_path in sorted(Path(__file__).parent.glob("*.py")):
        if file_path.name == "vector_store.py":
            continue
        documents.append(
            {"path": "app/" + file_path.name, "content": file_path.read_text(encoding="utf-8")}
        )

    chunks = chunk_documents(documents)
    print("Chunks to index:", len(chunks))
    print("Building the index (embedding every chunk)...")
    store = VectorStore(chunks)
    print("Vectors stored in FAISS:", store.index.ntotal)

    # Each question is paired with the file that SHOULD appear in the top 3.
    tests = [
        ("extract the owner and repo name from a github url", "app/github_loader.py"),
        ("download a file from github and decode its base64 text", "app/repo_downloader.py"),
        ("split a long function into overlapping windows of lines", "app/chunker.py"),
        ("skip images and node_modules folders when choosing files", "app/file_filter.py"),
    ]

    for question, expected_path in tests:
        results = store.search(question, k=3)
        top_paths = [r["chunk"]["path"] for r in results]
        status = "PASS" if expected_path in top_paths else "FAIL"

        print()
        print(status, "-", question)
        for rank, r in enumerate(results, start=1):
            c = r["chunk"]
            print(
                "   ", rank, c["path"], "lines", c["start_line"], "-", c["end_line"],
                "|", c["symbol"], "| score", round(r["score"], 2),
            )