from app.chunker import chunk_documents
from app.file_filter import filter_repo_files
from app.github_loader import fetch_repo_metadata, list_repo_files, parse_repo_url
from app.repo_downloader import download_repo_files
from app.vector_store import VectorStore


class RepoIndex:
    """Everything we know about one indexed repository."""

    def __init__(self, owner, repo, metadata, store, file_count, skipped):
        self.owner = owner
        self.repo = repo
        self.metadata = metadata
        self.store = store
        self.file_count = file_count
        self.skipped = skipped

    @property
    def chunk_count(self):
        return len(self.store.chunks)


def build_repo_index(url, progress=None):
    """Go from a GitHub URL to a searchable index.

    progress: an optional function that receives status messages (strings).
    Raises GitHubError or ValueError with a message that is safe to show.
    """

    def report(message):
        if progress:
            progress(message)

    owner, repo = parse_repo_url(url)

    report("Fetching repository information...")
    metadata = fetch_repo_metadata(owner, repo)
    branch = metadata["default_branch"]

    report("Listing files...")
    all_files = list_repo_files(owner, repo, branch)
    wanted = filter_repo_files(all_files)
    if not wanted:
        raise ValueError(
            "No supported source files found. This version only indexes Python (.py) files."
        )

    report("Downloading " + str(len(wanted)) + " files...")
    documents, skipped = download_repo_files(owner, repo, branch, wanted)
    if not documents:
        raise ValueError("Could not read any source files from this repository.")

    report("Splitting code into chunks...")
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("The source files contained no code to index.")

    report("Building the search index (" + str(len(chunks)) + " chunks)...")
    store = VectorStore(chunks)

    report("Done.")
    return RepoIndex(owner, repo, metadata, store, len(documents), skipped)


def retrieve(repo_index, question, k=5):
    """Return the k chunks most relevant to the question."""
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    return repo_index.store.search(question.strip(), k=k)


def format_source(chunk):
    """A citation like 'src/flask/app.py:120-150'."""
    return chunk["path"] + ":" + str(chunk["start_line"]) + "-" + str(chunk["end_line"])


if __name__ == "__main__":
    import time

    from app.github_loader import GitHubError

    # Bad input should give a clean message, not a crash.
    try:
        build_repo_index("https://github.com/pallets")
    except (GitHubError, ValueError) as error:
        print("Expected error test ->", error)
    print()

    try:
        started = time.time()
        repo_index = build_repo_index("https://github.com/pallets/flask", progress=print)
        seconds = round(time.time() - started)
        print()
        print(
            "Indexed", repo_index.file_count, "files into",
            repo_index.chunk_count, "chunks in", seconds, "seconds",
        )

        questions = [
            "How does Flask register a URL route?",
            "How are templates rendered?",
            "How are sessions stored in cookies?",
            "What is the recipe for chocolate cake?",  # NOT in the repo on purpose
        ]
        for question in questions:
            results = retrieve(repo_index, question, k=3)
            print()
            print("Q:", question)
            for rank, r in enumerate(results, start=1):
                c = r["chunk"]
                print("   ", rank, format_source(c), "|", c["symbol"], "| score", round(r["score"], 2))

        # Peek at the text of the best chunk for the first question.
        best = retrieve(repo_index, questions[0], k=1)[0]["chunk"]
        print()
        print("First lines of the top chunk for the first question:")
        for line in best["text"].splitlines()[:5]:
            print("   ", line.encode("ascii", "replace").decode())

    except (GitHubError, ValueError) as error:
        print("ERROR:", error)