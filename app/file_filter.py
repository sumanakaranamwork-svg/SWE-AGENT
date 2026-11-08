import os

# Only files with these extensions are indexed. Add more later if needed.
ALLOWED_EXTENSIONS = {".py"}

# Files inside any folder with one of these names are skipped.
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
}

# Files bigger than this (in bytes) are usually generated, so we skip them.
MAX_FILE_BYTES = 100_000


def should_include(path, size):
    """Decide whether a repository file is worth indexing.

    path: the file path inside the repo, e.g. "src/flask/app.py"
    size: the file size in bytes
    """
    # "src/flask/app.py" -> folders = ["src", "flask"], filename = "app.py"
    *folders, filename = path.split("/")

    for folder in folders:
        if folder in EXCLUDED_DIRS or folder.endswith(".egg-info"):
            return False

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False

    if size == 0 or size > MAX_FILE_BYTES:
        return False

    return True


def filter_repo_files(files):
    """Keep only the files worth indexing.

    files: the list of {"path", "size"} dicts from list_repo_files().
    """
    return [f for f in files if should_include(f["path"], f["size"])]


if __name__ == "__main__":
    # Part 1: offline rule tests (no internet needed).
    # Each line is: (path, size, should it be kept?)
    samples = [
        ("src/app.py", 1200, True),
        ("tests/test_app.py", 2000, True),
        ("README.md", 900, False),                    # wrong extension
        ("docs/logo.png", 5000, False),               # image
        ("venv/lib/site.py", 300, False),             # excluded folder
        ("node_modules/pkg/index.py", 300, False),    # excluded folder
        ("src/__init__.py", 0, False),                # empty file
        ("src/huge_generated.py", 500000, False),     # too large
    ]
    for path, size, expected in samples:
        result = should_include(path, size)
        status = "PASS" if result == expected else "FAIL"
        print(status, "-", path, "->", "keep" if result else "skip")

    # Part 2: live test on a real repository.
    from app.github_loader import (
        GitHubError,
        fetch_repo_metadata,
        list_repo_files,
        parse_repo_url,
    )

    try:
        owner, repo = parse_repo_url("https://github.com/pallets/flask")
        metadata = fetch_repo_metadata(owner, repo)
        all_files = list_repo_files(owner, repo, metadata["default_branch"])
        kept = filter_repo_files(all_files)

        print()
        print("Live test on", metadata["full_name"])
        print("Total files:", len(all_files))
        print("Kept:", len(kept))
        print("Skipped:", len(all_files) - len(kept))
        print("First 15 kept files:")
        for f in kept[:15]:
            print("  ", f["size"], "bytes -", f["path"])

    except (GitHubError, ValueError) as error:
        print("ERROR:", error)