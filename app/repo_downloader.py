import base64
from urllib.parse import quote

from app.github_loader import GitHubError, github_get


def download_file(owner, repo, branch, path):
    """Download one file and return its text.

    Returns None if the file cannot be read as text.
    """
    response = github_get(
        "/repos/" + owner + "/" + repo + "/contents/" + quote(path, safe="/"),
        params={"ref": branch},
    )
    data = response.json()

    # For normal files GitHub sends the text as base64.
    if data.get("encoding") != "base64":
        return None

    raw_bytes = base64.b64decode(data["content"])
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def download_repo_files(owner, repo, branch, files):
    """Download the text of every file in `files`.

    files: the filtered list of {"path", "size"} dicts.
    Returns (documents, skipped_paths), where documents is a list of
    {"path", "content"} dicts.
    """
    documents = []
    skipped = []

    for index, f in enumerate(files, start=1):
        path = f["path"]

        try:
            text = download_file(owner, repo, branch, path)
        except GitHubError as error:
            # A rate limit affects every file, so stop and report it.
            if "rate limit" in str(error).lower():
                raise
            skipped.append(path)
            continue

        if text is None or not text.strip():
            skipped.append(path)
            continue

        documents.append({"path": path, "content": text})

        # Temporary progress output (Stage 7 replaces this with a progress bar).
        if index % 10 == 0:
            print("  downloaded", index, "of", len(files))

    return documents, skipped


if __name__ == "__main__":
    from app.file_filter import filter_repo_files
    from app.github_loader import fetch_repo_metadata, list_repo_files, parse_repo_url

    try:
        owner, repo = parse_repo_url("https://github.com/pallets/flask")
        metadata = fetch_repo_metadata(owner, repo)
        branch = metadata["default_branch"]

        limit = github_get("/rate_limit").headers.get("x-ratelimit-limit")
        print("Hourly request limit:", limit)

        all_files = list_repo_files(owner, repo, branch)
        wanted = filter_repo_files(all_files)
        print("Files to download:", len(wanted))

        documents, skipped = download_repo_files(owner, repo, branch, wanted)

        total_chars = sum(len(d["content"]) for d in documents)
        print()
        print("Downloaded:", len(documents))
        print("Skipped:", len(skipped))
        print("Total characters:", total_chars)

        print("First 3 documents:")
        for d in documents[:3]:
            print("  ", d["path"], "-", len(d["content"].splitlines()), "lines")

        remaining = github_get("/rate_limit").headers.get("x-ratelimit-remaining")
        print("Requests left this hour:", remaining)

    except (GitHubError, ValueError) as error:
        print("ERROR:", error)