from datetime import datetime
from urllib.parse import urlparse

import requests

from app.config import GITHUB_TOKEN

API_BASE = "https://api.github.com"
REQUEST_TIMEOUT = 15  # seconds


class GitHubError(Exception):
    """A GitHub problem, with a message that is safe to show to the user."""


def parse_repo_url(url):
    """Extract (owner, repo) from a GitHub repository URL.

    Raises ValueError if the URL is not a valid GitHub repo URL.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a GitHub repository URL.")

    url = url.strip()
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Not a github.com URL: " + url)

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("URL must look like https://github.com/owner/repo")

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo:
        raise ValueError("URL must look like https://github.com/owner/repo")

    return owner, repo


def _build_headers():
    """Headers sent with every GitHub request."""
    headers = {"Accept": "application/vnd.github+json"}
    # Only send the token if it is a real one, not the placeholder.
    if GITHUB_TOKEN and not GITHUB_TOKEN.startswith("your_"):
        headers["Authorization"] = "Bearer " + GITHUB_TOKEN
    return headers


def github_get(path, params=None):
    """Make a GET request to the GitHub API.

    Returns the response if successful.
    Raises GitHubError with a clear message otherwise.
    """
    url = API_BASE + path
    try:
        response = requests.get(
            url,
            headers=_build_headers(),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise GitHubError(
            "Could not reach GitHub. Check your internet connection and try again."
        ) from error

    status = response.status_code

    if status == 200:
        return response

    if status == 404:
        raise GitHubError(
            "Not found on GitHub. The repository may not exist, or it may be private."
        )

    if status == 409:
        raise GitHubError("This repository is empty or has no files to read.")

    if status == 401:
        raise GitHubError("GitHub rejected the token in your .env file. Check GITHUB_TOKEN.")

    if status in (403, 429):
        if response.headers.get("x-ratelimit-remaining") == "0":
            reset = response.headers.get("x-ratelimit-reset", "")
            if reset.isdigit():
                when = datetime.fromtimestamp(int(reset)).strftime("%H:%M")
                raise GitHubError(
                    "GitHub rate limit reached. Try again after " + when
                    + ", or add a GITHUB_TOKEN to your .env file."
                )
            raise GitHubError("GitHub rate limit reached. Try again later.")
        if status == 429 or "rate limit" in response.text.lower():
            raise GitHubError("GitHub asked us to slow down. Wait a minute and try again.")
        raise GitHubError("GitHub refused this request (403).")

    raise GitHubError("GitHub returned an unexpected status: " + str(status))


def fetch_repo_metadata(owner, repo):
    """Return basic information about a public repository."""
    response = github_get("/repos/" + owner + "/" + repo)
    data = response.json()

    if data.get("private"):
        raise GitHubError("Private repositories are not supported.")

    return {
        "full_name": data["full_name"],
        "description": data.get("description"),
        "default_branch": data["default_branch"],
        "language": data.get("language"),
        "stars": data.get("stargazers_count"),
        "size_kb": data.get("size"),
        "archived": data.get("archived"),
    }


def list_repo_files(owner, repo, branch):
    """Return every file in the repository as a list of {"path", "size"} dicts."""
    response = github_get(
        "/repos/" + owner + "/" + repo + "/git/trees/" + branch,
        params={"recursive": "1"},
    )
    data = response.json()

    if data.get("truncated"):
        raise GitHubError("This repository is too large to list completely.")

    files = []
    for item in data.get("tree", []):
        if item.get("type") == "blob":  # "blob" = a file, "tree" = a folder
            files.append({"path": item["path"], "size": item.get("size", 0)})
    return files


if __name__ == "__main__":
    try:
        owner, repo = parse_repo_url("https://github.com/pallets/flask")
        metadata = fetch_repo_metadata(owner, repo)
        files = list_repo_files(owner, repo, metadata["default_branch"])

        print("Repository:", metadata["full_name"])
        print("Total files:", len(files))

        print("First 10 paths:")
        for f in files[:10]:
            print("  ", f["path"])

        print("3 largest files:")
        for f in sorted(files, key=lambda item: item["size"], reverse=True)[:3]:
            print("  ", f["size"], "bytes -", f["path"])

    except (GitHubError, ValueError) as error:
        print("ERROR:", error)

    # Deliberate failure test: this repo does not exist, so we EXPECT an error message.
    try:
        fetch_repo_metadata("pallets", "this-repo-does-not-exist-12345")
    except GitHubError as error:
        print("Expected error test ->", error)