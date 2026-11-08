import ast
import warnings

MAX_CHUNK_LINES = 60   # the longest chunk we allow
OVERLAP_LINES = 10     # lines repeated between neighbouring windows
MIN_CHUNK_CHARS = 20   # ignore leftover pieces with almost no content


def _make_chunk(path, lines, start, end, symbol):
    """Build one chunk from lines start..end (line numbers start at 1)."""
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "symbol": symbol,
        "text": "\n".join(lines[start - 1:end]),
    }


def _split_by_window(path, lines, start, end, symbol):
    """Cut lines start..end into overlapping windows of at most MAX_CHUNK_LINES."""
    chunks = []
    step = MAX_CHUNK_LINES - OVERLAP_LINES
    current = start
    while current <= end:
        window_end = min(current + MAX_CHUNK_LINES - 1, end)
        chunks.append(_make_chunk(path, lines, current, window_end, symbol))
        if window_end == end:
            break
        current += step
    return chunks


def _node_range(node):
    """First and last line of a function/class, including its decorators."""
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(start, node.decorator_list[0].lineno)
    return start, node.end_lineno


def _chunks_for_node(path, lines, node, prefix=""):
    """Make chunks for one function or class."""
    start, end = _node_range(node)
    symbol = prefix + node.name

    # Short enough: keep the whole thing as one chunk.
    if end - start + 1 <= MAX_CHUNK_LINES:
        return [_make_chunk(path, lines, start, end, symbol)]

    # A long class: header chunk, then each method separately.
    if isinstance(node, ast.ClassDef):
        members = [
            n for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if members:
            chunks = []
            first_start = _node_range(members[0])[0]
            if first_start > start:
                chunks.extend(_split_by_window(path, lines, start, first_start - 1, symbol))
            for member in members:
                chunks.extend(_chunks_for_node(path, lines, member, prefix=symbol + "."))
            return chunks

    # A long function (or a long class with no methods): overlapping windows.
    return _split_by_window(path, lines, start, end, symbol)


def _leftover_chunks(path, lines, chunks):
    """Chunks for lines not covered yet: imports, constants, main blocks."""
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk["start_line"], chunk["end_line"] + 1))

    result = []
    run_start = None
    # Go one line past the end so the last run is always closed.
    for number in range(1, len(lines) + 2):
        is_free = number <= len(lines) and number not in covered
        if is_free and run_start is None:
            run_start = number
        elif not is_free and run_start is not None:
            run_end = number - 1
            text = "\n".join(lines[run_start - 1:run_end]).strip()
            if len(text) >= MIN_CHUNK_CHARS:
                result.extend(_split_by_window(path, lines, run_start, run_end, "<module>"))
            run_start = None
    return result


def chunk_python_file(path, content):
    """Split one Python file into chunks."""
    lines = content.splitlines()
    if not lines:
        return []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(content)
    except (SyntaxError, ValueError):
        # Cannot understand the structure, so fall back to plain windows.
        return _split_by_window(path, lines, 1, len(lines), None)

    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunks.extend(_chunks_for_node(path, lines, node))

    chunks.extend(_leftover_chunks(path, lines, chunks))
    chunks.sort(key=lambda c: c["start_line"])
    return chunks


def chunk_documents(documents):
    """Chunk every document ({"path", "content"} dicts) and give each chunk an id."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_python_file(doc["path"], doc["content"]))
    for number, chunk in enumerate(all_chunks):
        chunk["id"] = number
    return all_chunks


if __name__ == "__main__":
    from pathlib import Path

    # Build a small sample file that exercises every rule.
    long_function = "def long_function():\n" + "\n".join(
        "    value_" + str(i) + " = " + str(i) for i in range(70)
    )
    big_class = 'class Big:\n    """Docstring of Big."""\n\n' + "\n\n".join(
        "    def method_" + str(m) + "(self):\n"
        + "\n".join("        v" + str(j) + " = " + str(j) for j in range(30))
        for m in range(3)
    )
    sample = (
        '"""Sample module."""\n'
        "import os\n\n"
        "LIMIT = 10\n\n\n"
        "def short_function(x):\n"
        '    """Return x doubled."""\n'
        "    return x * 2\n\n\n"
        "class Small:\n"
        "    def a(self):\n"
        "        return 1\n\n\n"
        + long_function + "\n\n\n"
        + big_class + "\n"
    )

    chunks = chunk_python_file("sample.py", sample)
    symbols = [c["symbol"] for c in chunks]

    def find(symbol):
        return [c for c in chunks if c["symbol"] == symbol]

    checks = [
        ("short function is one 3-line chunk",
         len(find("short_function")) == 1
         and find("short_function")[0]["end_line"] - find("short_function")[0]["start_line"] == 2),
        ("small class is one chunk", len(find("Small")) == 1),
        ("long function is split into windows", len(find("long_function")) >= 2),
        ("big class is split into its methods",
         all(("Big.method_" + str(m)) in symbols for m in range(3))),
        ("leftover imports/constants become <module>",
         any("LIMIT" in c["text"] for c in find("<module>"))),
        ("no chunk is longer than MAX_CHUNK_LINES",
         all(c["end_line"] - c["start_line"] + 1 <= MAX_CHUNK_LINES for c in chunks)),
        ("broken code falls back to a plain chunk",
         len(chunk_python_file("broken.py", "def broken(:\n    pass\n")) == 1),
    ]
    for label, passed in checks:
        print("PASS" if passed else "FAIL", "-", label)

    print()
    print("Chunks of the sample file:")
    for c in chunks:
        print("  ", str(c["symbol"]).ljust(15), "lines", c["start_line"], "-", c["end_line"])

    # Chunk this project's own files (no internet needed).
    print()
    print("Chunking this project's own files:")
    documents = []
    for file_path in sorted(Path(__file__).parent.glob("*.py")):
        documents.append(
            {"path": "app/" + file_path.name, "content": file_path.read_text(encoding="utf-8")}
        )
    all_chunks = chunk_documents(documents)
    for doc in documents:
        count = len([c for c in all_chunks if c["path"] == doc["path"]])
        print("  ", doc["path"].ljust(26), count, "chunks")
    print("Total chunks:", len(all_chunks))