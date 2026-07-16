#!/usr/bin/env python3
"""Pull punctuated Wikipedia text into raw/all.txt for training data.

The crawler walks outgoing article links breadth-first. It keeps an in-memory
cache of page titles visited during the current run so the same page is not
written twice.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_BEGIN_AT = "Việt Nam"
DEFAULT_LANG = "vi"
DEFAULT_OUTPUT = Path("raw/all.txt")
USER_AGENT = "Edge-Punct-Casing data puller (https://www.wikipedia.org/)"
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("--overwritten must be True or False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull text from Wikipedia pages and write one sentence per line to raw/all.txt."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum outgoing-link search depth. Use 0 to pull only --begin-at.",
    )
    parser.add_argument(
        "--size",
        type=float,
        help="Stop after raw/all.txt first becomes larger than this size in megabytes.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Stop after this many unique Wikipedia pages have been written.",
    )
    parser.add_argument(
        "--begin-at",
        default=DEFAULT_BEGIN_AT,
        help=f"Root Wikipedia page title. Default: {DEFAULT_BEGIN_AT!r}.",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"Wikipedia language code to pull from. Default: {DEFAULT_LANG!r}.",
    )
    parser.add_argument(
        "--overwritten",
        type=parse_bool,
        default=True,
        metavar="True|False",
        help="Overwrite raw/all.txt before pulling. Use False to append. Default: True.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Output text file. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="Seconds to sleep between Wikipedia API requests. Default: 0.1.",
    )
    args = parser.parse_args()

    if args.max_depth is None and args.size is None and args.max_pages is None:
        parser.error("set at least one stopping condition: --max-depth, --size, or --max-pages")
    if args.max_depth is not None and args.max_depth < 0:
        parser.error("--max-depth must be greater than or equal to 0")
    if args.size is not None and args.size <= 0:
        parser.error("--size must be greater than 0")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be greater than 0")
    if args.request_delay < 0:
        parser.error("--request-delay must be greater than or equal to 0")

    return args


def api_request(lang: str, params: Dict[str, str]) -> Dict:
    base_url = f"https://{lang}.wikipedia.org/w/api.php"
    query = {
        "format": "json",
        "formatversion": "2",
        **params,
    }
    url = f"{base_url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_page_text(lang: str, title: str) -> Tuple[str, str]:
    data = api_request(
        lang,
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "redirects": "1",
            "explaintext": "1",
            "exsectionformat": "plain",
        },
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return title, ""
    page = pages[0]
    return page.get("title", title), page.get("extract", "")


def fetch_outgoing_links(lang: str, title: str, request_delay: float) -> List[str]:
    links: List[str] = []
    plcontinue: Optional[str] = None

    while True:
        params = {
            "action": "query",
            "prop": "links",
            "titles": title,
            "redirects": "1",
            "plnamespace": "0",
            "pllimit": "max",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        data = api_request(lang, params)
        pages = data.get("query", {}).get("pages", [])
        if pages and not pages[0].get("missing"):
            links.extend(link["title"] for link in pages[0].get("links", []) if "title" in link)

        plcontinue = data.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break
        if request_delay:
            time.sleep(request_delay)

    return links


def iter_sentences(text: str) -> Iterable[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for paragraph in text.split("\n"):
        paragraph = WHITESPACE_RE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        for sentence in SENTENCE_END_RE.split(paragraph):
            sentence = sentence.strip()
            if is_good_sentence(sentence):
                yield sentence


def is_good_sentence(sentence: str) -> bool:
    if not sentence.endswith((".", "!", "?")):
        return False
    if len(sentence.split()) < 3:
        return False
    if sentence.startswith(("==", "*", "#")):
        return False
    return True


def ensure_append_starts_on_new_line(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("rb") as output_file:
        output_file.seek(-1, 2)
        last_byte = output_file.read(1)

    if last_byte != b"\n":
        with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
            output_file.write("\n")


def write_page(output_path: Path, text: str) -> int:
    sentences = list(iter_sentences(text))
    if not sentences:
        return 0

    with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
        for sentence in sentences:
            output_file.write(sentence + "\n")

    return len(sentences)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def should_pull_depth(depth: int, max_depth: Optional[int]) -> bool:
    return max_depth is None or depth <= max_depth


def should_enqueue_children(depth: int, max_depth: Optional[int]) -> bool:
    return max_depth is None or depth < max_depth


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    output_path: Path = args.output
    size_limit_mb: Optional[float] = args.size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwritten:
        output_path.write_text("", encoding="utf-8")
    else:
        ensure_append_starts_on_new_line(output_path)

    visited: Set[str] = set()
    queued: Set[str] = {args.begin_at.casefold()}
    queue: Deque[Tuple[str, int]] = deque([(args.begin_at, 0)])
    pages_written = 0
    sentences_written = 0

    print(f"Writing to: {output_path}")
    print(f"Language: {args.lang}")
    print(f"Starting page: {args.begin_at}")

    try:
        while queue:
            requested_title, depth = queue.popleft()
            if not should_pull_depth(depth, args.max_depth):
                continue

            print(f"Pulling: {requested_title} (depth {depth})")

            try:
                title, text = fetch_page_text(args.lang, requested_title)
                if args.request_delay:
                    time.sleep(args.request_delay)
                normalized_title = title.casefold()
                if normalized_title in visited:
                    print(f"Skipped duplicate: {title}")
                    continue
                visited.add(normalized_title)

                sentence_count = write_page(output_path, text)
                if sentence_count:
                    pages_written += 1
                    sentences_written += sentence_count
                    current_size_mb = file_size_mb(output_path)
                    print(
                        f"Pushed: {title} ({sentence_count} sentences, "
                        f"{current_size_mb:.3f} MB total)"
                    )
                else:
                    current_size_mb = file_size_mb(output_path)
                    print(f"Skipped empty/unsupported page: {title}")

                if size_limit_mb is not None and current_size_mb > size_limit_mb:
                    print(f"Stopping: output is larger than --size {size_limit_mb:g} MB")
                    break
                if args.max_pages is not None and pages_written >= args.max_pages:
                    print(f"Stopping: reached --max-pages {args.max_pages}")
                    break

                if should_enqueue_children(depth, args.max_depth):
                    for link_title in fetch_outgoing_links(args.lang, title, args.request_delay):
                        link_key = link_title.casefold()
                        if link_key not in visited and link_key not in queued:
                            queue.append((link_title, depth + 1))
                            queued.add(link_key)
                    if args.request_delay:
                        time.sleep(args.request_delay)

            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"Failed: {requested_title} ({exc})", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        elapsed = time.monotonic() - started_at
        print(f"Pages pushed: {pages_written}")
        print(f"Sentences pushed: {sentences_written}")
        print(f"Final size: {file_size_mb(output_path):.3f} MB")
        print(f"Total time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
