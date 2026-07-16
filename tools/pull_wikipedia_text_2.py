#!/usr/bin/env python3
"""Pull punctuated Wikipedia text into raw/all.txt for training data.

This is a faster, more polite replacement for pull_wikipedia_text.py.

The original script makes separate API requests for page text and outgoing
links. This version batches page titles and fetches extracts plus links in the
same query. It still sends API requests serially by default, uses maxlag, and
honors Retry-After so the speedup comes from fewer requests instead of noisy
parallel traffic.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_BEGIN_AT = "Việt Nam"
DEFAULT_LANG = "vi"
DEFAULT_OUTPUT = Path("raw/all.txt")
DEFAULT_FLUSH_BYTES = 1024 * 1024
DEFAULT_USER_AGENT = (
    "Edge-Punct-Casing-WikiPull/2.0 "
    "(local training data script; https://www.wikipedia.org/) "
    "Python-urllib"
)

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")
LANG_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")


@dataclass
class PageBundle:
    title: str
    depth: int = 0
    extract: str = ""
    links: List[str] = field(default_factory=list)
    missing: bool = False
    _seen_links: Set[str] = field(default_factory=set, repr=False)

    def add_links(self, links: Iterable[str]) -> None:
        for link in links:
            key = link.casefold()
            if key in self._seen_links:
                continue
            self.links.append(link)
            self._seen_links.add(key)


class WikiAPIError(RuntimeError):
    """Raised when the MediaWiki API returns a non-retryable error."""


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("--overwritten must be True or False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull text from Wikipedia pages and write one sentence per line to "
            "raw/all.txt. This v2 crawler batches API queries and follows "
            "continuation data while keeping request pacing conservative."
        )
    )

    # Original compatibility flags.
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
        default=0.2,
        help=(
            "Minimum seconds between API requests. Default: 0.2. "
            "Use a larger value for very long crawls."
        ),
    )

    # New conservative tuning flags.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Page titles per query batch. Default: 20. Maximum: 50.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry attempts for transient API/network errors. Default: 5.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.7,
        help="Exponential retry multiplier. Default: 1.7.",
    )
    parser.add_argument(
        "--maxlag",
        type=int,
        default=5,
        help="MediaWiki maxlag value for polite background jobs. Default: 5.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=(
            "HTTP User-Agent. For large crawls, include real contact info so "
            "Wikimedia operators can reach you."
        ),
    )
    parser.add_argument(
        "--flush-mb",
        type=float,
        default=DEFAULT_FLUSH_BYTES / (1024 * 1024),
        help=(
            "Buffered output size in MB before flushing to disk. Default: 1. "
            "Larger values reduce file opens/writes but keep more text in RAM."
        ),
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
    if not LANG_RE.fullmatch(args.lang):
        parser.error("--lang must be a simple Wikipedia language code, e.g. vi, en, zh, be-x-old")
    if args.request_delay < 0:
        parser.error("--request-delay must be greater than or equal to 0")
    if not 1 <= args.batch_size <= 50:
        parser.error("--batch-size must be between 1 and 50")
    if args.max_retries < 0:
        parser.error("--max-retries must be greater than or equal to 0")
    if args.retry_backoff <= 1:
        parser.error("--retry-backoff must be greater than 1")
    if args.maxlag < 0:
        parser.error("--maxlag must be greater than or equal to 0")
    if args.flush_mb <= 0:
        parser.error("--flush-mb must be greater than 0")

    return args


def is_good_sentence(sentence: str) -> bool:
    if not sentence.endswith((".", "!", "?")):
        return False
    if len(sentence.split()) < 3:
        return False
    if sentence.startswith(("==", "*", "#")):
        return False
    return True


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


def ensure_append_starts_on_new_line(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("rb") as output_file:
        output_file.seek(-1, 2)
        last_byte = output_file.read(1)

    if last_byte != b"\n":
        with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
            output_file.write("\n")


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def bytes_to_mb(size: int) -> float:
    return size / (1024 * 1024)


class BufferedSentenceWriter:
    def __init__(self, output_path: Path, flush_bytes: int) -> None:
        self.output_path = output_path
        self.flush_bytes = flush_bytes
        self.buffer: List[bytes] = []
        self.buffered_bytes = 0
        self.flushed_bytes = output_path.stat().st_size if output_path.exists() else 0

    def write_page(self, text: str) -> int:
        sentences = list(iter_sentences(text))
        if not sentences:
            return 0

        data = ("\n".join(sentences) + "\n").encode("utf-8")
        self.buffer.append(data)
        self.buffered_bytes += len(data)

        if self.buffered_bytes >= self.flush_bytes:
            self.flush()

        return len(sentences)

    def flush(self) -> None:
        if not self.buffer:
            return

        with self.output_path.open("ab") as output_file:
            output_file.writelines(self.buffer)

        self.flushed_bytes += self.buffered_bytes
        self.buffer.clear()
        self.buffered_bytes = 0

    def size_mb(self) -> float:
        return bytes_to_mb(self.flushed_bytes + self.buffered_bytes)


def should_pull_depth(depth: int, max_depth: Optional[int]) -> bool:
    return max_depth is None or depth <= max_depth


def should_enqueue_children(depth: int, max_depth: Optional[int]) -> bool:
    return max_depth is None or depth < max_depth


def batched_queue_pop(
    queue: Deque[Tuple[str, int]],
    batch_size: int,
    max_depth: Optional[int],
) -> List[Tuple[str, int]]:
    batch: List[Tuple[str, int]] = []
    while queue and len(batch) < batch_size:
        title, depth = queue.popleft()
        if should_pull_depth(depth, max_depth):
            batch.append((title, depth))
    return batch


class PoliteWikiClient:
    def __init__(
        self,
        lang: str,
        user_agent: str,
        request_delay: float,
        max_retries: int,
        retry_backoff: float,
        maxlag: int,
    ) -> None:
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.maxlag = maxlag
        self.last_request_at = 0.0

    def fetch_pages(
        self,
        requested: List[Tuple[str, int]],
        fetch_links: bool,
    ) -> List[PageBundle]:
        titles = [title for title, _ in requested]
        requested_depths = {title.casefold(): depth for title, depth in requested}
        title_key_aliases = dict((title.casefold(), title.casefold()) for title in titles)
        pages_by_key: Dict[str, PageBundle] = {}
        continue_params: Dict[str, str] = {"continue": ""}
        seen_continue_tokens: Set[Tuple[Tuple[str, str], ...]] = set()

        while True:
            params = {
                "action": "query",
                "prop": "extracts|links" if fetch_links else "extracts",
                "titles": "|".join(titles),
                "redirects": "1",
                "explaintext": "1",
                "exsectionformat": "plain",
                **continue_params,
            }
            if fetch_links:
                params.update(
                    {
                        "plnamespace": "0",
                        "pllimit": "max",
                    }
                )
            data = self.api_request(params)
            query = data.get("query", {})
            self._record_title_aliases(query, requested_depths, title_key_aliases)

            for page in query.get("pages", []):
                title = page.get("title", "")
                if not title:
                    continue

                key = title.casefold()
                bundle = pages_by_key.get(key)
                if bundle is None:
                    bundle = PageBundle(title=title, missing=bool(page.get("missing")))
                    pages_by_key[key] = bundle

                if page.get("extract") and not bundle.extract:
                    bundle.extract = page.get("extract", "")

                bundle.add_links(
                    link["title"]
                    for link in page.get("links", [])
                    if "title" in link
                )

            next_continue = data.get("continue")
            if not next_continue:
                break

            token = tuple(sorted((str(k), str(v)) for k, v in next_continue.items()))
            if token in seen_continue_tokens:
                raise WikiAPIError("API returned a repeated continuation token")
            seen_continue_tokens.add(token)
            continue_params = {str(k): str(v) for k, v in next_continue.items()}

        depth_by_canonical_key = self._depths_for_canonical_titles(
            requested_depths,
            title_key_aliases,
        )
        fallback_depth = min(requested_depths.values()) if requested_depths else 0
        for key, bundle in pages_by_key.items():
            bundle.depth = depth_by_canonical_key.get(key, fallback_depth)

        ordered: List[PageBundle] = []
        emitted: Set[str] = set()
        for title in titles:
            requested_key = title.casefold()
            canonical_key = title_key_aliases.get(requested_key, requested_key)
            bundle = pages_by_key.get(canonical_key)
            if bundle is None or canonical_key in emitted:
                continue
            ordered.append(bundle)
            emitted.add(canonical_key)

        for key, bundle in pages_by_key.items():
            if key not in emitted:
                ordered.append(bundle)

        return ordered

    def api_request(self, params: Dict[str, str]) -> Dict:
        query = {
            "format": "json",
            "formatversion": "2",
            "maxlag": str(self.maxlag),
            **params,
        }

        for attempt in range(self.max_retries + 1):
            self._pace_request()
            try:
                data = self._open_json(query)
                error = data.get("error")
                if error:
                    if error.get("code") == "maxlag":
                        if attempt == self.max_retries:
                            raise WikiAPIError(f"API maxlag persisted: {error.get('info')}")
                        delay = self._maxlag_delay(error, attempt)
                        print(f"API maxlag reported; sleeping {delay:.1f}s", file=sys.stderr)
                        time.sleep(delay)
                        continue
                    raise WikiAPIError(f"API error {error.get('code')}: {error.get('info')}")
                return data
            except urllib.error.HTTPError as exc:
                if not self._is_retryable_status(exc.code) or attempt == self.max_retries:
                    raise
                delay = self._retry_delay(attempt, retry_after=exc.headers.get("Retry-After"))
                print(f"HTTP {exc.code}; retrying in {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == self.max_retries:
                    raise
                delay = self._retry_delay(attempt)
                print(f"Request failed: {exc}; retrying in {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)

        raise RuntimeError("unreachable retry state")

    def _pace_request(self) -> None:
        if self.request_delay <= 0:
            return
        elapsed = time.monotonic() - self.last_request_at
        remaining = self.request_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_request_at = time.monotonic()

    def _open_json(self, query: Dict[str, str]) -> Dict:
        encoded = urllib.parse.urlencode(query)
        url = f"{self.base_url}?{encoded}"
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        if len(url) > 7000:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            request = urllib.request.Request(
                self.base_url,
                data=encoded.encode("utf-8"),
                headers=headers,
            )
        else:
            request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _retry_delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                return max(float(retry_after), self.request_delay)
            except ValueError:
                pass
        return max(self.retry_backoff ** attempt, self.request_delay) + random.uniform(0, 0.25)

    def _maxlag_delay(self, error: Dict, attempt: int) -> float:
        retry_after = error.get("retry-after")
        if retry_after is not None:
            try:
                return max(float(retry_after), self.request_delay)
            except (TypeError, ValueError):
                pass
        lag = error.get("lag")
        try:
            return max(float(lag), self.retry_backoff ** attempt, self.request_delay)
        except (TypeError, ValueError):
            return self._retry_delay(attempt)

    @staticmethod
    def _is_retryable_status(status: int) -> bool:
        return status in {429, 500, 502, 503, 504}

    @staticmethod
    def _record_title_aliases(
        query: Dict,
        requested_depths: Dict[str, int],
        title_key_aliases: Dict[str, str],
    ) -> None:
        for normalized in query.get("normalized", []):
            source = normalized.get("from")
            target = normalized.get("to")
            if source and target and source.casefold() in requested_depths:
                title_key_aliases[source.casefold()] = target.casefold()

        for redirect in query.get("redirects", []):
            source = redirect.get("from")
            target = redirect.get("to")
            if not source or not target:
                continue

            source_key = source.casefold()
            target_key = target.casefold()
            for requested_key, current_key in list(title_key_aliases.items()):
                if requested_key == source_key or current_key == source_key:
                    title_key_aliases[requested_key] = target_key
            if source_key in requested_depths:
                title_key_aliases[source_key] = target_key

    @staticmethod
    def _depths_for_canonical_titles(
        requested_depths: Dict[str, int],
        title_key_aliases: Dict[str, str],
    ) -> Dict[str, int]:
        depth_by_canonical_key: Dict[str, int] = {}
        for requested_key, depth in requested_depths.items():
            canonical_key = title_key_aliases.get(requested_key, requested_key)
            current_depth = depth_by_canonical_key.get(canonical_key)
            if current_depth is None or depth < current_depth:
                depth_by_canonical_key[canonical_key] = depth
        return depth_by_canonical_key


def run(args: argparse.Namespace) -> None:
    started_at = time.monotonic()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwritten:
        output_path.write_text("", encoding="utf-8")
    else:
        ensure_append_starts_on_new_line(output_path)

    writer = BufferedSentenceWriter(
        output_path=output_path,
        flush_bytes=max(1, int(args.flush_mb * 1024 * 1024)),
    )

    client = PoliteWikiClient(
        lang=args.lang,
        user_agent=args.user_agent,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        maxlag=args.maxlag,
    )

    visited: Set[str] = set()
    queued: Set[str] = {args.begin_at.casefold()}
    queue: Deque[Tuple[str, int]] = deque([(args.begin_at, 0)])
    pages_written = 0
    sentences_written = 0

    print(f"Writing to: {output_path}")
    print(f"Language: {args.lang}")
    print(f"Starting page: {args.begin_at}")
    print(f"Batch size: {args.batch_size}")
    print(f"Request delay: {args.request_delay:g}s")
    print(f"Flush threshold: {args.flush_mb:g} MB")

    try:
        while queue:
            if args.max_pages is not None and pages_written >= args.max_pages:
                print(f"Stopping: reached --max-pages {args.max_pages}")
                break
            if args.size is not None and writer.size_mb() > args.size:
                print(f"Stopping: output is larger than --size {args.size:g} MB")
                break

            batch = batched_queue_pop(queue, args.batch_size, args.max_depth)
            if not batch:
                continue

            depth_by_requested_key = {title.casefold(): depth for title, depth in batch}
            print(
                "Pulling batch: "
                f"{len(batch)} pages, depths {min(depth_by_requested_key.values())}-"
                f"{max(depth_by_requested_key.values())}, queue={len(queue)}"
            )

            try:
                fetch_links = any(
                    should_enqueue_children(depth, args.max_depth)
                    for _, depth in batch
                )
                pages = client.fetch_pages(batch, fetch_links=fetch_links)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, WikiAPIError) as exc:
                titles = ", ".join(title for title, _ in batch[:3])
                if len(batch) > 3:
                    titles += ", ..."
                print(f"Failed batch [{titles}]: {exc}", file=sys.stderr)
                continue

            for page in pages:
                normalized_title = page.title.casefold()
                if normalized_title in visited:
                    continue
                visited.add(normalized_title)

                depth = page.depth

                if page.missing:
                    print(f"Skipped missing page: {page.title}")
                    continue

                sentence_count = writer.write_page(page.extract)
                current_size_mb = writer.size_mb()
                if sentence_count:
                    pages_written += 1
                    sentences_written += sentence_count
                    print(
                        f"Pushed: {page.title} ({sentence_count} sentences, "
                        f"{current_size_mb:.3f} MB total)"
                    )
                else:
                    print(f"Skipped empty/unsupported page: {page.title}")

                if args.size is not None and current_size_mb > args.size:
                    print(f"Stopping: output is larger than --size {args.size:g} MB")
                    queue.clear()
                    break
                if args.max_pages is not None and pages_written >= args.max_pages:
                    print(f"Stopping: reached --max-pages {args.max_pages}")
                    queue.clear()
                    break

                if should_enqueue_children(depth, args.max_depth):
                    for link_title in page.links:
                        link_key = link_title.casefold()
                        if link_key not in visited and link_key not in queued:
                            queue.append((link_title, depth + 1))
                            queued.add(link_key)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        writer.flush()
        elapsed = time.monotonic() - started_at
        print(f"Pages pushed: {pages_written}")
        print(f"Sentences pushed: {sentences_written}")
        print(f"Final size: {file_size_mb(output_path):.3f} MB")
        print(f"Total time taken: {elapsed:.2f} seconds")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
