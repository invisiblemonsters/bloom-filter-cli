"""bloom.py - a small, dependency-free Bloom filter with a JSON file format.

A Bloom filter answers "have I seen this before?" using a few bits per item.
It never returns a false negative: if it says "definitely not", the item was
never added. It may return a false positive at a rate you choose up front.

Usage:
    python bloom.py build --items things.txt --capacity 1000 --out demo.bloom
    python bloom.py check --filter demo.bloom --items queries.txt
    python bloom.py info  --filter demo.bloom
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys

LN2 = math.log(2.0)
FORMAT = "bloom-1"


def optimal_params(capacity: int, fp_rate: float = 0.01) -> tuple[int, int]:
    """Return (bit_count, hash_count) for the given capacity and target rate."""
    if capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if not (0.0 < fp_rate < 1.0):
        raise ValueError("fp_rate must be strictly between 0 and 1")
    bits = max(1, math.ceil(-capacity * math.log(fp_rate) / (LN2 ** 2)))
    hashes = max(1, min(bits, round(bits / capacity * LN2)))
    return bits, hashes


class BloomFilter:
    """Fixed-size Bloom filter. Items are str (or bytes) and hashed with SHA-256."""

    def __init__(self, capacity: int, fp_rate: float = 0.01,
                 *, m: int | None = None, k: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not (0.0 < fp_rate < 1.0):
            raise ValueError("fp_rate must be strictly between 0 and 1")
        self.capacity = int(capacity)
        self.fp_rate = float(fp_rate)
        if m is None and k is None:
            self.m, self.k = optimal_params(self.capacity, self.fp_rate)
        elif m is None or k is None:
            raise ValueError("m and k must be given together")
        else:
            self.m, self.k = int(m), int(k)
            if self.m < 1 or self.k < 1 or self.k > self.m:
                raise ValueError("invalid bit or hash count")
        self._nbytes = (self.m + 7) // 8
        self._bits = bytearray(self._nbytes)
        self.items_added = 0

    # -- hashing ---------------------------------------------------------

    def _indexes(self, item) -> list[int]:
        data = item.encode("utf-8") if isinstance(item, str) else bytes(item)
        digest = hashlib.sha256(data).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1  # odd => never a zero step
        # Enhanced double hashing (Dillinger & Manolios): spreads k indexes
        # without needing k independent hash functions.
        return [(h1 + i * h2 + i * i) % self.m for i in range(self.k)]

    def _bit(self, index: int) -> int:
        return (self._bits[index >> 3] >> (index & 7)) & 1

    # -- public API ------------------------------------------------------

    def add(self, item) -> "BloomFilter":
        for index in self._indexes(item):
            self._bits[index >> 3] |= 1 << (index & 7)
        self.items_added += 1
        return self

    def extend(self, items) -> "BloomFilter":
        for item in items:
            self.add(item)
        return self

    def __contains__(self, item) -> bool:
        return all(self._bit(index) for index in self._indexes(item))

    def set_bits(self) -> int:
        return sum(bin(byte).count("1") for byte in self._bits)

    def fill_ratio(self) -> float:
        return self.set_bits() / self.m

    def theoretical_fp_rate(self) -> float:
        """False-positive rate implied by the current fill (not the target)."""
        return self.fill_ratio() ** self.k

    def estimated_items(self) -> float:
        """Invert the fill ratio to estimate how many distinct items were added."""
        fill = self.fill_ratio()
        if fill >= 1.0:
            return float("inf")
        if fill <= 0.0:
            return 0.0
        return -(self.m / self.k) * math.log(1.0 - fill)

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "m": self.m,
            "k": self.k,
            "capacity": self.capacity,
            "fp_rate": self.fp_rate,
            "items_added": self.items_added,
            "bits": base64.b64encode(bytes(self._bits)).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BloomFilter":
        if not isinstance(data, dict):
            raise ValueError("filter data must be a JSON object")
        if data.get("format") != FORMAT:
            raise ValueError("unsupported filter format")
        try:
            m = int(data["m"])
            k = int(data["k"])
            capacity = int(data["capacity"])
            fp_rate = float(data["fp_rate"])
            raw = base64.b64decode(data["bits"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed filter data: {exc}") from None
        obj = cls(capacity, fp_rate, m=m, k=k)
        if len(raw) != obj._nbytes:
            raise ValueError("bit array length does not match m")
        obj._bits = bytearray(raw)
        obj.items_added = int(data.get("items_added", 0))
        return obj

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path) -> "BloomFilter":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# -- CLI -----------------------------------------------------------------

def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _emit(text: str) -> None:
    # Write through the binary buffer: text-mode stdout rewrites "\n" as
    # "\r\n" on Windows, which would break byte-exact output.
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


def cmd_build(args) -> int:
    items = _read_lines(args.items)
    bloom = BloomFilter(args.capacity, args.fp_rate)
    bloom.extend(items)
    bloom.save(args.out)
    _emit(
        f"built {args.out}: {len(items)} items, {bloom.m} bits, {bloom.k} hashes, "
        f"{bloom._nbytes} bytes on disk\n"
    )
    return 0


def cmd_check(args) -> int:
    bloom = BloomFilter.load(args.filter)
    queries = _read_lines(args.items)
    hits = 0
    chunks = []
    for query in queries:
        hit = query in bloom
        hits += hit
        chunks.append(("+ " if hit else "- ") + query + "\n")
    _emit("".join(chunks))
    _emit(f"{hits} of {len(queries)} items probably present\n")
    # grep-like: 1 means at least one item is definitely not in the filter.
    return 0 if hits == len(queries) else 1


def cmd_info(args) -> int:
    bloom = BloomFilter.load(args.filter)
    _emit(
        f"bits:          {bloom.m}\n"
        f"hash count:    {bloom.k}\n"
        f"bytes on disk: {bloom._nbytes}\n"
        f"target rate:   {bloom.fp_rate:g}\n"
        f"items added:   {bloom.items_added}\n"
        f"fill ratio:    {bloom.fill_ratio():.4f}\n"
        f"current FPR:   {bloom.theoretical_fp_rate():.6f}\n"
        f"est. distinct: {bloom.estimated_items():.1f}\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bloom.py",
        description="Build and query a Bloom filter stored as JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="create a filter from a line-delimited file")
    build.add_argument("--items", required=True, help="input file, one item per line")
    build.add_argument("--out", required=True, help="output .bloom file")
    build.add_argument("--capacity", type=int, default=1000,
                       help="expected number of distinct items (default 1000)")
    build.add_argument("--fp-rate", type=float, default=0.01,
                       help="target false-positive rate (default 0.01)")
    build.set_defaults(func=cmd_build)

    check = sub.add_parser("check", help="test items against a filter")
    check.add_argument("--filter", required=True)
    check.add_argument("--items", required=True, help="file of items to test")
    check.set_defaults(func=cmd_check)

    info = sub.add_parser("info", help="print filter statistics")
    info.add_argument("--filter", required=True)
    info.set_defaults(func=cmd_info)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
