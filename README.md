# bloom-filter-cli

A tiny, dependency-free Bloom filter in Python — plus a CLI that saves a
filter to a JSON file and runs membership checks against it.

A Bloom filter is a compact set: a few bits per item instead of the item
itself. Lookups have **no false negatives** — if it says an item was never
added, that is certain — but a lookup can occasionally say "probably
present" for something that was never added. You pick the false-positive
rate up front, and the filter sizes itself to hit it.

## Usage

```bash
# Build a filter over a line-delimited file.
python bloom.py build --items domains.txt --capacity 10000 --fp-rate 0.01 --out seen.bloom

# Check a list of candidates (- = definitely absent, + = probably present).
python bloom.py check --filter seen.bloom --items candidates.txt

# Inspect the saved filter.
python bloom.py info --filter seen.bloom
```

Example:

```
$ python bloom.py build --items domains.txt --capacity 10000 --out seen.bloom
built seen.bloom: 10000 items, 95851 bits, 7 hashes, 11982 bytes on disk

$ python bloom.py check --filter seen.bloom --items candidates.txt
+ example.com
- new-domain.example
1 of 2 items probably present
```

`check` exits `0` when every query was a possible hit and `1` if any query was
definitely absent (grep-like), so it composes with shell `if`. File and format
errors exit `2`.

## Library use

```python
from bloom import BloomFilter

bf = BloomFilter(capacity=1000, fp_rate=0.01)
bf.add("apple")
bf.extend(["banana", "cherry"])

"apple" in bf          # True  (never a false negative)
bf.fill_ratio()        # fraction of bits set
bf.estimated_items()   # distinct items inferred from the fill
bf.save("seen.bloom")
BloomFilter.load("seen.bloom")
```

Items may be `str` (hashed as UTF-8) or `bytes`. Indexes come from one
SHA-256 digest with enhanced double hashing, so builds are byte-for-byte
deterministic.

## Accuracy

On a 1,000-item filter built for a 1% target rate, 50,000 absent queries
produced 526 false positives — **1.05% observed vs 1.00% requested**. The
`info` command reports the rate implied by the current fill, which creeps up
as you add more items than the capacity you planned for: size capacity to the
real item count if you care about the number.

The JSON file is not compressed; a 1,000-item filter is about 1.2 KB and a
10,000-item filter about 12 KB, because it uses ~9.6 bits per item.

## Tests

```bash
python -m unittest test_bloom -v
```

19 tests: parameter math, a no-false-negative sweep across filter sizes,
measured false-positive rate, determinism, fill/estimate behavior, JSON
round-trips, malformed-input rejection, and byte-exact CLI output.

## License

MIT
