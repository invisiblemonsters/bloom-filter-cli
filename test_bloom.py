"""Tests for bloom.py: parameters, no false negatives, FPR, IO and CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import bloom

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "bloom.py")


def run_cli(*args):
    """Run the CLI and return (returncode, stdout bytes, stderr bytes)."""
    proc = subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,  # bytes, not text: output must be byte-exact
        cwd=HERE,
    )
    return proc.returncode, proc.stdout, proc.stderr


class ParamTests(unittest.TestCase):
    def test_known_parameters(self):
        m, k = bloom.optimal_params(1000, 0.01)
        self.assertEqual(m, 9586)
        self.assertEqual(k, 7)

    def test_parameters_scale_with_rate(self):
        tight = bloom.optimal_params(500, 0.001)
        loose = bloom.optimal_params(500, 0.1)
        self.assertGreater(tight[0], loose[0])
        self.assertGreater(tight[1], loose[1])

    def test_invalid_inputs(self):
        for capacity, rate in [(0, 0.01), (-5, 0.01), (10, 0.0), (10, 1.0), (10, 2.5)]:
            with self.assertRaises(ValueError):
                bloom.optimal_params(capacity, rate)
            with self.assertRaises(ValueError):
                bloom.BloomFilter(capacity, rate)

    def test_partial_params_rejected(self):
        with self.assertRaises(ValueError):
            bloom.BloomFilter(10, 0.01, m=100)
        with self.assertRaises(ValueError):
            bloom.BloomFilter(10, 0.01, k=3)


class MembershipTests(unittest.TestCase):
    def test_no_false_negatives_sweep(self):
        # Every size, several seeds: anything added must always be reported.
        checked = 0
        for capacity in (1, 2, 7, 64, 500):
            for seed in range(5):
                items = [f"c{capacity}-s{seed}-item-{i}" for i in range(capacity)]
                bf = bloom.BloomFilter(capacity, 0.05)
                for item in items:
                    bf.add(item)
                for item in items:
                    self.assertIn(item, bf, (capacity, seed, item))
                    checked += 1
        self.assertEqual(checked, sum((1, 2, 7, 64, 500)) * 5)

    def test_empty_filter_never_matches(self):
        bf = bloom.BloomFilter(100, 0.01)
        self.assertEqual(bf.set_bits(), 0)
        self.assertEqual(bf.fill_ratio(), 0.0)
        self.assertEqual(bf.estimated_items(), 0.0)
        # With no bits set, every lookup must miss.
        self.assertNotIn("anything", bf)
        self.assertEqual(bf.theoretical_fp_rate(), 0.0)

    def test_bytes_items_supported(self):
        bf = bloom.BloomFilter(50, 0.01)
        payload = b"\x00\xffbinary"
        bf.add(payload)
        self.assertIn(payload, bf)
        self.assertNotIn(b"definitely-absent-payload", bf)

    def test_items_added_counter(self):
        bf = bloom.BloomFilter(10, 0.01)
        bf.extend(["a", "b", "c"])
        self.assertEqual(bf.items_added, 3)

    def test_determinism(self):
        items = [f"determinism-{i}" for i in range(200)]
        one = bloom.BloomFilter(200, 0.01).extend(items)
        two = bloom.BloomFilter(200, 0.01).extend(items)
        self.assertEqual(bytes(one._bits), bytes(two._bits))
        self.assertEqual(one.to_dict()["bits"], two.to_dict()["bits"])


class RateTests(unittest.TestCase):
    def test_observed_false_positive_rate_is_close_to_target(self):
        # Measured once, not asserted tight: enhanced double hashing is a bit
        # worse than ideal independent hashes, so allow headroom above target.
        capacity = 1000
        bf = bloom.BloomFilter(capacity, 0.01).extend(
            [f"present-{i}" for i in range(capacity)]
        )
        probes = 50000
        false_positives = sum(
            1 for i in range(probes) if f"absent-{i}" in bf
        )
        observed = false_positives / probes
        self.assertLess(observed, 0.03)
        self.assertLess(bf.theoretical_fp_rate(), 0.02)

    def test_fill_grows_and_stays_in_range(self):
        bf = bloom.BloomFilter(500, 0.01)
        previous = 0.0
        for step in range(5):
            bf.extend([f"fill-{step}-{i}" for i in range(100)])
            current = bf.fill_ratio()
            self.assertGreater(current, previous)
            self.assertLessEqual(current, 1.0)
            previous = current

    def test_estimated_items_tracks_reality(self):
        for capacity in (100, 1000):
            bf = bloom.BloomFilter(capacity, 0.01).extend(
                [f"est-{capacity}-{i}" for i in range(capacity)]
            )
            estimate = bf.estimated_items()
            self.assertLess(abs(estimate - capacity) / capacity, 0.20,
                            (capacity, estimate))


class PersistenceTests(unittest.TestCase):
    def test_dict_round_trip(self):
        items = [f"rt-{i}" for i in range(300)]
        original = bloom.BloomFilter(300, 0.01).extend(items)
        clone = bloom.BloomFilter.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(clone.m, original.m)
        self.assertEqual(clone.k, original.k)
        self.assertEqual(clone.capacity, original.capacity)
        self.assertEqual(clone.fp_rate, original.fp_rate)
        self.assertEqual(clone.items_added, 300)
        self.assertEqual(bytes(clone._bits), bytes(original._bits))
        for item in items:
            self.assertIn(item, clone)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.bloom")
            bloom.BloomFilter(50, 0.02).extend(["alpha", "beta"]).save(path)
            loaded = bloom.BloomFilter.load(path)
            self.assertIn("alpha", loaded)
            self.assertIn("beta", loaded)
            self.assertNotIn("gamma", loaded)

    def test_malformed_data_rejected(self):
        good = bloom.BloomFilter(10, 0.01).to_dict()
        with self.assertRaises(ValueError):
            bloom.BloomFilter.from_dict({**good, "format": "nope"})
        with self.assertRaises(ValueError):
            bloom.BloomFilter.from_dict({k: v for k, v in good.items() if k != "bits"})
        with self.assertRaises(ValueError):
            bloom.BloomFilter.from_dict({**good, "bits": ""})  # wrong length
        with self.assertRaises(ValueError):
            bloom.BloomFilter.from_dict("not a dict")


class CliTests(unittest.TestCase):
    def _write(self, path, lines):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_build_check_info_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = os.path.join(tmp, "items.txt")
            queries = os.path.join(tmp, "queries.txt")
            filt = os.path.join(tmp, "demo.bloom")
            self._write(items, ["apple", "banana", "cherry"])
            self._write(queries, ["apple", "durian"])

            code, out, err = run_cli("build", "--items", items, "--out", filt,
                                     "--capacity", "100", "--fp-rate", "0.001")
            self.assertEqual(code, 0, err)
            self.assertIn(b"built", out)
            self.assertIn(b"1438 bits, 10 hashes", out)
            self.assertIn(b"3 items", out)
            self.assertTrue(os.path.exists(filt))

            code, out, err = run_cli("check", "--filter", filt, "--items", queries)
            # One hit and one definite miss: grep-like exit status 1.
            self.assertEqual(code, 1, err)
            lines = out.split(b"\n")
            self.assertEqual(lines[0], b"+ apple")
            self.assertEqual(lines[1], b"- durian")
            self.assertEqual(lines[2], b"1 of 2 items probably present")
            self.assertNotIn(b"\r", out)  # byte-exact LF output

            code, out, err = run_cli("info", "--filter", filt)
            self.assertEqual(code, 0, err)
            self.assertIn(b"fill ratio:", out)
            self.assertIn(b"items added:   3", out)

    def test_check_without_hits_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            filt = os.path.join(tmp, "f.bloom")
            queries = os.path.join(tmp, "q.txt")
            bloom.BloomFilter(50, 0.001).extend(["only-this"]).save(filt)
            self._write(queries, ["nope-1", "nope-2"])
            code, out, _ = run_cli("check", "--filter", filt, "--items", queries)
            self.assertEqual(code, 1)
            self.assertTrue(out.startswith(b"- nope-1\n- nope-2\n"))

    def test_check_all_hits_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            filt = os.path.join(tmp, "f.bloom")
            queries = os.path.join(tmp, "q.txt")
            bloom.BloomFilter(50, 0.001).extend(["only-this", "and-this"]).save(filt)
            self._write(queries, ["only-this", "and-this"])
            code, out, _ = run_cli("check", "--filter", filt, "--items", queries)
            self.assertEqual(code, 0)
            self.assertTrue(out.endswith(b"2 of 2 items probably present\n"))

    def test_missing_file_is_clean_error(self):
        code, out, err = run_cli("info", "--filter", "does-not-exist.bloom")
        self.assertEqual(code, 2)
        self.assertEqual(out, b"")
        self.assertTrue(err.startswith(b"error:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
