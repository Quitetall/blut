# SPDX-License-Identifier: AGPL-3.0-or-later
"""The benchmark gate fails on a slowdown seen in two passes, and only then."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "bench_regression_gate",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "bench_regression_gate.py"),
)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def write_changes(root, changes):
    """`changes` maps a benchmark name to (median change, lower bound)."""
    for bench, (point, lower) in changes.items():
        d = os.path.join(root, bench, "change")
        os.makedirs(d)
        estimate = {"point_estimate": point, "confidence_interval": {"lower_bound": lower}}
        with open(os.path.join(d, "estimates.json"), "w", encoding="utf-8") as f:
            json.dump({"median": estimate, "mean": estimate}, f)


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gate.main(args)
    return code, out.getvalue(), err.getvalue()


class BenchRegressionGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.first = os.path.join(self.tmp.name, "first")
        self.second = os.path.join(self.tmp.name, "second")
        self.suspects = os.path.join(self.tmp.name, "suspects")

    def test_a_single_pass_fails_on_a_lower_bound_over_the_limit(self):
        write_changes(self.first, {"fast (a)": (0.02, 0.01), "slow": (0.20, 0.17)})
        code, out, err = run([self.first, "0.10"])
        self.assertEqual(code, 1)
        self.assertIn("REGRESSED", out)
        self.assertIn("slow", err)
        self.assertNotIn("fast (a)", err)

    def test_a_wide_interval_reaching_below_the_limit_is_not_flagged(self):
        write_changes(self.first, {"noisy": (0.30, 0.04)})
        self.assertEqual(run([self.first, "0.10"])[0], 0)

    def test_no_estimates_is_an_error_not_a_pass(self):
        os.makedirs(self.first)
        self.assertEqual(run([self.first, "0.10"])[0], 2)

    def test_a_slowdown_that_does_not_repeat_passes(self):
        # The recorded case: +17.3% in one pass, level in the next.
        write_changes(self.first, {"status emit": (0.173, 0.173), "other": (0.0, -0.01)})
        code, _, _ = run([self.first, "0.10", "--suspects-out", self.suspects])
        self.assertEqual(code, 1)
        with open(self.suspects, encoding="utf-8") as f:
            self.assertEqual(f.read(), "status emit\n")
        write_changes(self.second, {"status emit": (0.007, 0.005), "other": (0.0, -0.01)})
        code, out, _ = run([self.second, "0.10", "--confirm", self.suspects])
        self.assertEqual(code, 0)
        self.assertIn("not confirmed by this pass: status emit", out)

    def test_a_slowdown_in_both_passes_fails(self):
        write_changes(self.first, {"real": (0.25, 0.22)})
        run([self.first, "0.10", "--suspects-out", self.suspects])
        write_changes(self.second, {"real": (0.24, 0.21)})
        code, out, err = run([self.second, "0.10", "--confirm", self.suspects])
        self.assertEqual(code, 1)
        self.assertIn("REGRESSED", out)
        self.assertIn("regressed in both passes: real", err)

    def test_a_slowdown_only_in_the_second_pass_is_unconfirmed(self):
        write_changes(self.first, {"a": (0.25, 0.22), "b": (0.0, 0.0)})
        run([self.first, "0.10", "--suspects-out", self.suspects])
        write_changes(self.second, {"a": (0.0, 0.0), "b": (0.25, 0.22)})
        code, out, _ = run([self.second, "0.10", "--confirm", self.suspects])
        self.assertEqual(code, 0)
        self.assertIn("unconfirmed", out)

    def test_a_clean_first_pass_writes_an_empty_suspect_list(self):
        write_changes(self.first, {"a": (0.0, 0.0)})
        code, _, _ = run([self.first, "0.10", "--suspects-out", self.suspects])
        self.assertEqual(code, 0)
        self.assertEqual(os.path.getsize(self.suspects), 0)


if __name__ == "__main__":
    unittest.main()
