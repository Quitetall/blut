"""Smoke for build_optimizer resolution paths.

Doesn't need transformers, torch, or apollo_torch — uses pure stdlib
to verify the AdamW/AdamW8bit branch returns the expected mapping
without I/O. The APOLLO branches are tested via the integration
suite (gated behind the heavy deps); we verify the contract here:

  - AdamW    → ('adamw_torch', None)
  - AdamW8bit → ('adamw_8bit', None)
  - APOLLO without transformers + apollo_torch → RuntimeError with hint
  - unknown optim name → RuntimeError

Run via: python3 test_build_optimizer.py
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import trainer  # noqa: E402


class FakeModel:
    """Stand-in for a torch model. Only `parameters()` is inspected
    by build_optimizer when constructing APOLLO."""

    def parameters(self):
        return []


class BuildOptimizerTests(unittest.TestCase):
    def test_adamw_returns_string_only(self):
        s, o = trainer.build_optimizer("adam_w", FakeModel(), 1e-4)
        self.assertEqual(s, "adamw_torch")
        self.assertIsNone(o)

    def test_adamw_8bit_returns_string_only(self):
        s, o = trainer.build_optimizer("adam_w8bit", FakeModel(), 1e-4)
        self.assertEqual(s, "adamw_8bit")
        self.assertIsNone(o)

    def test_unknown_optim_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            trainer.build_optimizer("nonsense", FakeModel(), 1e-4)
        self.assertIn("unknown optimizer", str(ctx.exception))

    def test_apollo_without_either_path_raises_helpful_error(self):
        # Force both resolution paths to fail:
        #   - Pretend transformers OptimizerNames doesn't include apollo
        #   - Pretend apollo_torch isn't installed
        with mock.patch.dict(sys.modules, {"transformers.training_args": None}):
            # mock.patch.dict with None makes the import raise; but
            # the existing transformers may still resolve via the
            # parent package import. Use a more direct approach:
            # block the apollo_torch import outright and accept that
            # the transformers builtin check returns False on this
            # host (it does — the user's transformers doesn't have
            # the PR yet).
            with mock.patch.dict(sys.modules, {"apollo_torch": None}):
                with self.assertRaises(RuntimeError) as ctx:
                    trainer.build_optimizer("apollo_mini", FakeModel(), 1e-4)
                msg = str(ctx.exception)
                self.assertIn("APOLLO", msg)
                self.assertIn("pip install apollo-torch", msg)


if __name__ == "__main__":
    unittest.main()
