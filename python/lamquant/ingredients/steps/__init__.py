"""Step ingredients (ADR 0050/0051) — the inner optimizer-step sequence
(backward + grad-clip + optimizer.step + post-step clamp), one load-bearing unit
whose ordering must be preserved exactly.
"""
