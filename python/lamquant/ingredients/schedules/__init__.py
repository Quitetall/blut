"""Scheduler ingredients (ADR 0050/0051) — LR schedules a trainer wraps its
optimizer in. ``wsd`` (Warmup-Stable-Decay) is the first; relocated out of the
codec trainer so the SNN trainers no longer cross-import it.
"""
