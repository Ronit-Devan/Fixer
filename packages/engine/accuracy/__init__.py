"""Ground-truth accuracy harness for the ET verdict engine.

See ``accuracy.ground_truth`` for the planters and ``run_all`` entry point.
This package intentionally does NOT depend on torch at import time so it
remains importable on the no-GPU dev / CI box; torch is loaded lazily inside
each planter.
"""
