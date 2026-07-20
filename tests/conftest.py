"""Pytest configuration.

Skip markers live in ``tests/marks.py`` so they can be imported by test
modules. Data-dependent tests are skipped when the gitignored processed marts
are absent (see ``tests/marks.py``).
"""
