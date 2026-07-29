"""Pytest configuration.

Its mere presence at the repository root puts that root on ``sys.path`` (pytest
"prepend" import mode), so tests can ``import src`` regardless of whether they
are launched via ``pytest`` or ``python -m pytest``.
"""
