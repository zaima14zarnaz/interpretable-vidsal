#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in confirm_candidate_pairs.py."""
from __future__ import annotations

import sqlite3

from confirm_candidate_pairs import main

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f'Error: {exc}')
