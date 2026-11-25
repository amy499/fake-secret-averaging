"""Shared CLI helpers for the scripts."""

from __future__ import annotations

import sys


def utf8_stdio() -> None:
    """Write UTF-8 to stdout/stderr even when they are redirected.

    On Windows a redirected stream (pipe, `> log.txt`, Tee-Object) defaults to cp1252, and on a
    cluster a C/POSIX locale can mean ASCII; printing "≥" or tokenizer pieces would then crash
    the script after its run folder was already written.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
