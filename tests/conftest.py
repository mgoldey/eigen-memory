"""Shared test doubles for the kernel tests.

FakeConn / FakeCursor: full-featured, logs SQL for assertions.
FakeLLM: minimal LLM stub that records prompts and returns a canned reply.
NullConn: minimal conn that silently accepts all SQL (no logging).
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Full-featured fakes (used by test_kernel_consolidation and others that
# assert on what SQL was executed).
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))


class FakeConn:
    def __init__(self):
        self.executed = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.executed)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class FakeLLM:
    """Mimics client.chat.completions.create; records prompts."""

    def __init__(self, reply="<thought>...</thought>\nRULE: requests and reports route oppositely."):
        self.prompts = []
        self.reply = reply
        chat = type("Chat", (), {})()
        completions = type("Completions", (), {})()
        completions.create = self._create
        chat.completions = completions
        self.chat = chat

    def _create(self, model=None, messages=None, **kw):
        self.prompts.append(messages[-1]["content"])
        msg = type("Msg", (), {"content": self.reply})()
        choice = type("Choice", (), {"message": msg})()
        return type("Resp", (), {"choices": [choice]})()


# ---------------------------------------------------------------------------
# Minimal no-op connection (used by tests that only need the kernel to run
# without touching a real database).
# ---------------------------------------------------------------------------

class NullConn:
    """Accepts all SQL silently; tracks inserts and retirements for assertions."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.retired = []
        self.inserts = 0

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                s = " ".join(sql.split())
                if "INSERT INTO semantic_core" in s:
                    conn.inserts += 1
                elif "UPDATE semantic_core" in s and "retired" in s:
                    conn.retired.append(params[0] if params else None)
                self._last = s

            def fetchall(self):
                if "FROM semantic_core" in getattr(self, "_last", ""):
                    return [(r[0], r[1]) for r in conn.rows]
                return []

        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass
