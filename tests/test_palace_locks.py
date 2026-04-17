"""Tests for mine_global_lock: non-blocking cross-process lock semantics."""
import threading

import pytest

from mempalace.palace import MineAlreadyRunning, mine_global_lock


def test_mine_global_lock_acquired(tmp_path, monkeypatch):
    """Lock is acquired and released without error."""
    monkeypatch.setenv("HOME", str(tmp_path))
    with mine_global_lock():
        pass  # should not raise


def test_mine_global_lock_second_acquire_raises(tmp_path, monkeypatch):
    """Concurrent second acquire raises MineAlreadyRunning."""
    monkeypatch.setenv("HOME", str(tmp_path))
    results: list[str] = []

    with mine_global_lock():
        # While this lock is held, spawn a thread that tries to acquire.
        def try_acquire():
            try:
                with mine_global_lock():
                    results.append("acquired")
            except MineAlreadyRunning:
                results.append("blocked")

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=5)

    assert results == ["blocked"]


def test_mine_global_lock_reusable_after_release(tmp_path, monkeypatch):
    """Lock can be re-acquired after the context manager exits."""
    monkeypatch.setenv("HOME", str(tmp_path))

    with mine_global_lock():
        pass  # first acquire + release

    # Second acquire must succeed; MineAlreadyRunning would propagate as failure.
    with mine_global_lock():
        pass


def test_mine_global_lock_exception_still_releases(tmp_path, monkeypatch):
    """Lock is released even when the body raises."""
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ValueError):
        with mine_global_lock():
            raise ValueError("boom")

    # Must be acquirable again after the exception.
    with mine_global_lock():
        pass
