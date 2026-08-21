"""Deterministically shard a pytest collection across independent workers."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cubic correctness sharding")
    group.addoption("--cubic-shard-count", type=int, default=1)
    group.addoption("--cubic-shard-index", type=int, default=0)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    count = config.getoption("--cubic-shard-count")
    index = config.getoption("--cubic-shard-index")
    if count < 1:
        raise pytest.UsageError("--cubic-shard-count must be positive")
    if not 0 <= index < count:
        raise pytest.UsageError("--cubic-shard-index must be in [0, count)")
    if count == 1:
        return
    selected = [
        item for position, item in enumerate(items) if position % count == index
    ]
    deselected = [
        item for position, item in enumerate(items) if position % count != index
    ]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
