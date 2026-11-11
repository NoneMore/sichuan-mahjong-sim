"""Shared cache fixtures for the Python test suite.

``shared_cache`` is written once per session and must stay read-only: readers
reuse it directly, while tests that mutate the manifest, arrays, or inventory
request ``fresh_cache`` for an independent copy. This keeps cross-test
isolation without regenerating the same captured interval for every case.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def shared_cache(tmp_path_factory):
    # Import lazily so collecting pytest does not import the package under
    # test (and its optional native dependencies) before a test asks for it.
    from .cache_fixture import write_cache

    return write_cache(tmp_path_factory.mktemp("shared-cache"))


@pytest.fixture
def fresh_cache(tmp_path, shared_cache):
    from .cache_fixture import copy_cache

    return copy_cache(shared_cache, tmp_path)
