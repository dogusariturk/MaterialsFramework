"""Tests for Sqs2tdb.

Sqs2tdb wraps the ATAT sqs2tdb binary. All functional tests are skipped when
the binary is not present in PATH.
"""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("pycalphad")

pytestmark = pytest.mark.integration

_SQS2TDB_AVAILABLE = shutil.which("sqs2tdb") is not None


def test_raises_oserror_without_binary() -> None:
    """Sqs2tdb() raises OSError when sqs2tdb binary is not on PATH."""
    if _SQS2TDB_AVAILABLE:
        pytest.skip("sqs2tdb is installed, skipping absence test")

    from materialsframework.tools.sqs2tdb import Sqs2tdb

    with pytest.raises(OSError, match="sqs2tdb is not installed"):
        Sqs2tdb()


@pytest.mark.skipif(not _SQS2TDB_AVAILABLE, reason="sqs2tdb binary not found")
def test_init_stores_default_params() -> None:
    """Sqs2tdb stores MD and relaxation parameters correctly."""
    from materialsframework.tools.sqs2tdb import Sqs2tdb

    wrapper = Sqs2tdb()
    assert wrapper.md_temperature == pytest.approx(1000)
    assert wrapper.md_pressure == pytest.approx(1)
    assert wrapper.md_timestep == pytest.approx(1.0)
    assert wrapper.fmax == pytest.approx(0.001)
    assert wrapper.verbose is False
    assert wrapper.dbf is None


@pytest.mark.skipif(not _SQS2TDB_AVAILABLE, reason="sqs2tdb binary not found")
def test_calculator_lazy_property() -> None:
    """_calculator is None until the .calculator property is accessed."""
    from materialsframework.tools.sqs2tdb import Sqs2tdb

    wrapper = Sqs2tdb()
    assert wrapper._calculator is None
