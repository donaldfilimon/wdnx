import pytest

import wdbx


def test_public_api_members():
    # Ensure core functions and classes are exposed
    assert hasattr(wdbx, 'WDBX')
    assert hasattr(wdbx, 'AsyncWDBX')
    assert hasattr(wdbx, 'download_file')
    assert hasattr(wdbx, 'configure_database')
    assert hasattr(wdbx, 'start_metrics_server')
    assert hasattr(wdbx, 'initialize_backend')
    assert hasattr(wdbx, 'initialize_async_backend') 