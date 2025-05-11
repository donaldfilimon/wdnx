import os
import pytest
from pathlib import Path

from wdbx.download import download_file

class DummyResponse:
    def __init__(self, content=b'test'):
        self._content = content
        self.status_code = 200
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size=8192):
        yield self._content

def test_download_file(tmp_path, monkeypatch):
    url = "http://example.com/test.pdf"
    dest_dir = tmp_path / "downloads"
    # Monkeypatch requests.get
    import wdbx.download as download_module
    class DummyResponseClass(DummyResponse):
        pass
    def dummy_get(u, stream, timeout):
        assert u == url
        return DummyResponseClass()
    monkeypatch.setattr(download_module.requests, 'get', dummy_get)
    # Call download_file
    output = download_file(url, dest_dir)
    assert output.endswith('test.pdf')
    file_path = Path(output)
    assert file_path.exists()
    assert file_path.read_bytes() == b'test'
    # Cleanup
    os.remove(output) 