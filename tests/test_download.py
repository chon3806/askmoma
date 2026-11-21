"""The data download refuses Git LFS pointer files, which are a few hundred bytes."""

import io

from askmoma import download


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_small_download_is_reported_as_an_lfs_pointer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 70000000\n"
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(pointer))
    assert download.main() == 1
    assert "Git LFS pointer" in capsys.readouterr().out


def test_real_sized_download_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    content = b"id,title\n" + b"1,x\n" * 2000
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(content))
    assert download.main() == 0
    assert (tmp_path / "Artworks.csv").read_bytes() == content


def test_pointer_leaves_no_file_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"version git-lfs"))
    assert download.main() == 1
    assert list(tmp_path.iterdir()) == []  # neither a pointer posing as Artworks.csv nor a .part file


def test_interrupted_download_keeps_the_previous_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    (tmp_path / "Artworks.csv").write_bytes(b"the old, complete data")

    class Broken(FakeResponse):
        def read(self, *args):
            raise ConnectionResetError("network dropped")

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Broken(b""))
    try:
        download.main()
    except ConnectionResetError:
        pass
    assert (tmp_path / "Artworks.csv").read_bytes() == b"the old, complete data"
    assert not list(tmp_path.glob("*.part"))
