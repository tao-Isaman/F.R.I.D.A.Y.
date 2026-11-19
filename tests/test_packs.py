"""Claude edition: voice models and the CUDA DLLs are not in the installer. packs.py fetches them from their own
public sources on the owner's PC (Hugging Face, GitHub, PyPI), checks SHA-256, retries, and reports progress."""
import hashlib, io, os, zipfile

import pytest

import packs


class Net:
    """Fake urlopen: url -> bytes; can fail the first N tries of a url, or serve wrong bytes."""
    def __init__(self, files, fail_first=None, corrupt=()):
        self.files, self.fail_first, self.corrupt, self.calls = files, dict(fail_first or {}), set(corrupt), []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if self.fail_first.get(url, 0) > 0:
            self.fail_first[url] -= 1
            raise OSError("connection reset")
        body = self.files[url]
        if url in self.corrupt:
            body = b"x" + body[1:]
        r = io.BytesIO(body)
        r.headers = {"Content-Length": str(len(body))}
        r.__enter__ = lambda *a: r
        return _Ctx(r)


class _Ctx:
    def __init__(self, r): self.r = r
    def __enter__(self): return self.r
    def __exit__(self, *a): return False


def _sha(b): return hashlib.sha256(b).hexdigest()


def _wheel():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nvidia/cublas/bin/cublas64_12.dll", b"DLL1")
        z.writestr("nvidia/cudnn/bin/cudnn64_9.dll", b"DLL2")
        z.writestr("nvidia/cudnn/include/cudnn.h", b"header")  # not a DLL: not kept
    return buf.getvalue()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model, wheel = b"M" * 5000, _wheel()
    monkeypatch.setattr(packs, "ROOT", str(tmp_path))
    monkeypatch.setattr(packs, "PACKS", {
        "voice": packs.Pack("ชุดเสียง", [packs.File("https://hf/m.onnx", _sha(model), len(model), "models/x/m.onnx")]),
        "gpu": packs.Pack("GPU", [packs.File("https://pypi/w.whl", _sha(wheel), len(wheel), "gpu/w.whl",
                                             extract=r"nvidia/[^/]+/bin/[^/]+\.dll")], marker="gpu/.complete"),
    })
    return tmp_path, {"https://hf/m.onnx": model, "https://pypi/w.whl": wheel}


def test_fetch_checks_hash_and_reports_progress(setup):
    root, files = setup
    seen = []
    assert not packs.installed("voice")
    packs.fetch("voice", progress=lambda done, total: seen.append((done, total)), opener=Net(files))
    assert packs.installed("voice")
    assert (root / "models/x/m.onnx").read_bytes() == files["https://hf/m.onnx"]
    assert seen[-1] == (5000, 5000) and not list(root.rglob("*.part"))


def test_wrong_bytes_are_never_kept(setup):
    root, files = setup
    net = Net(files, corrupt={"https://hf/m.onnx"})
    with pytest.raises(packs.PackError, match="SHA-256"):
        packs.fetch("voice", opener=net)
    assert not packs.installed("voice") and not (root / "models/x/m.onnx").exists()
    assert len(net.calls) == packs.TRIES and not list(root.rglob("*.part"))


def test_a_dropped_connection_is_retried(setup):
    root, files = setup
    net = Net(files, fail_first={"https://hf/m.onnx": 2})
    packs.fetch("voice", opener=net)
    assert packs.installed("voice") and len(net.calls) == 3


def test_files_already_there_are_not_downloaded_again(setup):
    root, files = setup
    packs.fetch("voice", opener=Net(files))
    net = Net(files)
    packs.fetch("voice", opener=net)
    assert net.calls == []


def test_gpu_wheel_leaves_only_its_dlls(setup):
    root, files = setup
    packs.fetch("gpu", opener=Net(files))
    assert packs.installed("gpu")
    assert sorted(p.name for p in (root / "gpu").iterdir()) == [".complete", "cublas64_12.dll", "cudnn64_9.dll"]
    assert packs.dll_dirs() == [str(root / "gpu")]


def test_no_dll_dir_before_the_gpu_pack(setup):
    assert packs.dll_dirs() == []


def test_real_pins_are_complete():
    for name, p in packs.REAL_PACKS.items():
        assert p.files and p.title
        for f in p.files:
            assert f.url.startswith("https://") and len(f.sha256) == 64 and f.size > 0 and not os.path.isabs(f.dest)
            assert "/resolve/main/" not in f.url  # pinned to a commit, not a moving branch
    assert packs.size_mb("gpu", packs.REAL_PACKS) > 1000 and packs.size_mb("voice", packs.REAL_PACKS) > 200
