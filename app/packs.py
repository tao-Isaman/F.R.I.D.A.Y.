"""Claude edition: parts that are not in the installer, fetched from their own public sources on the owner's PC the
first time they are needed (asked first). Nothing is re-hosted by us.
  voice  Thai voice (VaniraTTS, Hugging Face), English voice (Piper hfc_female, Hugging Face), owner's-voice model
         (WeSpeaker ResNet34, sherpa-onnx GitHub release) — ~260 MB
  gpu    cuBLAS + cuDNN DLLs for Whisper on an NVIDIA card: the official NVIDIA wheels from PyPI; only their bin/*.dll
         are kept, in <app>/gpu — ~1.2 GB download
Every file is pinned (commit / version) with its SHA-256; a file is only kept after its hash matches. The Whisper
speech models are fetched by faster-whisper itself (friday.ensure_ears)."""
import hashlib, os, re, shutil, urllib.request, zipfile
from dataclasses import dataclass, field

ROOT = os.environ.get("FRIDAY_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # install folder: models/, gpu/
TRIES = 3
CHUNK = 1 << 20


class PackError(Exception):
    pass


@dataclass(frozen=True)
class File:
    url: str
    sha256: str
    size: int
    dest: str  # relative to ROOT
    extract: str = ""  # a zip (wheel): keep only members matching this regex, flat, next to dest; then drop the zip


@dataclass(frozen=True)
class Pack:
    title: str
    files: list = field(default_factory=list)
    marker: str = ""  # written when every file is in (needed when files are extracted and removed)


_HF_VANIRA = "https://huggingface.co/VIZINTZOR/VaniraTTS/resolve/e82bf30417787e56f37b9cda3ee65a33cd29b235"
_HF_PIPER = ("https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512"
             "/en/en_US/hfc_female/medium")
_PYPI = "https://files.pythonhosted.org/packages"
_DLLS = r"nvidia/[^/]+/bin/[^/]+\.dll"

REAL_PACKS = {
    "voice": Pack("ชุดเสียงพูดไทย/อังกฤษ + จำเสียงเจ้าของ", [
        File(f"{_HF_VANIRA}/tts.onnx", "b5d41ae55bc7c9585804bbaa033041f57e09b2a6b87295883e8b3fee3c93cf05", 184861041,
             "models/vanira/tts.onnx"),
        File(f"{_HF_VANIRA}/vocab.json", "4a6fec845f5ca0e022e8582b35aa88fef098030678bb1a4e1d5121040ce9c52c", 2671,
             "models/vanira/vocab.json"),
        File(f"{_HF_PIPER}/en_US-hfc_female-medium.onnx",
             "5b3d6e36a33a1eb0e50b54900c229127e3059400885eb5e8f128ff04b31cd73c", 63221984,
             "models/piper/en_US-hfc_female-medium.onnx"),
        File(f"{_HF_PIPER}/en_US-hfc_female-medium.onnx.json",
             "8c7ed7408e7a37602ad6aff4785f44ecb69616b0153ec6a23e920713eba51fac", 6385,
             "models/piper/en_US-hfc_female-medium.onnx.json"),
        File("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "wespeaker_en_voxceleb_resnet34.onnx",
             "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94", 26534365,
             "models/speaker/wespeaker_en_voxceleb_resnet34.onnx"),
    ]),
    "gpu": Pack("GPU pack (NVIDIA cuBLAS + cuDNN)", [
        File(f"{_PYPI}/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
             "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
             "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661", 553162896, "gpu/cublas.whl", _DLLS),
        File(f"{_PYPI}/c7/bd/1d6709d2223e99f04079f16005f8ae6e11a14ee16b5eab6833686cf72fe1/"
             "nvidia_cudnn_cu12-9.28.1.4-py3-none-win_amd64.whl",
             "3467aafa9b260e6adf5fbbfc0e21d180133305b42032aa1160be2d3142bb8b09", 696300805, "gpu/cudnn.whl", _DLLS),
    ], marker="gpu/.complete"),
}
PACKS = REAL_PACKS


def _path(rel):
    return os.path.join(ROOT, *rel.split("/"))


def installed(name):
    p = PACKS[name]
    if p.marker:
        return os.path.exists(_path(p.marker))
    return all(os.path.exists(_path(f.dest)) and os.path.getsize(_path(f.dest)) == f.size for f in p.files)


def size_mb(name, packs=None):
    return round(sum(f.size for f in (packs or PACKS)[name].files) / 1e6)


def _download(f, opener, progress, base, total):
    """One file to dest (via .part), hash checked while streaming. Retried TRIES times; raises PackError."""
    dest = _path(f.dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part, last = dest + ".part", ""
    for _ in range(TRIES):
        h, done = hashlib.sha256(), 0
        try:
            with opener(f.url, timeout=60) as r, open(part, "wb") as out:
                while chunk := r.read(CHUNK):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    progress(base + done, total)
            if h.hexdigest() != f.sha256:
                last = f"SHA-256 ไม่ตรง ({os.path.basename(f.dest)})"
                continue
            os.replace(part, dest)
            return
        except OSError as e:
            last = f"{type(e).__name__}: {e}"
        finally:
            if os.path.exists(part):
                os.remove(part)
    raise PackError(last)


def _extract(f):
    zpath, keep = _path(f.dest), re.compile(f.extract)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if keep.fullmatch(m):
                with z.open(m) as src, open(os.path.join(os.path.dirname(zpath), os.path.basename(m)), "wb") as out:
                    shutil.copyfileobj(src, out)
    os.remove(zpath)


def fetch(name, progress=lambda done, total: None, opener=urllib.request.urlopen):
    """Every missing file of a pack; progress(bytes_done, bytes_total) as it goes. Raises PackError (nothing half-kept)."""
    p = PACKS[name]
    total, base = sum(f.size for f in p.files), 0
    for f in p.files:
        dest = _path(f.dest)
        if not f.extract and os.path.exists(dest) and os.path.getsize(dest) == f.size:
            base += f.size
            continue
        _download(f, opener, progress, base, total)
        if f.extract:
            _extract(f)
        base += f.size
    if p.marker:
        open(_path(p.marker), "w").close()
    progress(total, total)


def dll_dirs():
    """Folders to add to the DLL search path before Whisper loads (the GPU pack, once it is complete)."""
    return [os.path.dirname(_path(PACKS["gpu"].marker))] if installed("gpu") else []


def has_nvidia():
    return bool(shutil.which("nvidia-smi")) or os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                                                         "System32", "nvidia-smi.exe"))
