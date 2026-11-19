"""Claude edition: the big speech models (~3 GB) are not installed with FRIDAY. The first time the owner speaks, FRIDAY
asks once, downloads them in the background (typing keeps working) and says when listening is ready."""
import threading


class Downloads:
    def __init__(self, cached=()):
        self.cached, self.fetched, self.done = set(cached), [], threading.Event()

    def __call__(self, name, local_files_only=False):
        if local_files_only:
            if name not in self.cached:
                raise FileNotFoundError(name)
            return f"/cache/{name}"
        self.fetched.append(name)
        self.cached.add(name)
        return f"/cache/{name}"


class Packs:
    """Fake packs module state: which packs are on the PC, what got fetched."""
    def __init__(self, have=(), nvidia=False, fail=()):
        self.have, self.nvidia, self.fail, self.fetched = set(have), nvidia, set(fail), []

    def installed(self, name):
        return name in self.have

    def fetch(self, name, progress=lambda d, t: None):
        if name in self.fail:
            import packs
            raise packs.PackError("no internet")
        progress(50, 100)
        progress(100, 100)
        self.fetched.append(name)
        self.have.add(name)

    def has_nvidia(self):
        return self.nvidia


def _setup(fr, monkeypatch, cached=(), answer=True, have=("voice",), nvidia=False, answers=None):
    d = Downloads(cached)
    p = Packs(have, nvidia)
    monkeypatch.setattr(fr, "_download_model", d)
    monkeypatch.setattr(fr.packs, "installed", p.installed)
    monkeypatch.setattr(fr.packs, "fetch", p.fetch)
    monkeypatch.setattr(fr.packs, "has_nvidia", p.has_nvidia)
    replies = list(answers) if answers is not None else None
    monkeypatch.setattr(fr, "confirm", lambda q, detail="", strong=False: (
        fr.ui.confirms.append(q), replies.pop(0) if replies is not None else answer)[1])
    monkeypatch.setattr(fr, "EARS", {})
    started = []
    monkeypatch.setattr(fr, "_background", lambda fn: (started.append(fn), fn()))
    fr.ui.packs = p
    return d, started


def test_ready_when_the_models_are_already_there(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, cached=fr.BIG_EARS)
    assert fr.ensure_ears() is True and not fr.ui.confirms and not d.fetched and not started


def test_first_voice_use_asks_then_downloads_in_the_background(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, answer=True)
    assert fr.ensure_ears() is False  # this utterance is skipped: nothing to transcribe with yet
    assert len(fr.ui.confirms) == 1 and fr.ears_size_text() in fr.ui.confirms[0]  # the set's real size, not a fixed "3 GB"
    assert d.fetched == [m for m in fr.BIG_EARS] and len(started) == 1
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m]
    assert any("พร้อมฟังเสียงแล้ว" in t for t in told), told
    assert fr.ensure_ears() is True and len(fr.ui.confirms) == 1  # no second question


def test_declined_is_not_asked_again_this_session(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, answer=False)
    assert fr.ensure_ears() is False and fr.ensure_ears() is False
    assert len(fr.ui.confirms) == 1 and not d.fetched and not started


def test_failed_download_can_be_retried_later(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, answer=True)

    def broken(name, local_files_only=False):
        if local_files_only:
            raise FileNotFoundError(name)
        raise OSError("no internet")
    monkeypatch.setattr(fr, "_download_model", broken)
    assert fr.ensure_ears() is False
    assert any("ไม่สำเร็จ" in m["msg"]["text"] for m in fr.ui.panel if "msg" in m)
    monkeypatch.setattr(fr, "_download_model", d)
    assert fr.ensure_ears() is False and len(fr.ui.confirms) == 2 and d.fetched  # asked again, downloaded this time


def test_live_captions_never_ask(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch)
    assert fr._transcribe(object(), quiet=True) == ""
    assert not fr.ui.confirms and not started


def test_voice_pack_comes_with_the_first_download(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, have=())
    assert fr.ensure_ears() is False
    assert fr.ui.packs.fetched == ["voice"] and "MB" in fr.ui.confirms[0]
    assert fr.ensure_ears() is True


def test_ready_needs_the_voice_pack_too(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, cached=fr.BIG_EARS, have=())
    assert fr.ensure_ears() is False and fr.ui.packs.fetched == ["voice"] and not d.fetched


def test_gpu_pack_is_offered_only_with_an_nvidia_card(fr, monkeypatch):
    _setup(fr, monkeypatch, nvidia=True, answers=[True, True])
    fr.ensure_ears()
    assert fr.ui.packs.fetched == ["gpu"] and len(fr.ui.confirms) == 2 and "NVIDIA" in fr.ui.confirms[1]
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m]
    assert any("เปิด FRIDAY ใหม่" in t for t in told), told


def test_gpu_pack_declined_keeps_the_cpu(fr, monkeypatch):
    _setup(fr, monkeypatch, nvidia=True, answers=[True, False])
    fr.ensure_ears()
    assert "gpu" not in fr.ui.packs.fetched and len(fr.ui.confirms) == 2


def test_no_gpu_question_without_nvidia(fr, monkeypatch):
    _setup(fr, monkeypatch, nvidia=False)
    fr.ensure_ears()
    assert len(fr.ui.confirms) == 1


def test_thai_speech_uses_the_windows_voice_until_the_pack_is_in(fr, monkeypatch):
    _setup(fr, monkeypatch, have=())
    used = []
    monkeypatch.setattr(fr.tts_th, "synth_mixed", lambda *a, **k: used.append("neural") or b"")
    monkeypatch.setattr(fr, "_synth", lambda text: used.append("windows") or _done(b""))
    fr.tts_wav("สวัสดีค่ะ")
    assert used == ["windows"]


def test_enrolling_a_voice_needs_the_pack(fr, monkeypatch):
    _setup(fr, monkeypatch, have=())
    monkeypatch.setattr(fr.remote, "current", lambda: None)
    assert "ชุดเสียง" in fr.enroll_voice()


def test_whisper_runs_on_the_gpu_only_with_the_gpu_pack(fr, monkeypatch):
    _setup(fr, monkeypatch, have=())
    monkeypatch.delenv("FRIDAY_WHISPER_DEVICE", raising=False)
    assert fr.whisper_device() == "cpu"
    fr.ui.packs.have.add("gpu")
    assert fr.whisper_device() == "cuda"
    monkeypatch.setenv("FRIDAY_WHISPER_DEVICE", "cpu")
    assert fr.whisper_device() == "cpu"


async def _done(v):
    return v


def test_a_broken_thai_voice_falls_back_once_and_stays_off(fr, monkeypatch):
    _setup(fr, monkeypatch, have=("voice",))
    monkeypatch.setattr(fr, "TTS_OFF", {})
    monkeypatch.setattr(fr, "TH_VOICE", "vanira:1")
    calls, used = [], []

    def broken(*a, **k):
        calls.append(1)
        raise RuntimeError("tts_th worker died")
    monkeypatch.setattr(fr.tts_th, "synth_mixed", broken)
    monkeypatch.setattr(fr, "_synth", lambda text: used.append("windows") or _done(b""))
    fr.tts_wav("สวัสดีค่ะ")
    fr.tts_wav("วันนี้อากาศดี")
    assert len(calls) == 1 and used == ["windows", "windows"]  # no new worker for every reply
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m]
    assert sum("เสียง Windows" in t for t in told) == 1, told


def test_voice_is_offered_at_startup_without_speaking(fr, monkeypatch):
    _setup(fr, monkeypatch, have=())
    fr.offer_voice()
    assert len(fr.ui.confirms) == 1 and fr.ui.packs.fetched == ["voice"]


def test_a_startup_no_is_remembered_but_the_menu_still_works(fr, monkeypatch):
    import rules
    _setup(fr, monkeypatch, have=(), answer=False)
    fr.offer_voice()
    assert rules.settings().get("voice_offer") == "declined"
    fr.offer_voice()  # next start: not asked again
    assert len(fr.ui.confirms) == 1
    fr.offer_voice(manual=True)  # tray › ดาวน์โหลดโมเดลเสียง…
    assert len(fr.ui.confirms) == 2


def test_menu_when_complete_offers_the_gpu_pack(fr, monkeypatch):
    _setup(fr, monkeypatch, cached=fr.BIG_EARS, have=("voice",), nvidia=True, answer=True)
    fr.offer_voice(manual=True)
    assert fr.ui.packs.fetched == ["gpu"] and "GPU" in fr.ui.confirms[0]


class _Model:
    def __init__(self, name, en_prob=0.0):
        self.name, self.en_prob, self.calls = name, en_prob, []

    def detect_language(self, audio):
        return None, None, [("en", self.en_prob), ("th", 1 - self.en_prob)]

    def transcribe(self, audio, **kw):
        self.calls.append(kw["language"])
        return [type("S", (), {"text": f" {self.name}"})()], None


def test_run_stt_routes_by_detected_language(fr):
    th, en = _Model("th-model"), _Model("en-model")
    assert fr._run_stt(_Model("det", en_prob=0.95), lambda: th, en, b"") == ("en", "en-model")
    assert fr._run_stt(_Model("det", en_prob=0.30), lambda: th, en, b"") == ("th", "th-model")
    assert fr._run_stt(_Model("det", en_prob=0.30), lambda: None, en, b"") == ("th", "en-model")  # no Thai model
    assert fr._run_stt(_Model("det"), lambda: th, en, b"", lang="en") == ("en", "en-model")  # FRIDAY_LANG forced
