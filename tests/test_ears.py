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


def _setup(fr, monkeypatch, cached=(), answer=True):
    d = Downloads(cached)
    monkeypatch.setattr(fr, "_download_model", d)
    monkeypatch.setattr(fr, "confirm", lambda q, detail="", strong=False: (fr.ui.confirms.append(q), answer)[1])
    monkeypatch.setattr(fr, "EARS", {})
    started = []
    monkeypatch.setattr(fr, "_background", lambda fn: (started.append(fn), fn()))
    return d, started


def test_ready_when_the_models_are_already_there(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, cached=fr.BIG_EARS)
    assert fr.ensure_ears() is True and not fr.ui.confirms and not d.fetched and not started


def test_first_voice_use_asks_then_downloads_in_the_background(fr, monkeypatch):
    d, started = _setup(fr, monkeypatch, answer=True)
    assert fr.ensure_ears() is False  # this utterance is skipped: nothing to transcribe with yet
    assert len(fr.ui.confirms) == 1 and "3 GB" in fr.ui.confirms[0]
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
