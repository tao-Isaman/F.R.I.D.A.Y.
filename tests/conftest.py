"""Run FRIDAY's Windows app on Linux CI/dev for logic tests: Windows-only modules are stubbed, every file the code
writes goes to a temp folder, DPAPI is replaced by identity (encryption itself is only testable on Windows)."""
import ctypes, os, sys
from unittest import mock

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
if os.name != "nt":
    for m in ("winsound", "sounddevice"):
        sys.modules.setdefault(m, mock.MagicMock())
    ctypes.windll = getattr(ctypes, "windll", mock.MagicMock())
    ctypes.WinDLL = getattr(ctypes, "WinDLL", mock.MagicMock())
    ctypes.WINFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import rules, security
    monkeypatch.setattr(rules, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(security, "SECRETS", str(tmp_path / "secrets.dat"))
    monkeypatch.setattr(security, "AUDIT", str(tmp_path / "audit.log"))
    if os.name != "nt":
        monkeypatch.setattr(security, "_dpapi", lambda data, protect: data)
    rules.settings()
    return tmp_path


@pytest.fixture
def fr(home, monkeypatch):
    """friday.py with its UI seams replaced: windows answer from a script, alerts are collected."""
    import friday
    ui = mock.Mock(answers=[], alerts=[], confirms=[], panel=[])
    monkeypatch.setattr(friday, "on_tk_thread", lambda fn: fn())
    monkeypatch.setattr(friday, "_ask_line", lambda title, note, hide=False: (ui.notes.append(note), ui.answers.pop(0) if ui.answers else "")[1])
    monkeypatch.setattr(friday, "phone_alert", ui.alerts.append)
    monkeypatch.setattr(friday, "confirm", lambda q, d="", strong=False: (ui.confirms.append(q), True)[1])
    monkeypatch.setattr(friday.panel, "send", lambda **kw: ui.panel.append(kw))
    ui.notes = []
    friday.ui = ui
    return friday


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    """Tests never fetch models: a test that forgot to stub _download_model downloaded ~12 MB from Hugging Face
    (2026-11-19). Any real download attempt now fails the test instead."""
    import faster_whisper.utils as fwu

    def refuse(*a, **k):
        raise AssertionError(f"test tried to download a model: {a} {k}")
    monkeypatch.setattr(fwu, "download_model", refuse)
