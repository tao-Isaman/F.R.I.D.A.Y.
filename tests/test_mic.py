"""The owner said "FRIDAY doesn't hear the mic" (build f91780f). Every way the mic can be silent must say so in the
panel: no input device, a device that only delivers zeros (Windows mic privacy off), the wake-word model still
downloading or failing, and errors that used to go only to the log."""
import numpy as np

import wakeword


def frames(n, value=0):
    return [np.full(wakeword.FRAME, value, np.int16) for _ in range(n)]


def test_dead_mic_fires_once_after_seconds_of_exact_zeros():
    w = wakeword.DeadMicWatch(seconds=1.0)
    need = int(1.0 * wakeword.SR / wakeword.FRAME)
    fired = [w(f) for f in frames(need * 3)]
    assert fired.count(True) == 1 and fired.index(True) == need - 1


def test_real_sound_means_the_mic_works():
    w = wakeword.DeadMicWatch(seconds=1.0)
    assert not w(frames(1, 3)[0])
    assert not any(w(f) for f in frames(100))  # quiet room after real sound: not an alarm


def test_mic_status_names_the_device(fr, monkeypatch):
    monkeypatch.setattr(fr.sd, "query_devices", lambda kind=None: {"name": "Microphone (Realtek)", "default_samplerate": 48000.0}, raising=False)
    ok, text = fr.mic_status()
    assert ok and "Microphone (Realtek)" in text


def test_no_input_device_is_said_plainly(fr, monkeypatch):
    def none(kind=None):
        raise ValueError("No input device matching -1")
    monkeypatch.setattr(fr.sd, "query_devices", none, raising=False)
    ok, text = fr.mic_status()
    assert not ok and "ไม่พบไมค์" in text


def test_dead_mic_tells_the_panel_how_to_fix_it(fr):
    fr.on_dead_mic()
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m]
    assert any("Privacy" in t and "ไมค์" in t for t in told), told


def test_wake_model_download_is_announced_and_failures_shown(fr, monkeypatch):
    monkeypatch.setattr(wakeword, "_tiny", None)
    def broken():
        raise OSError("HF unreachable")
    monkeypatch.setattr(wakeword, "_load", broken)
    assert fr.prepare_wake() is False
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m]
    assert any("คำปลุก" in t and "75 MB" in t for t in told) and any("HF unreachable" in t for t in told), told


def test_wake_model_ready_is_quiet(fr, monkeypatch):
    monkeypatch.setattr(wakeword, "_load", lambda: None)
    monkeypatch.setattr(wakeword, "_tiny", object())
    assert fr.prepare_wake() is True
    assert not [m for m in fr.ui.panel if "msg" in m]
