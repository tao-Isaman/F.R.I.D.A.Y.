"""Claude edition first run: the brain defaults to Claude, and a new install asks for a key that is tried before it is kept."""
import brain, brain_claude


def test_fresh_install_thinks_with_claude_sonnet(home):
    assert brain.current() == ("anthropic", "claude-sonnet-5-5")


def test_wizard_keeps_a_key_only_after_it_works(fr, monkeypatch):
    good = ["claude-sonnet-5-5", "claude-opus-5-5"]
    monkeypatch.setattr(brain_claude, "models", lambda k: good if k == "sk-good" else (_ for _ in ()).throw(_Status(401)))
    fr.ui.answers[:] = ["sk-bad", "sk-good"]
    fr.first_run_key()
    assert brain.key("anthropic") == "sk-good"
    assert "API key ไม่ถูกต้อง" in fr.ui.notes[1]  # second window says why the first key failed
    assert len(fr.ui.confirms) == 1 and fr.ui.alerts[-1].startswith("พร้อมแล้วค่ะ")


def test_three_bad_keys_save_nothing(fr, monkeypatch):
    monkeypatch.setattr(brain_claude, "models", lambda k: (_ for _ in ()).throw(_Status(401)))
    fr.ui.answers[:] = ["a", "b", "c", "never asked"]
    fr.first_run_key()
    assert brain.key("anthropic") is None and fr.ui.answers == ["never asked"] and not fr.ui.confirms
    assert "ยังไม่ได้บันทึก" in fr.ui.alerts[-1]


def test_cancel_saves_nothing(fr):
    fr.first_run_key()  # no answers: the window was closed
    assert brain.key("anthropic") is None and not fr.ui.alerts


def test_no_window_when_a_key_is_already_there(fr):
    brain.set_key("anthropic", "sk-there")
    fr.ui.answers[:] = ["sk-other"]
    fr.first_run_key()
    assert fr.ui.answers == ["sk-other"] and brain.key("anthropic") == "sk-there"


def test_turn_without_key_or_ollama_tells_the_owner(fr, monkeypatch):
    monkeypatch.setattr(brain, "OLLAMA", "http://127.0.0.1:9")
    monkeypatch.setattr(fr, "TRAY", False)
    history = []
    fr._turn(history, "สวัสดี")
    assert history == []
    told = [m["msg"]["text"] for m in fr.ui.panel if "msg" in m and m["msg"]["role"] == "tool"]
    assert any("ยังไม่ได้ใส่ API key" in t and "ใส่ API key" in t for t in told), told


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code
