"""Prompt caching for the Claude brain: tools + the stable part of FRIDAY's system prompt are one cached prefix; what
changes every turn (time, memory notes, per-turn hints) comes after it; usage is recorded so hits can be checked."""
import datetime, types

import brain, brain_claude, security


class FakeClient:
    def __init__(self, usage=None):
        self.calls = []
        self.messages = types.SimpleNamespace(create=self.create)
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))
        self.usage = usage or types.SimpleNamespace(input_tokens=40, output_tokens=5, cache_read_input_tokens=3000,
                                                    cache_creation_input_tokens=0)

    def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(stop_reason="end_turn", usage=self.usage, content=[
            types.SimpleNamespace(type="text", text="ค่ะ", model_dump=lambda **k: {"type": "text", "text": "ค่ะ"})])


def _send(client, monkeypatch, messages, tools=None):
    monkeypatch.setattr(brain_claude, "_client", lambda key: client)
    return brain_claude.chat("k", "claude-sonnet-5-5", messages, tools, None, 1000)


def test_stable_part_is_cached_and_turn_part_is_not(monkeypatch):
    c = FakeClient()
    system = "You are FRIDAY. Rules." + "\nThe time now is 10:31."
    _send(c, monkeypatch, [{"role": "system", "content": system, "_cache_upto": len("You are FRIDAY. Rules.")},
                           {"role": "user", "content": "hi"}])
    kw = c.calls[0]
    assert kw["system"][0] == {"type": "text", "text": "You are FRIDAY. Rules.", "cache_control": {"type": "ephemeral"}}
    assert kw["system"][1] == {"type": "text", "text": "\nThe time now is 10:31."} and len(kw["system"]) == 2
    assert kw["cache_control"] == {"type": "ephemeral"}  # automatic breakpoint: the tool loop's growing tail


def test_system_without_marker_is_one_cached_block(monkeypatch):
    c = FakeClient()
    _send(c, monkeypatch, [{"role": "system", "content": "short"}, {"role": "user", "content": "hi"}])
    assert c.calls[0]["system"] == [{"type": "text", "text": "short", "cache_control": {"type": "ephemeral"}}]


def test_usage_comes_back_and_is_audited(home, monkeypatch):
    c = FakeClient()
    monkeypatch.setattr(brain_claude, "_client", lambda key: c)
    brain.set_key("anthropic", "sk-x")
    seen = []
    monkeypatch.setattr(security, "audit", lambda *a: seen.append(a))
    msg = brain.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    assert msg["_usage"] == {"input": 40, "output": 5, "cache_read": 3000, "cache_write": 0}
    assert "cache read 3000" in seen[-1][-1], seen[-1]


def test_local_brain_sees_the_same_single_system_text(home, monkeypatch):
    got = []
    monkeypatch.setattr(brain, "_ollama", lambda model, messages, *a: got.append(messages) or {"role": "assistant", "content": "ok"})
    brain.set_brain("ollama", brain.LOCAL)
    brain.chat([{"role": "system", "content": "AB", "_cache_upto": 1}, {"role": "user", "content": "hi"}])
    assert brain.clean(got[0]) == [{"role": "system", "content": "AB"}, {"role": "user", "content": "hi"}]


def test_friday_system_prefix_does_not_change_with_the_clock(fr, monkeypatch):
    sent = []

    def fake_chat(messages, tools=None, **kw):
        if tools:
            sent.append(messages[0])
        return {"role": "assistant", "content": "ค่ะ"}
    monkeypatch.setattr(brain, "chat", fake_chat)
    monkeypatch.setattr(fr, "relevant_notes", lambda user, context: "NOTES")
    monkeypatch.setattr(fr, "pick_skill", lambda user, context: "")
    real = datetime.datetime
    for minute in (1, 2):
        class Fixed(real):
            @classmethod
            def now(cls, tz=None):
                return real(2026, 11, 19, 10, minute)
        monkeypatch.setattr(fr.datetime, "datetime", Fixed)
        fr.chat([{"role": "user", "content": "สวัสดี"}])
    (a, b) = sent
    cut_a, cut_b = a["_cache_upto"], b["_cache_upto"]
    assert a["content"][:cut_a] == b["content"][:cut_b] and cut_a > 500
    assert "10:01" not in a["content"][:cut_a] and "10:01" in a["content"][cut_a:]
