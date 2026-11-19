"""Zerem chose B for the CPU (run 44529861235): base picks the language, base.en transcribes English, Thonburian
medium Thai — 4.3x faster than turbo + Thonburian with the same transcripts on the benchmark clips. With the GPU pack
the turbo set stays. The set in use is named in the log and decides what gets downloaded."""


def test_cpu_uses_set_b(fr):
    assert fr.stt_set("cpu") == {"detect": "base", "en": "base.en", "th": fr.WHISPER_TH, "name": "B (CPU)"}


def test_gpu_keeps_the_turbo_set(fr):
    assert fr.stt_set("cuda") == {"detect": fr.WHISPER, "en": fr.WHISPER, "th": fr.WHISPER_TH, "name": "turbo (GPU)"}


def test_env_overrides_win(fr, monkeypatch):
    monkeypatch.setenv("FRIDAY_WHISPER_DETECT", "small")
    monkeypatch.setenv("FRIDAY_WHISPER_EN", "small.en")
    s = fr.stt_set("cpu")
    assert s["detect"] == "small" and s["en"] == "small.en" and "env" in s["name"]


def test_downloads_follow_the_set(fr):
    assert fr.ears_models("cpu") == ("base", "base.en", fr.WHISPER_TH)
    assert fr.ears_models("cuda") == (fr.WHISPER, fr.WHISPER_TH)  # one turbo for both jobs
