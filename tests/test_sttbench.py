"""Option 1 experiment (CPU speech-to-text): the same clips through each model set, scored the same way —
Thai by character error rate, English by word error rate, plus whether the language was picked right and the
warm time per second of audio."""
import sttbench


def test_cer_ignores_spaces_and_punctuation_and_polite_endings_are_errors():
    assert sttbench.cer("สวัสดีค่ะ วันนี้อากาศดี", "สวัสดีค่ะวันนี้อากาศดี.") == 0
    assert round(sttbench.cer("สวัสดีค่ะ", "สวัสดีครับ"), 3) == round(3 / 9, 3)  # ค่ะ -> ครับ: 3 edits / 9 chars


def test_wer_is_case_and_punctuation_blind():
    assert sttbench.wer("Open the music player", "open the music player.") == 0
    assert sttbench.wer("set a timer for ten minutes", "set a timer for 10 minutes") == 1 / 6


def test_bench_scores_each_config_on_the_same_clips():
    clips = [sttbench.Clip("th", "เปิดเพลง", [0.0] * 32000), sttbench.Clip("en", "open the door", [0.0] * 16000)]
    answers = {"good": {"th": ("th", "เปิดเพลง"), "en": ("en", "open the door")},
               "bad": {"th": ("en", "pert pleng"), "en": ("en", "open door")}}
    clock = iter(range(100))

    def run(config, clip):
        return answers[config][clip.lang]
    rows = sttbench.bench(["good", "bad"], clips, run, now=lambda: next(clock))
    good, bad = rows
    assert good["config"] == "good" and good["lang_ok"] == "2/2" and good["th_cer"] == 0 and good["en_wer"] == 0
    assert bad["lang_ok"] == "1/2" and bad["th_cer"] > 0.9 and bad["en_wer"] == round(1 / 3, 3)
    assert good["x_realtime"] > 0 and "warm" in good["note"]


def test_a_recorded_set_round_trips(tmp_path):
    import numpy as np, wave
    lines = [("th", "เปิดเพลงให้หน่อย"), ("en", "Open the music player.")]
    for i, (lang, text) in enumerate(lines, 1):
        sttbench.save_clip(tmp_path, i, lang, text, np.full(16000, 0.1, np.float32))
    clips = sttbench.load_set(tmp_path)
    assert [(c.lang, c.ref, len(c.audio)) for c in clips] == [("th", "เปิดเพลงให้หน่อย", 16000), ("en", "Open the music player.", 16000)]
    with wave.open(str(tmp_path / "01.wav")) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1


def test_owner_sentences_cover_both_languages():
    th = [t for lang, t in sttbench.OWNER_SET if lang == "th"]
    en = [t for lang, t in sttbench.OWNER_SET if lang == "en"]
    assert 10 <= len(sttbench.OWNER_SET) <= 20 and len(th) >= 10 and len(en) >= 5


def test_report_lists_every_config_and_clip(tmp_path):
    rows = [{"config": "A", "lang_ok": "2/2", "th_cer": 0.0, "en_wer": 0.0, "sec_per_clip": 3.0, "x_realtime": 1.5,
             "note": "warm", "samples": ["th:'x'", "en:'y'"]}]
    path = sttbench.write_report(tmp_path, rows, device="cpu")
    text = path.read_text(encoding="utf-8")
    assert "A" in text and "th:'x'" in text and "cpu" in text


def test_stt_compare_records_then_reports(fr, tmp_path, monkeypatch):
    import numpy as np
    asked = []
    monkeypatch.setattr(fr, "_ask_ok", lambda title, text: (asked.append(text), True)[1])
    monkeypatch.setattr(fr.wakeword, "record_utterance", lambda **kw: np.full(16000, 0.1, np.float32))
    fetched = []
    monkeypatch.setattr(fr, "_download_model", lambda m, local_files_only=False: fetched.append(m))  # no network in tests
    seen = {}
    monkeypatch.setattr(fr, "stt_bench_rows", lambda clips, bench_dir=None: (seen.setdefault("n", len(clips)), [
        {"config": "B", "lang_ok": "17/17", "th_cer": 0.0, "en_wer": 0.0, "sec_per_clip": 1.0, "x_realtime": 1.0,
         "note": "warm", "samples": []}])[1])
    report = fr.stt_compare(str(tmp_path))
    assert seen["n"] == len(sttbench.OWNER_SET) and len(asked) == len(sttbench.OWNER_SET)
    assert (tmp_path / "17.wav").exists() and "B" in open(report, encoding="utf-8").read()
    assert "base" in fetched and "base.en" in fetched  # the sets it compares are fetched first
    asked.clear()
    fr.stt_compare(str(tmp_path))  # a second run reuses the recordings: no new prompts
    assert asked == []


def test_stt_compare_stops_when_the_owner_cancels(fr, tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "_ask_ok", lambda title, text: False)
    assert fr.stt_compare(str(tmp_path)) is None and not (tmp_path / "refs.tsv").exists()
