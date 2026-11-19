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
