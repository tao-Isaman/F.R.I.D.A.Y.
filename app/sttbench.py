"""Speech-to-text model sets compared on the same clips (option 1: smaller models on CPU). Pure: the caller hands in
the clips and a run(config, clip) -> (lang, text) function; this scores them the same way for every config.
  Thai     character error rate (spaces and punctuation ignored — Thai has no word spaces)
  English  word error rate (lower case, no punctuation)
  language picked right, and warm seconds per second of audio (each config is run once before timing)."""
import re
from dataclasses import dataclass, field

SR = 16000


@dataclass
class Clip:
    lang: str  # "th" / "en": what was said
    ref: str   # the text the TTS was given
    audio: list = field(repr=False)


def _edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _chars(s):
    return re.sub(r"[\s\W_]+", "", s)


def cer(ref, hyp):
    r = _chars(ref)
    return _edits(r, _chars(hyp)) / max(1, len(r))


def _words(s):
    return re.sub(r"[^\w\s]", "", s.lower()).split()


def wer(ref, hyp):
    r = _words(ref)
    return _edits(r, _words(hyp)) / max(1, len(r))


def bench(configs, clips, run, now):
    rows = []
    for config in configs:
        run(config, clips[0])  # warm: load the models before timing
        lang_ok, th, en, secs, audio_s, samples = 0, [], [], 0.0, 0.0, []
        for c in clips:
            t0 = now()
            lang, text = run(config, c)
            secs += now() - t0
            audio_s += len(c.audio) / SR
            lang_ok += lang == c.lang
            (th if c.lang == "th" else en).append(cer(c.ref, text) if c.lang == "th" else wer(c.ref, text))
            samples.append(f"{c.lang}:{text!r}")
        rows.append({
            "config": config,
            "lang_ok": f"{lang_ok}/{len(clips)}",
            "th_cer": round(sum(th) / len(th), 3) if th else None,
            "en_wer": round(sum(en) / len(en), 3) if en else None,
            "sec_per_clip": round(secs / len(clips), 2),
            "x_realtime": round(secs / max(audio_s, 1e-9), 2),
            "note": "warm (models loaded before timing)",
            "samples": samples,
        })
    return rows
