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


# ---------------------------------------------------------------- the owner's own voice (C/D are on hold until this)
# FRIDAY.exe --stt-compare <folder>: records these (one window per sentence, ended by silence), then runs every model
# set on them and writes report.txt. Real voice, real mic, real room — what the TTS clips cannot show.
OWNER_SET = [
    ("th", "เปิดเพลงให้หน่อย"), ("th", "พรุ่งนี้ฝนจะตกไหม"), ("th", "ตั้งเตือนอีกสิบนาที"), ("th", "วันนี้มีงานอะไรบ้าง"),
    ("th", "ปิดโปรแกรมทั้งหมด"), ("th", "เพิ่มงานส่งรายงานวันศุกร์"), ("th", "เสียงเบาลงหน่อย"), ("th", "สลับไปหน้าต่าง vscode"),
    ("th", "จำไว้ว่าฉันแพ้ถั่ว"), ("th", "สรุปข่าวเช้านี้ให้ฟัง"), ("th", "โทรหาแม่ตอนหกโมงเย็น"), ("th", "ช่วยอ่านอีเมลล่าสุด"),
    ("en", "Open the music player."), ("en", "What is the weather tomorrow?"), ("en", "Set a timer for ten minutes."),
    ("en", "What do I have to do today?"), ("en", "Turn the volume down."),
]


def save_clip(folder, i, lang, text, audio16k):
    """One recorded sentence: NN.wav (16 kHz mono int16) + a line in refs.tsv (file, lang, text)."""
    import wave, numpy as np
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{i:02d}.wav"
    with wave.open(str(folder / name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(np.asarray(audio16k, np.float32), -1, 1) * 32767).astype(np.int16).tobytes())
    with open(folder / "refs.tsv", "a", encoding="utf-8") as f:
        f.write(f"{name}\t{lang}\t{text}\n")


def load_set(folder):
    import wave, numpy as np
    from pathlib import Path
    folder, clips = Path(folder), []
    for line in (folder / "refs.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, lang, text = line.split("\t", 2)
        with wave.open(str(folder / name)) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        clips.append(Clip(lang, text, a))
    return clips


def write_report(folder, rows, device):
    from pathlib import Path
    lines = [f"STT compare on the owner's voice · device {device} · warm timings (each set run once before timing)", ""]
    for r in rows:
        lines.append(f"{r['config']}: lang {r['lang_ok']} · Thai CER {r['th_cer']} · English WER {r['en_wer']} · "
                     f"{r['sec_per_clip']} s/clip (x{r['x_realtime']} real time)")
        lines += [f"    {s}" for s in r["samples"]]
        lines.append("")
    path = Path(folder) / "report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
