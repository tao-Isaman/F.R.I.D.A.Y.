"""The owner: "หยุดพูดไปนานแล้วแต่ friday ยังขึ้นว่าฟังอยู่". The recording had ended; the panel kept saying
"listening" through speech-to-text (~20 s on CPU), and live captions held the Whisper lock the real transcript then
waited for. The panel must say what is happening, the log must time each part, and captions stay off on CPU."""
import threading, time

import numpy as np

import wakeword


def test_transcribe_records_lock_wait_and_decode_time(fr, monkeypatch):
    monkeypatch.setattr(fr, "_transcribe", lambda a, q: (time.sleep(0.05), "ok")[1])
    fr.WHISPER_LOCK.acquire()
    threading.Timer(0.1, fr.WHISPER_LOCK.release).start()  # a live caption still decoding
    assert fr.transcribe(np.zeros(16000, np.float32)) == "ok"
    wait, decode = fr.LAST_STT
    assert wait >= 0.08 and decode >= 0.04


def test_live_captions_only_on_the_gpu(fr, monkeypatch):
    monkeypatch.setattr(fr, "WHISPER_DEVICE", "cpu")
    assert fr.live_caption() is None
    monkeypatch.setattr(fr, "WHISPER_DEVICE", "cuda")
    assert isinstance(fr.live_caption(), fr.LiveCaption)


def test_panel_knows_the_transcribing_state():
    html = open("app/ui.html", encoding="utf-8").read()
    assert 'transcribing: "⏳ กำลังถอดเสียง…"' in html and 'listening: "🎤 ฟังอยู่…"' in html and 'thinking: "💭 กำลังคิด…"' in html
    assert "transcribing:" in html.split("const LOOK")[1].split("};")[0]


class _VAD:
    """Speech where the frame is loud (the clip), not speech for steady low noise — what Silero does in the smoke."""
    def reset_states(self):
        pass

    def predict(self, f, frame_size=640):
        return 0.9 if np.abs(f).mean() > 2000 else 0.1


def test_recording_ends_soon_after_speech_even_with_steady_noise(monkeypatch):
    monkeypatch.setattr(wakeword, "_load", lambda: None)
    monkeypatch.setattr(wakeword, "_vad", _VAD())
    rng = np.random.default_rng(0)
    speech_s, noise_s = 2.0, 10.0
    speech = (rng.standard_normal(int(speech_s * 16000)) * 6000).astype(np.int16)
    noise = (rng.standard_normal(int(noise_s * 16000)) * 300).astype(np.int16)  # ~ -40 dBFS fan/aircon
    a = np.concatenate([speech, noise])
    frames = list(a[: len(a) // wakeword.FRAME * wakeword.FRAME].reshape(-1, wakeword.FRAME))
    used = []
    audio = wakeword.record_utterance(frames=(used.append(f) or f for f in frames))
    ended_after = len(used) * wakeword.FRAME / 16000 - speech_s
    assert len(audio) and ended_after <= wakeword.END_SILENCE + 0.2, ended_after
