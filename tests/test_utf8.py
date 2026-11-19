"""vaniratts opens its vocab JSON with open(path, 'r') — no encoding — so on Windows (cp1252) the Thai voice died
with UnicodeDecodeError (CI voice smoke, run 44523396354). The fix is Python's UTF-8 mode for FRIDAY.exe and its
worker; this reproduces the failure under a non-UTF-8 locale and shows UTF-8 mode is what fixes it."""
import json, subprocess, sys, textwrap


SCRIPT = textwrap.dedent("""
    import json, sys
    with open(sys.argv[1], 'r') as f:  # exactly how vaniratts reads vocab.json
        print(len(json.load(f)))
""")


def run(tmp_path, *flags):
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"ก": 1, "ข": 2, "ค": 3}, ensure_ascii=False), encoding="utf-8")
    env = {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, *flags, "-c", SCRIPT, str(vocab)], capture_output=True, text=True, env=env)


def test_without_utf8_mode_the_vocab_does_not_load(tmp_path):
    r = run(tmp_path, "-X", "utf8=0")
    assert r.returncode != 0 and "UnicodeDecodeError" in r.stderr


def test_utf8_mode_loads_it(tmp_path):
    r = run(tmp_path, "-X", "utf8")
    assert r.returncode == 0 and r.stdout.strip() == "3"


def test_both_exes_are_built_in_utf8_mode():
    for spec in ("friday.spec", "voice.spec"):
        src = open(spec, encoding="utf-8").read()
        # PyInstaller's spelling (doc/spec-files.rst): 'X utf8=1'. 'X utf8_mode=1' was silently ignored (run 44525010833).
        assert "('X utf8=1', None, 'OPTION')" in src, f"{spec}: FRIDAY.exe / FRIDAY-voice.exe must run in UTF-8 mode"
        assert "utf8_mode=1', None" not in src, f"{spec}: wrong option name"
