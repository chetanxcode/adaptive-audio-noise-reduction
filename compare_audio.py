"""
compare_audio.py  -  objective comparison of a raw and a cleaned live recording
===============================================================================
The cleaned file is delayed by one block (256 samples at 16 kHz), so the two files must be
ALIGNED before comparing them. This script does that, then reports how the level changed in
the quiet parts (background noise, pauses) and in the loud parts (your speech).

    python compare_audio.py                       # newest pair in live_out
    python compare_audio.py --folder live_out
    python compare_audio.py --raw a.wav --clean b.wav
    python compare_audio.py --start 10            # ignore the first 10 s (echo canceller still learning)

What good looks like (denoise test): quiet parts clearly lower (for example -6 dB or more),
                       loud parts almost unchanged (within about 1-2 dB).
What good looks like (call/echo test): after the first seconds the cleaned level is far below
                       the raw level, because the far-end echo is removed (use --start 10).
Writes compare.png next to the files (frame level of raw and cleaned over time).
"""
import os, glob, argparse
import numpy as np
from scipy.io import wavfile


def latest_pair(folder):
    raws = sorted(glob.glob(os.path.join(folder, "mic_raw_*.wav")), key=os.path.getmtime)
    if raws:
        raw = raws[-1]
        clean = os.path.join(folder, "cleaned_" + os.path.basename(raw)[len("mic_raw_"):])
        if os.path.exists(clean):
            return raw, clean
    return os.path.join(folder, "mic_raw.wav"), os.path.join(folder, "cleaned.wav")


def read(path):
    fs, d = wavfile.read(path)
    d = d.astype(np.float64)
    if d.ndim > 1:
        d = d.mean(axis=1)
    return fs, d / (32768.0 if np.max(np.abs(d)) > 1.5 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="live_out")
    ap.add_argument("--raw"); ap.add_argument("--clean")
    ap.add_argument("--delay", type=int, default=256, help="delay of the cleaned file in samples (one block)")
    ap.add_argument("--frame", type=int, default=512)
    ap.add_argument("--start", type=float, default=0.0, help="ignore the first N seconds")
    a = ap.parse_args()
    raw, clean = (a.raw, a.clean) if a.raw and a.clean else latest_pair(a.folder)
    fs, x = read(raw); fs2, y = read(clean)
    assert fs == fs2 and len(x) == len(y), "files must have the same sampling rate and length"
    x, y = x[:len(x) - a.delay], y[a.delay:]                     # align: cleaned is late by one block
    s0 = int(a.start * fs)
    x, y = x[s0:], y[s0:]
    nf = len(x) // a.frame
    px = np.array([np.mean(x[i*a.frame:(i+1)*a.frame] ** 2) for i in range(nf)])
    py = np.array([np.mean(y[i*a.frame:(i+1)*a.frame] ** 2) for i in range(nf)])
    floor = 10 ** (-90 / 10)                                    # frames below -90 dBFS are digital silence
    valid = px > floor
    d_db = 10 * np.log10((py + 1e-20) / (px + 1e-20))
    idx = np.where(valid)[0]
    order = idx[np.argsort(px[idx])]
    k = max(1, int(0.3 * len(order)))
    quiet, loud = order[:k], order[-k:]
    db = lambda p: 10 * np.log10(p + 1e-20)
    print(f"RAW:     {raw}\nCLEANED: {clean}")
    print(f"{len(x)/fs:.1f} s analysed (from {a.start:.0f} s) at {fs} Hz, aligned by {a.delay} samples; "
          f"{nf - len(idx)} of {nf} frames are digital silence (below -90 dBFS) and are ignored\n")
    print(f"overall level          raw {db(np.mean(x**2)):6.1f} dBFS    cleaned {db(np.mean(y**2)):6.1f} dBFS")
    print(f"quietest 30% frames    raw {db(np.mean(px[quiet])):6.1f} dBFS    cleaned {db(np.mean(py[quiet])):6.1f} dBFS"
          f"    change {np.median(d_db[quiet]):+5.1f} dB (median)   <- background noise / pauses")
    print(f"loudest 30% frames     raw {db(np.mean(px[loud])):6.1f} dBFS    cleaned {db(np.mean(py[loud])):6.1f} dBFS"
          f"    change {np.median(d_db[loud]):+5.1f} dB (median)   <- your speech")
    print(f"frames lowered by more than 6 dB: {100*np.mean(d_db[idx] < -6):.0f} %")
    q, l = np.median(d_db[quiet]), np.median(d_db[loud])
    if q < -3 and abs(l) < 2:
        print("\nResult: background reduced, speech level preserved.")
    elif q >= -3:
        print("\nResult: little background reduction. Probably the recording has very little noise, or the "
              "microphone driver already removed it. Try again with a fan or music playing nearby.")
    elif l < -6:
        print("\nResult: the loud parts were lowered a lot. In a call/echo test (you stayed silent) that is the far-end "
              "echo being removed: good. In a denoise test it would mean your speech is being cut: listen carefully.")
    else:
        print("\nResult: speech level changed by more than 2 dB. Listen carefully for muffled or watery speech.")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = (np.arange(nf) + 0.5) * a.frame / fs
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(t, db(px), label="raw"); ax.plot(t, db(py), label="cleaned")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frame level (dBFS)"); ax.legend(); ax.set_ylim(-100, 0)
        fig.tight_layout(); out = os.path.join(os.path.dirname(raw) or ".", "compare.png")
        fig.savefig(out, dpi=130); plt.close(fig); print(f"Plot saved: {out}")
    except Exception as ex:
        print("(plot not saved:", ex, ")")


if __name__ == "__main__":
    main()
