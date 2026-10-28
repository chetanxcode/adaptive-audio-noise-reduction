"""
evaluate.py  -  measure how well voice_cleaner.py removes echo and noise
=======================================================================
Builds test recordings from real speech, runs the cleaner on them, and prints:
  Test 1  echo only                 -> echo reduction (dB)
  Test 2  speech + noise (no echo)  -> SI-SDR before / after, for 3 noise types x 3 SNRs
  Test 3  full call: far-end echo + near-end speech + car-like noise, with double-talk
                                    -> SI-SDR for: no processing / noise suppression only /
                                       echo cancellation only / both
  Test 4  the room changes (echo path replaced after 20 s) -> echo reduction over time
  Test 5  double-talk (near-end speech for 20 s over the echo) -> echo reduction during / after

SI-SDR (scale-invariant signal-to-distortion ratio, dB) compares the output with the clean
near-end speech; higher is better.  It is an objective measure, not a listening test.

Usage
    python evaluate.py --data data --out eval_out     # data = folder with sp01.wav ... sp20.wav (NOIZEUS, 8 kHz)
    python evaluate.py --out eval_out                 # no data: synthetic speech-like signals
"""
import os, argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import voice_cleaner as vcm

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# test material
# ----------------------------------------------------------------------------
def synthetic_sentence(fs, seconds, seed):
    r = np.random.default_rng(seed)
    n = int(fs * seconds)
    x = lfilter([1.0], [1.0, -1.2, 0.5], r.standard_normal(n))
    t = np.arange(n) / fs
    env = np.clip(0.55 + 0.45 * np.sin(2 * np.pi * (2.5 + r.random()) * t) * np.sin(2 * np.pi * 0.5 * t + r.random() * 6), 0.02, None)
    x = x * env
    return x / np.max(np.abs(x)) * 0.5


def load_pool(data_dir):
    """Returns (pool dict {1..20: signal}, fs)."""
    if data_dir and os.path.exists(os.path.join(data_dir, "sp01.wav")):
        pool, fs = {}, None
        for k in range(1, 21):
            fs, d = wavfile.read(os.path.join(data_dir, f"sp{k:02d}.wav"))
            pool[k] = d.astype(np.float64) / 32768.0
        return pool, fs
    print("(no data folder found: using synthetic speech-like signals)")
    fs = 8000
    return {k: synthetic_sentence(fs, 2.5, k) for k in range(1, 21)}, fs


def cat(pool, ids):
    return np.concatenate([pool[k] for k in ids])


def noise_signal(kind, n, pool, seed=1):
    r = np.random.default_rng(seed)
    if kind == "white":
        v = r.standard_normal(n)
    elif kind == "car-like":                      # low-frequency rumble
        v = lfilter([1.0], [1.0, -0.97], r.standard_normal(n))
    elif kind == "babble":                        # six overlapping sentences (sp01-sp08)
        base = cat(pool, range(1, 9))
        v = np.zeros(n)
        for s in range(6):
            sh = r.integers(0, len(base))
            seg = np.roll(base, sh)
            seg = np.tile(seg, int(np.ceil(n / len(seg))))[:n]
            v += seg
    else:
        raise ValueError(kind)
    return v - np.mean(v)


def add_noise_at_snr(clean, noise, snr_db):
    act = np.abs(clean) > 1e-6
    p_s = np.mean(clean[act] ** 2) if np.any(act) else np.mean(clean ** 2)
    v = noise[:len(clean)] * np.sqrt(p_s / 10 ** (snr_db / 10) / np.mean(noise[:len(clean)] ** 2))
    return clean + v, v


def echo_path(fs, delay_ms=20, tail_ms=100, gain=0.5, seed=7):
    r = np.random.default_rng(seed)
    L = int(tail_ms / 1000 * fs)
    h = r.standard_normal(L) * np.exp(-np.arange(L) / (0.35 * L))
    h = np.concatenate([np.zeros(int(delay_ms / 1000 * fs)), h])
    return h / np.linalg.norm(h) * gain


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def si_sdr(ref, est):
    ref = ref - np.mean(ref); est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    tgt = alpha * ref
    return 10 * np.log10(np.sum(tgt ** 2) / (np.sum((est - tgt) ** 2) + 1e-20) + 1e-20)


def best_lag(ref, est, max_lag):
    """Smallest delay l in [0, max_lag] such that est[n + l] best matches ref[n]."""
    n = min(len(ref), len(est))
    best, arg = -np.inf, 0
    for l in range(max_lag + 1):
        c = np.dot(ref[:n - l], est[l:n])
        if c > best:
            best, arg = c, l
    return arg


def score(ref, est, start, max_lag=0):
    """SI-SDR of est against ref over samples >= start, after compensating the delay of est."""
    lag = best_lag(ref[start:], est[start:], max_lag) if max_lag else 0
    e = est[lag:]; r = ref[:len(e)]
    return si_sdr(r[start:], e[start:])


def wav(path, x, fs):
    wavfile.write(path, fs, (np.clip(x / (np.max(np.abs(x)) + 1e-12) * 0.9, -1, 1) * 32767).astype(np.int16))


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test1_echo_only(pool, fs, h):
    far = cat(pool, range(1, 9))
    echo = lfilter(h, [1.0], far)
    a = len(far) // 2
    res = []
    for name, kw in [("echo canceller only", dict(ns=False)), ("echo canceller + post filter", dict())]:
        out, vc = vcm.process_signal(echo, far, fs=fs, **kw)
        res.append((name, 10 * np.log10(np.sum(echo[a:] ** 2) / np.sum(out[a:] ** 2))))
    print("\nTest 1. Echo only (no near-end speech, no noise): echo reduction over the second half")
    for n, v in res:
        print(f"   {n:32s} {v:5.1f} dB")


def test2_noise_only(pool, fs, outdir):
    near = cat(pool, range(13, 21))
    print("\nTest 2. Near-end speech + noise, no echo: SI-SDR (dB) before -> after  (improvement)")
    print("   noise       SNR 0 dB                SNR 5 dB                SNR 10 dB")
    for kind in ["white", "car-like", "babble"]:
        cells = []
        for snr in [0, 5, 10]:
            mic, _ = add_noise_at_snr(near, noise_signal(kind, len(near), pool), snr)
            out, vc = vcm.process_signal(mic, None, fs=fs, aec=False)
            before = si_sdr(near, mic)
            after = score(near, out, 0, max_lag=4 * vc.B)
            cells.append(f"{before:5.1f} -> {after:5.1f} ({after-before:+5.1f})")
            if kind == "car-like" and snr == 5:
                wav(os.path.join(outdir, "test2_noisy_input.wav"), mic, fs)
                wav(os.path.join(outdir, "test2_cleaned.wav"), out, fs)
        print(f"   {kind:9s}  " + "   ".join(cells))


def test3_full_call(pool, fs, h, outdir):
    lead = 6 * fs                                           # 6 s of echo only, then the near-end talker joins
    near_speech = cat(pool, range(13, 21))
    near = np.concatenate([np.zeros(lead), near_speech])
    far = cat(pool, range(1, 13))
    far = np.tile(far, int(np.ceil(len(near) / len(far))))[:len(near)]
    echo = lfilter(h, [1.0], far)
    mic_nn, v = add_noise_at_snr(near, noise_signal("car-like", len(near), pool, seed=3), 10)
    mic = mic_nn + echo
    print(f"\nTest 3. Full call: far-end echo + near-end speech + car-like noise (10 dB SNR), "
          f"double-talk from t = 6 s; SI-SDR (dB) from t = 8 s")
    start = lead + 2 * fs
    rows = [("no processing", mic, None)]
    for name, kw in [("noise suppression only", dict(aec=False)), ("echo cancellation only", dict(ns=False)),
                     ("echo cancellation + noise suppression", dict())]:
        out, vc = vcm.process_signal(mic, far, fs=fs, **kw)
        rows.append((name, out, vc))
    for name, out, vc in rows:
        lag = vc.latency_samples if vc is not None else 0
        s = score(near, out, start, max_lag=lag)
        print(f"   {name:40s} {s:6.1f} dB")
    both = rows[-1][1]
    ec_only = rows[3][1]
    wav(os.path.join(outdir, "test3_mic_with_echo_and_noise.wav"), mic, fs)
    wav(os.path.join(outdir, "test3_cleaned.wav"), both, fs)
    wav(os.path.join(outdir, "test3_clean_reference.wav"), near, fs)
    try:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for a, (t, s) in zip(ax, [("Microphone: near-end speech + echo + noise", mic),
                                  ("After echo cancellation + noise suppression", both),
                                  ("Clean near-end speech (reference)", near)]):
            a.specgram(s, NFFT=256, Fs=fs, noverlap=192, vmin=-120, vmax=-30, cmap="magma")
            a.set_title(t); a.set_ylabel("Hz")
            a.axvline(6, color="w", ls="--", lw=0.8)
        ax[-1].set_xlabel("Time (s)  (dashed line: near-end talker starts)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "test3_spectrograms.png"), dpi=130); plt.close(fig)
    except Exception as ex:
        print("   (spectrogram not saved:", ex, ")")


def test4_room_change(pool, fs):
    far = np.tile(cat(pool, range(1, 13)), 2)[:40 * fs]
    h1, h2 = echo_path(fs, seed=7), echo_path(fs, seed=21)
    sw = 20 * fs
    echo = np.concatenate([lfilter(h1, [1.0], far)[:sw], lfilter(h2, [1.0], far)[sw:]])
    out, vc = vcm.process_signal(echo, far, fs=fs, ns=False)
    print("\nTest 4. The room changes at t = 20 s (echo only, echo canceller alone): echo reduction per 2 s window")
    cells = []
    for t0 in [18, 20, 22, 24, 26, 28, 30, 38]:
        a, b = t0 * fs, (t0 + 2) * fs
        cells.append(f"{t0}-{t0+2} s: {10*np.log10(np.sum(echo[a:b]**2)/np.sum(out[a:b]**2)):5.1f} dB")
    print("   " + "   ".join(cells[:4])); print("   " + "   ".join(cells[4:]))


def test5_double_talk(pool, fs, h):
    far = np.tile(cat(pool, range(1, 13)), 2)[:40 * fs]
    echo = lfilter(h, [1.0], far)
    n1 = cat(pool, range(13, 21))
    near = np.zeros(len(far)); near[12 * fs:12 * fs + len(n1)] = n1[:len(far) - 12 * fs]
    out, vc = vcm.process_signal(echo + near, far, fs=fs, ns=False)
    end = 12 * fs + len(n1)
    res = out - near
    a, b = 16 * fs, 22 * fs
    print(f"\nTest 5. Double-talk: near-end speech from 12 s to {end/fs:.1f} s over the echo (echo canceller alone)")
    print(f"   echo reduction during double-talk (16-22 s): {10*np.log10(np.sum(echo[a:b]**2)/np.sum(res[a:b]**2)):.1f} dB")
    a, b = end + 2 * fs, end + 4 * fs
    print(f"   echo reduction 2-4 s after double-talk:      {10*np.log10(np.sum(echo[a:b]**2)/np.sum(out[a:b]**2)):.1f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pool, fs = load_pool(a.data)
    h = echo_path(fs)
    print(f"fs = {fs} Hz, block = {vcm.VoiceCleaner(fs=fs).B} samples, echo path = {len(h)/fs*1000:.0f} ms "
          f"(20 ms delay + 100 ms tail), canceller length = 128 ms")
    test1_echo_only(pool, fs, h)
    test2_noise_only(pool, fs, a.out)
    test3_full_call(pool, fs, h, a.out)
    test4_room_change(pool, fs)
    test5_double_talk(pool, fs, h)
    print(f"\nAudio and spectrogram saved in: {a.out}")
