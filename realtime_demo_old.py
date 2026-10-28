"""
realtime_demo.py  -  run the voice cleaner live on your microphone
===================================================================
Modes
  denoise : microphone -> noise suppression -> your HEADPHONES (and saved to a file).
            Use headphones: with speakers the cleaned voice would feed back into the mic.
  call    : plays a "far-end" sound file through the speakers while you talk; the cleaner
            removes the far-end echo and the noise from your microphone signal and saves
            the cleaned voice (what the person at the far end would hear).
  selftest: no sound card needed. Simulates a loudspeaker, a room and a microphone and
            checks that the real-time code path works.
 
Examples
    python realtime_demo.py --list-devices
    python realtime_demo.py --mode denoise --seconds 20
    python realtime_demo.py --mode call --far far_end.wav --seconds 30 --filter-ms 256
    python realtime_demo.py --mode selftest
 
Results are saved in the --out folder as mic_raw_<time>.wav and cleaned_<time>.wav (a new pair for every run;
the cleaned file is delayed by one block, about 16 ms). Compare them with: python compare_audio.py
"""
import os, sys, time, argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, lfilter
from voice_cleaner import VoiceCleaner
 
 
class LiveProcessor:
    """The logic of one audio callback. Kept separate from the sound card so it can be tested."""
 
    def __init__(self, vc, mode, far=None, monitor=True):
        self.vc, self.mode, self.far, self.monitor = vc, mode, far, monitor
        self.pos = 0
        self.mic_log, self.out_log = [], []
        self.in_pow = self.out_pow = 0.0
 
    def next_far(self, B):                       # far-end file, looped
        idx = (self.pos + np.arange(B)) % len(self.far)
        self.pos += B
        return self.far[idx]
 
    def handle(self, indata, outdata):
        """indata: (B, 1) microphone block; outdata: (B, 1) block to play."""
        mic = np.asarray(indata[:, 0], dtype=np.float64)
        if self.mode == "call":
            ref = self.next_far(len(mic))        # what we play now is the echo reference
            clean = self.vc.process(mic, ref)
            outdata[:, 0] = ref
        else:
            clean = self.vc.process(mic, None)
            outdata[:, 0] = np.clip(clean, -1, 1) if self.monitor else 0.0
        self.mic_log.append(mic.copy()); self.out_log.append(clean.copy())
        self.in_pow = 0.9 * self.in_pow + 0.1 * np.mean(mic ** 2)
        self.out_pow = 0.9 * self.out_pow + 0.1 * np.mean(clean ** 2)
 
    def save(self, folder, fs):
        """Save with a time stamp, so earlier recordings are never overwritten."""
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        paths = {}
        for label, log in [("mic_raw", self.mic_log), ("cleaned", self.out_log)]:
            x = np.concatenate(log)
            paths[label] = os.path.join(folder, f"{label}_{stamp}.wav")
            wavfile.write(paths[label], fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        print(f"Saved raw audio:     {paths['mic_raw']}")
        print(f"Saved cleaned audio: {paths['cleaned']}")
 
 
def db(p):
    return 10 * np.log10(p + 1e-12)
 
 
def parse_device(value):
    """--device 3 (one device)  or  --device 1,5 (input id, output id)  or a name."""
    if value is None:
        return None
    def one(v):
        v = v.strip()
        return int(v) if v.lstrip("-").isdigit() else v
    parts = value.split(",")
    return one(parts[0]) if len(parts) == 1 else (one(parts[0]), one(parts[1]))
 
 
def run_live(args):
    import sounddevice as sd
    fs = args.fs
    vc = VoiceCleaner(fs=fs, filter_ms=args.filter_ms, aec=(args.mode == "call"))
    far = None
    if args.mode == "call":
        if not args.far:
            sys.exit("call mode needs --far far_end.wav")
        f_in, far = wavfile.read(args.far)
        far = far.astype(np.float64)
        far = far.mean(axis=1) if far.ndim > 1 else far
        far = far / (np.max(np.abs(far)) + 1e-12) * 0.5
        if f_in != fs:
            g = np.gcd(int(f_in), fs); far = resample_poly(far, fs // g, int(f_in) // g)
    lp = LiveProcessor(vc, args.mode, far, monitor=not args.no_monitor)
 
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        lp.handle(indata, outdata)
 
    print(f"Starting: mode={args.mode}, fs={fs} Hz, block={vc.B} samples ({1000*vc.B/fs:.0f} ms), "
          f"filter={args.filter_ms:.0f} ms, {args.seconds:.0f} s.  Press Ctrl+C to stop early.")
    if args.mode == "denoise" and not args.no_monitor:
        print("USE HEADPHONES (or add --no-monitor), otherwise the output feeds back into the microphone.")
    try:
        with sd.Stream(samplerate=fs, blocksize=vc.B, channels=1, dtype="float32",
                       device=parse_device(args.device), callback=callback):
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                time.sleep(1.0)
                print(f"  t={time.time()-t0:4.0f}s   mic level {db(lp.in_pow):6.1f} dB   cleaned level {db(lp.out_pow):6.1f} dB")
    except KeyboardInterrupt:
        pass
    lp.save(args.out, fs)
 
 
def selftest(args):
    """Simulated loudspeaker -> room -> microphone loop through the same code path as live mode."""
    fs = args.fs
    rng = np.random.default_rng(0)
    vc = VoiceCleaner(fs=fs, filter_ms=args.filter_ms)
    B = vc.B
    n = int(10 * fs) // B * B
    t = np.arange(n) / fs
    env = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t) * np.sin(2 * np.pi * 0.4 * t), 0.02, None)
    far = lfilter([1.0], [1.0, -1.2, 0.5], rng.standard_normal(n)) * env
    far = far / np.max(np.abs(far)) * 0.5
    L = int(0.1 * fs)
    h = np.concatenate([np.zeros(int(0.02 * fs)), rng.standard_normal(L) * np.exp(-np.arange(L) / (0.35 * L))])
    h = h / np.linalg.norm(h) * 0.5
    lp = LiveProcessor(vc, "call", far)
    zi = np.zeros(len(h) - 1)
    echo_all = []
    for i in range(n // B):
        ref = far[i * B:(i + 1) * B]                 # what the loudspeaker plays in this block
        echo, zi = lfilter(h, [1.0], ref, zi=zi)     # the room
        mic = echo + 0.003 * rng.standard_normal(B)  # the microphone (plus a little noise)
        indata = mic.reshape(-1, 1).astype(np.float32)
        outdata = np.zeros((B, 1), dtype=np.float32)
        lp.handle(indata, outdata)
        echo_all.append(echo)
        assert np.allclose(outdata[:, 0], ref, atol=1e-6)    # the far end is played unchanged
    echo = np.concatenate(echo_all)
    mic = np.concatenate(lp.mic_log); out = np.concatenate(lp.out_log)
    a = n // 2
    red = db(np.sum(echo[a:] ** 2)) - db(np.sum(out[a:] ** 2))
    print(f"selftest: block = {B} samples, filter = {vc.ec.P * B} taps; "
          f"echo level reduced by {red:.1f} dB over the second half (expected > 25 dB)")
    print("PASS" if red > 25 else "FAIL")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["denoise", "call", "selftest"], default="denoise")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--filter-ms", type=float, default=256.0, help="echo path length to model (include sound-card delay)")
    ap.add_argument("--far", help="far-end sound file for call mode")
    ap.add_argument("--device", default=None, help="device id or name; use 1,5 for input 1 and output 5")
    ap.add_argument("--no-monitor", action="store_true", help="denoise mode: do not play the cleaned voice")
    ap.add_argument("--out", default="live_out")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        import sounddevice as sd; print(sd.query_devices()); sys.exit()
    selftest(args) if args.mode == "selftest" else run_live(args)
 