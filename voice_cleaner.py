"""
voice_cleaner.py  -  block-based echo cancellation + noise suppression
=======================================================================
Works block by block, so the same code runs offline (on a file) and in real time
(inside an audio callback).  Pure NumPy; no other dependency.
 
Pipeline for every block of B samples
    mic block d(n) ------------------+
                                      v
    reference block x(n) --> [ Echo canceller ]  --> e(n)  (mic minus estimated echo)
    (what the loudspeaker plays)       |  y(n) = estimated echo
                                       v
                          [ Noise + residual-echo suppressor ] --> clean block
                          (+ finger-snap / click suppressor inside it)
 
1. EchoCanceller  : partitioned-block frequency-domain NLMS (PBFDAF, overlap-save,
                    gradient constraint) with a two-path (foreground/background)
                    structure so that near-end speech (double-talk) cannot spoil the
                    filter that produces the output.
2. PostFilter     : STFT Wiener filter (50 % overlap, sqrt-Hann window).
                    Noise PSD  : minimum tracking of the smoothed periodogram.
                    A priori SNR: decision-directed estimate.
                    Residual echo is treated as extra noise, scaled from the echo estimate.
 
3. Snap suppressor (inside PostFilter): a short, sharp, broadband click (finger snap,
                    mouse click, key tap) is not "steady noise", so the minimum-tracking noise
                    estimate cannot see it.  It is detected from its time-domain shape (one
                    very strong 2 ms burst on top of a quiet frame, mostly high-frequency) and
                    its spectrum is limited to the level of the frames just before it.
 
Latency: the post filter delays the output by exactly one block (B samples,
about 16 ms).  With ns=False the output has no delay.
"""
import numpy as np
 
EPS = 1e-12
 
 
# ----------------------------------------------------------------------------
# 1. Echo canceller
# ----------------------------------------------------------------------------
class EchoCanceller:
    """
    Two-path partitioned-block frequency-domain adaptive filter.
 
    block      : B, samples per block
    partitions : P, the filter length is P * B samples (P * B / fs seconds)
    mu         : step size, 0 < mu < 2 (0.5 is a good start)
    """
 
    def __init__(self, block, partitions, mu=0.5, far_active_power=1e-6,
                 transfer_ratio=0.8, reset_ratio=4.0):
        self.B, self.N, self.P = block, 2 * block, partitions
        K = block + 1
        self.mu = mu
        self.far_active_power = far_active_power      # mean power below this = far end silent
        self.transfer_ratio = transfer_ratio          # background better than 0.8 x foreground -> copy
        self.reset_ratio = reset_ratio                # background worse than 4 x foreground -> reset
        self.Wf = np.zeros((partitions, K), complex)  # foreground filter (produces the output)
        self.Wb = np.zeros((partitions, K), complex)  # background filter (adapts)
        self.Xh = np.zeros((partitions, K), complex)  # spectra of the last P reference blocks
        self.x_prev = np.zeros(block)
        self.Px = np.zeros(K)                         # smoothed reference power per bin
        self.Pf = 0.0                                 # smoothed error energy, foreground
        self.Pb = 0.0                                 # smoothed error energy, background
 
    def process(self, d, x):
        """d: microphone block, x: reference block (both length B). Returns (e, y)."""
        B, N = self.B, self.N
        self.Xh = np.roll(self.Xh, 1, axis=0)         # newest reference spectrum at index 0
        self.Xh[0] = np.fft.rfft(np.concatenate([self.x_prev, x]))
        self.x_prev = x.copy()
 
        yf = np.fft.irfft((self.Wf * self.Xh).sum(axis=0), N)[B:]     # overlap-save: keep last B
        yb = np.fft.irfft((self.Wb * self.Xh).sum(axis=0), N)[B:]
        ef, eb = d - yf, d - yb
        self.Pf = 0.9 * self.Pf + 0.1 * np.sum(ef ** 2)
        self.Pb = 0.9 * self.Pb + 0.1 * np.sum(eb ** 2)
 
        if np.mean(x ** 2) > self.far_active_power:                   # adapt only while far end talks
            inst = np.sum(np.abs(self.Xh) ** 2, axis=0)               # power summed over partitions
            self.Px = 0.9 * self.Px + 0.1 * inst
            pw = np.maximum(self.Px, inst)                            # never lag behind speech onsets
            delta = 0.01 * np.mean(pw) + EPS                          # regularisation
            E = np.fft.rfft(np.concatenate([np.zeros(B), eb]))
            G = self.mu * np.conj(self.Xh) * E / (pw + delta)
            g = np.fft.irfft(G, N, axis=1)
            g[:, B:] = 0.0                                            # gradient constraint
            self.Wb = self.Wb + np.fft.rfft(g, axis=1)
            if self.Pb < self.transfer_ratio * self.Pf:               # background clearly better
                self.Wf = self.Wb.copy()
                self.Pf = self.Pb
            elif self.Pb > self.reset_ratio * self.Pf:                # background has gone astray
                self.Wb = self.Wf.copy()
        return ef, yf
 
 
# ----------------------------------------------------------------------------
# 2. Noise and residual-echo suppressor
# ----------------------------------------------------------------------------
class PostFilter:
    """STFT Wiener filter with minimum-tracking noise estimate and decision-directed SNR."""
 
    def __init__(self, block, fs, g_min_db=-30.0, echo_leak=0.05, alpha_dd=0.96,
                 noise_bias=3.5, rise_db_per_s=2.0, snap=True, snap_db=12.0):
        self.B, self.N, self.K = block, 2 * block, block + 1
        # --- finger-snap / click suppressor ---------------------------------
        self.snap = snap                             # switch on/off
        self.snap_ratio = 10 ** (snap_db / 10)       # burst must be this much above the frame's typical level
        self.snap_sub = max(8, int(round(0.002 * fs)))            # 2 ms sub-frames
        self.snap_hf_bin = int(round(1500.0 * self.N / fs))       # "high frequency" = above 1.5 kHz
        self.snap_lf_bin = int(round(150.0 * self.N / fs))
        self.snap_hf_frac = 0.15                     # at least this share of the energy above 1.5 kHz
        self.snap_max_hot = max(3, int(round(0.008 / (self.snap_sub / fs))))   # burst lasts < 8 ms
        self.snap_lim_bin = int(round(400.0 * self.N / fs))   # during a snap only bins above 400 Hz are limited (keeps the voice)
        self.snap_extra = 0.1                        # extra gain (linear) on the limited bins during a snap
        self.snap_cap = 1.0                          # during a snap the spectrum may not exceed its recent level
        self.snap_prom = 10 ** (12.0 / 10)           # burst must also exceed the long-term high-frequency level by 12 dB
        self.snap_lvl = None                         # long-term mean high-frequency power (slow average)
        self.snap_hold_frames = 0                    # frames after a hit that are treated as "snap" too (ringing)
        self.snap_hold = 0
        self.snap_prev = False
        self.snap_count = 0                          # number of snaps found (for the demo)
        self.g_min = 10 ** (g_min_db / 20)           # largest attenuation allowed
        self.echo_leak = echo_leak                   # residual echo power / estimated echo power
        self.alpha_dd = alpha_dd
        self.noise_bias = noise_bias                 # compensates that a minimum under-estimates the mean
        self.rise = 10 ** (rise_db_per_s / 10 * block / fs)   # slow upward drift of the noise estimate per block
        n = np.arange(self.N)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / self.N))  # sqrt-Hann: analysis = synthesis
        self.e_prev = np.zeros(block)
        self.y_prev = np.zeros(block)
        self.tail = np.zeros(block)
        self.first = True
 
    def _is_snap(self, frame, Pyy):
        """True if this frame holds a short, sharp, broadband click.
        Three conditions, all must hold:
          1. in the high-passed signal ONE 2 ms piece is >= snap_db stronger than the typical piece
             of the frame, and only a few pieces are close to that peak (speech and fan noise are
             spread over the whole frame, a snap is not);
          2. that burst is also snap_prom (12 dB) above the long-term high-frequency level and not tiny;
          3. a fair share of the energy is high-frequency (voiced speech is mostly below 1.5 kHz)."""
        hp = frame[2:] - 2.0 * frame[1:-1] + frame[:-2]              # 2nd difference = high-pass
        S = self.snap_sub
        m = (hp[:len(hp) // S * S].reshape(-1, S) ** 2).mean(axis=1)
        peak, med = m.max(), np.median(m)
        lvl = self.snap_lvl if self.snap_lvl is not None else m.mean()
        hit = (peak >= 3e-7 and peak >= self.snap_ratio * (med + 1e-12)
               and peak >= self.snap_prom * (lvl + 1e-12)
               and np.sum(m > 0.25 * peak) <= self.snap_max_hot)    # within 6 dB of the peak
        if hit:
            hit = Pyy[self.snap_hf_bin:].sum() > self.snap_hf_frac * (Pyy[self.snap_lf_bin:].sum() + 1e-20)
        if not hit:                                                 # follow the level only when it is not a snap
            self.snap_lvl = m.mean() if self.snap_lvl is None else 0.95 * self.snap_lvl + 0.05 * m.mean()
        return hit
 
    def process(self, e, y):
        """e: block after echo cancellation, y: estimated echo block. Returns the cleaned block
        (delayed by one block)."""
        B, N = self.B, self.N
        frame = np.concatenate([self.e_prev, e]); self.e_prev = e.copy()
        yfr = np.concatenate([self.y_prev, y]); self.y_prev = y.copy()
        Y = np.fft.rfft(frame * self.win)
        Pyy = np.abs(Y) ** 2
        Pecho = np.abs(np.fft.rfft(yfr * self.win)) ** 2
 
        if self.first:
            self.Ps, self.lam, self.Pecho = Pyy.copy(), Pyy.copy() + 1e-12, Pecho.copy()
            self.S2, self.lam_tot = Pyy.copy(), Pyy + 1e-12
            self.first = False
        if self.snap:                                                    # finger-snap / click suppressor
            hit = self._is_snap(frame, Pyy)
            active = hit or self.snap_hold > 0                           # the frame after a snap is also
            if hit and not self.snap_prev:                               # treated as "snap" (ringing)
                self.snap_count += 1
            self.snap_prev = hit
            self.snap_hold = self.snap_hold_frames if hit else max(0, self.snap_hold - 1)
            if active:
                lim = np.sqrt(np.minimum(1.0, self.snap_cap * self.Ps / (Pyy + 1e-20)))
                lim[self.snap_lim_bin:] *= self.snap_extra                # push the click down further
                lim[:self.snap_lim_bin] = 1.0                            # leave the low frequencies (voice) alone
                Y = Y * lim                                              # limit to the level just before
                Pyy = np.abs(Y) ** 2
        self.Ps = 0.7 * self.Ps + 0.3 * Pyy                              # smoothed periodogram
        self.lam = np.where(self.Ps < self.lam, self.Ps, self.lam * self.rise)   # minimum tracking
        self.lam = np.maximum(self.lam, 1e-12)
        self.Pecho = 0.6 * self.Pecho + 0.4 * Pecho
 
        lam_tot = self.noise_bias * self.lam + self.echo_leak * self.Pecho + 1e-12
        gamma = Pyy / lam_tot                                            # a posteriori SNR
        xi = (self.alpha_dd * self.S2 / self.lam_tot
              + (1 - self.alpha_dd) * np.maximum(gamma - 1.0, 0.0))      # decision-directed a priori SNR
        gain = np.maximum(xi / (1.0 + xi), self.g_min)                   # Wiener gain with a floor
        S = gain * Y
        self.S2, self.lam_tot = np.abs(S) ** 2, lam_tot
 
        s = np.fft.irfft(S, N) * self.win
        out = self.tail + s[:B]
        self.tail = s[B:].copy()
        return out
 
 
# ----------------------------------------------------------------------------
# 3. Complete pipeline
# ----------------------------------------------------------------------------
class VoiceCleaner:
    """
    vc = VoiceCleaner(fs=16000)
    for each block:  clean = vc.process(mic_block, ref_block)
 
    fs        : sampling rate (Hz)
    block     : samples per block (default: 16 ms rounded to a power of two)
    filter_ms : length of the echo path the canceller can model (include the sound-card delay!)
    aec / ns  : switch echo cancellation / noise suppression on or off
    snap      : switch the finger-snap / click suppressor on or off
    snap_db   : sensitivity; a burst must be this many dB above the rest of the frame (lower = more
                sensitive, more snaps removed but more risk to hard consonants like t, k, p)
    """
 
    def __init__(self, fs=16000, block=None, filter_ms=128.0, aec=True, ns=True, mu=0.5,
                 g_min_db=-30.0, echo_leak=0.05, noise_bias=3.5, snap=True, snap_db=12.0):
        if block is None:
            block = int(2 ** np.round(np.log2(0.016 * fs)))
        self.fs, self.B = fs, block
        self.aec, self.ns = aec, ns
        partitions = max(1, int(np.ceil(filter_ms / 1000.0 * fs / block)))
        self.ec = EchoCanceller(block, partitions, mu=mu)
        self.pf = PostFilter(block, fs, g_min_db=g_min_db, echo_leak=echo_leak, noise_bias=noise_bias,
                             snap=snap, snap_db=snap_db)
        self.latency_samples = block if ns else 0
 
    def process(self, mic, ref=None):
        mic = np.asarray(mic, dtype=np.float64)
        if mic.shape[0] != self.B:
            raise ValueError(f"expected blocks of {self.B} samples, got {mic.shape[0]}")
        if self.aec and ref is not None:
            e, y = self.ec.process(mic, np.asarray(ref, dtype=np.float64))
        else:
            e, y = mic, np.zeros(self.B)
        return self.pf.process(e, y) if self.ns else e
 
 
def process_signal(mic, ref=None, **kw):
    """Run a whole signal through VoiceCleaner block by block (offline helper).
    Returns (output, VoiceCleaner).  The output has the same length as `mic`."""
    vc = VoiceCleaner(**kw)
    B = vc.B
    n = len(mic)
    nb = int(np.ceil(n / B))
    m = np.zeros(nb * B); m[:n] = mic
    r = None
    if ref is not None:
        r = np.zeros(nb * B); r[:min(n, len(ref))] = ref[:n]
    out = np.zeros(nb * B)
    for i in range(nb):
        sl = slice(i * B, (i + 1) * B)
        out[sl] = vc.process(m[sl], None if r is None else r[sl])
    return out[:n], vc
 