# src/models/v3/mu_feature.py
"""Hanning-window convolution for continuous firing rate (CFR) estimation.

Port of DischRate_hann.m. Hanning window (400ms) approximates muscle
twitch force temporal profile. Convolution with binary spike train
produces smooth CFR with inherent electromechanical delay and
force summation effects.
"""

import numpy as np
from scipy.signal import convolve


class MURateEncoder:
    """Convert MU spike trains to continuous firing rate via Hanning convolution.

    Algorithm (matching MATLAB DischRate_hann.m):
      1. Binary spike train: zeros, 1 at each spike time
      2. Hanning window (400ms) convolution: conv(train, hanning, 'same')
      3. Scale: / win_duration_s * 2 -> units: pulses/sec (Hz)

    The Hanning window's shape physiologically approximates the
    muscle twitch force response to a single action potential.

    Supports multiple encoding modes:
        - 'hanning': Hanning window convolution (default, matches MATLAB)
        - 'spike_count': Spike count per sliding window (no convolution)
        - 'isi_rate': ISI-based instantaneous rate (1/mean_ISI in window)
    """

    def __init__(self, fsamp=2048, hanning_ms=400, mode="hanning"):
        self.fsamp = fsamp
        self.hanning_ms = hanning_ms
        self.mode = mode
        self.win_samples = int(hanning_ms * fsamp / 1000)
        self.win_duration_s = hanning_ms / 1000.0
        # Pre-compute normalized Hanning window
        self._hanning = np.hanning(self.win_samples)
        self._hanning = self._hanning.astype(np.float32)

    def encode(self, mu_pulses, n_samples):
        """Generate continuous firing rate for each MU.

        Args:
            mu_pulses: List[np.ndarray], spike sample indices per MU
            n_samples: Total EMG signal length in samples

        Returns:
            np.ndarray [n_samples, n_mus] continuous firing rate (Hz)
            Empty array if mu_pulses is empty.
        """
        if not mu_pulses:
            return np.empty((n_samples, 0), dtype=np.float32)

        if self.mode == "spike_count":
            return self._encode_spike_count(mu_pulses, n_samples)
        elif self.mode == "isi_rate":
            return self._encode_isi_rate(mu_pulses, n_samples)
        else:
            return self._encode_hanning(mu_pulses, n_samples)

    def _encode_hanning(self, mu_pulses, n_samples):
        """Hanning window convolution (default, MATLAB-aligned)."""
        n_mus = len(mu_pulses)
        cfr = np.zeros((n_samples, n_mus), dtype=np.float32)

        for mu_idx, pulses in enumerate(mu_pulses):
            if len(pulses) == 0:
                continue
            train = np.zeros(n_samples, dtype=np.float32)
            for p in pulses:
                if 0 <= p < n_samples:
                    train[p] += 1.0
            cfr_raw = convolve(train, self._hanning, mode="same")
            cfr[:, mu_idx] = cfr_raw / self.win_duration_s * 2.0

        return cfr

    def _encode_spike_count(self, mu_pulses, n_samples):
        """Spike count encoding: count spikes in each sample window.

        Uses Hanning window width for binning, no convolution.
        """
        n_mus = len(mu_pulses)
        half_win = self.win_samples // 2
        cfr = np.zeros((n_samples, n_mus), dtype=np.float32)

        for mu_idx, pulses in enumerate(mu_pulses):
            if len(pulses) == 0:
                continue
            pulses_sorted = np.sort(pulses.astype(np.int64))
            # For each sample, count spikes in [t - half_win, t + half_win]
            for p in pulses_sorted:
                lo = max(0, p - half_win)
                hi = min(n_samples, p + half_win + 1)
                cfr[lo:hi, mu_idx] += 1.0 / self.win_duration_s

        return cfr

    def _encode_isi_rate(self, mu_pulses, n_samples):
        """ISI-based instantaneous rate: 1 / mean_ISI in local window.

        For each spike, compute the local ISI, then fill rate = 1/ISI
        for the interval between spikes.
        """
        n_mus = len(mu_pulses)
        cfr = np.zeros((n_samples, n_mus), dtype=np.float32)

        for mu_idx, pulses in enumerate(mu_pulses):
            if len(pulses) < 2:
                continue
            pulses_sorted = np.sort(pulses.astype(np.int64))
            for i in range(len(pulses_sorted) - 1):
                p0 = pulses_sorted[i]
                p1 = pulses_sorted[i + 1]
                if p1 <= p0:
                    continue
                isi_samples = p1 - p0
                rate_hz = self.fsamp / isi_samples
                # Fill the interval [p0, p1) with this rate
                lo = max(0, p0)
                hi = min(n_samples, p1)
                cfr[lo:hi, mu_idx] = rate_hz

        return cfr

    def to_windows(self, cfr, window_ms=200, step_ms=50, n_skip_start_ms=0):
        """Downsample full-rate CFR to sliding-window feature matrix.

        Aligns with EMGForceDataset window parameters so X_rate
        and y_force have identical window count and alignment.

        Args:
            cfr: [n_samples, n_mus] from encode()
            window_ms: window length (match EMGForceDataset)
            step_ms: window step (match EMGForceDataset)
            n_skip_start_ms: skip initial transient (ms)

        Returns:
            np.ndarray [n_windows, n_mus] mean CFR per window
        """
        win_samp = int(window_ms * self.fsamp / 1000)
        step_samp = int(step_ms * self.fsamp / 1000)
        skip_samp = int(n_skip_start_ms * self.fsamp / 1000)

        n_samples, n_mus = cfr.shape
        n_windows = max(1, (n_samples - skip_samp - win_samp) // step_samp + 1)
        rates = np.zeros((n_windows, n_mus), dtype=np.float32)

        for win_idx in range(n_windows):
            start = skip_samp + win_idx * step_samp
            end = start + win_samp
            rates[win_idx, :] = cfr[start:end, :].mean(axis=0)

        return rates

    def __repr__(self):
        return f"MURateEncoder(fs={self.fsamp}Hz, hanning={self.hanning_ms}ms, mode={self.mode})"


# Convenience function: spike trains -> windowed feature matrix in one call
def spikes_to_features(
    mu_pulses, n_samples, fsamp=2048, hanning_ms=400, window_ms=200, step_ms=50, cfr_mode="hanning"
):
    """End-to-end: MU spike trains -> windowed CFR feature matrix.

    Args:
        mu_pulses: List[np.ndarray] spike indices
        n_samples: signal length
        fsamp: sampling rate
        hanning_ms: Hanning window duration
        window_ms, step_ms: sliding window params (match EMGForceDataset)
        cfr_mode: encoding mode ('hanning', 'spike_count', 'isi_rate')

    Returns:
        np.ndarray [n_windows, n_mus] mean CFR per window
    """
    encoder = MURateEncoder(fsamp=fsamp, hanning_ms=hanning_ms, mode=cfr_mode)
    cfr = encoder.encode(mu_pulses, n_samples)
    return encoder.to_windows(cfr, window_ms=window_ms, step_ms=step_ms)
