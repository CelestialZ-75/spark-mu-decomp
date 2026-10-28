# src/models/v3/ckc_decomposer.py
"""CKC blind source separation for motor unit spike train extraction.

Port of IPTExtraction_gCKC4GUI_v3.m and segmentDecomp.m to PyTorch
with GPU acceleration. Extracts MU discharge patterns from extended
high-density sEMG through iterative gradient descent.
"""

import gc
import time
import warnings

import numpy as np
import torch


def _get_best_cuda_device():
    """Select the NVIDIA discrete GPU with most VRAM.
    Filters out integrated GPUs (Intel/AMD) by vendor name.
    Falls back to CPU if no valid CUDA GPU found.
    """
    if not torch.cuda.is_available():
        return "cpu"
    best_idx = 0
    best_mem = 0
    found_nvidia = False
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_properties(i).name
        if (
            "NVIDIA" not in name
            and "GeForce" not in name
            and "RTX" not in name
            and "Quadro" not in name
        ):
            continue  # skip integrated/non-NVIDIA GPU
        found_nvidia = True
        mem = torch.cuda.get_device_properties(i).total_memory
        if mem > best_mem:
            best_mem = mem
            best_idx = i
    if not found_nvidia:
        # Fallback: no NVIDIA GPU detected, use cuda:0
        pass
    return f"cuda:{best_idx}"


# ===========================================================================
# Helper functions
# ===========================================================================


def _activity_index(eSig, invCorrSig, block_size=5000):
    """Compute MU activity index via block diagonal products.

    Port of calcActivityIndex.m.
    Computes: Ind[t] = eSig[:,t]^T @ invCorrSig @ eSig[:,t]
    Done in blocks to avoid O(N^2) memory.

    Args:
        eSig: [M, N] extended signal on GPU
        invCorrSig: [M, M] inverse correlation matrix on GPU
        block_size: samples per block

    Returns:
        torch.Tensor [N] activity index on GPU
    """
    _, N = eSig.shape
    act_ind = torch.zeros(N, dtype=torch.float32, device=eSig.device)

    for k in range(0, N, block_size):
        end = min(k + block_size, N)
        block = eSig[:, k:end]  # [M, block]
        tmp = invCorrSig @ block  # [M, block]
        act_ind[k:end] = (block * tmp).sum(dim=0)  # [block]

    return act_ind


def _find_hill_limits(signal, peak_idx, max_half_width=100):
    """Find peak region boundaries (hill climbing from peak).

    Port of findHillLim.m. Finds the local peak near peak_idx, then
    finds the local minima on both sides.

    Args:
        signal: 1D array-like (numpy or torch.Tensor)
        peak_idx: int, initial peak location

    Returns:
        (min_idx, max_idx) inclusive boundaries
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    N = len(signal)

    # 1. Climb uphill to local maximum
    tmp = np.diff(signal)
    if peak_idx >= len(tmp) or abs(tmp[peak_idx]) < 1e-12:
        return peak_idx, peak_idx

    dx = 1 if tmp[peak_idx] > 0 else -1
    newx = peak_idx
    while 0 < newx < N - 1 and signal[newx + dx] > signal[newx]:
        newx += dx

    # 2. Find valleys on both sides
    min_idx = newx
    max_idx = newx
    while min_idx > 0 and signal[min_idx - 1] < signal[min_idx] and newx - min_idx < max_half_width:
        min_idx -= 1
    while (
        max_idx < N - 1
        and signal[max_idx + 1] < signal[max_idx]
        and max_idx - newx < max_half_width
    ):
        max_idx += 1

    return min_idx, max_idx


def _rem_repeated_ind(ipt_seq, pulses, limit):
    """Port of remRepeatedInd.m: remove pulses within limit by keeping max amplitude."""
    if len(pulses) < 2:
        return pulses

    pulses = np.sort(pulses)
    r = len(pulses)
    new_pulses = [-1e9]
    pulse_step = 0
    k = 0

    while k < r:
        # Find first candidate far enough from last added pulse
        while k < r and (pulses[k] - new_pulses[pulse_step]) < limit:
            k += 1
        if k >= r:
            break

        first = k
        # Find all pulses within limit of the FIRST pulse in cluster
        while k + 1 < r and (pulses[k + 1] - pulses[first]) < limit:
            k += 1

        # Select best index in cluster
        cluster = pulses[first : k + 1]
        best_idx = cluster[np.argmax(ipt_seq[cluster])]
        new_pulses.append(best_idx)
        pulse_step += 1
        k += 1

    return np.array(new_pulses[1:], dtype=int)


def _kmeans_1d_spike_extraction(ipt_signal, fsamp, pulse_min_distance_ms=10):
    """Extract spike indices via 1D K-means (k=2) clustering.

    Port of spikeExtraction.m. Clusters IPT values into spike and
    noise classes, returns spike class indices.

    Args:
        ipt_signal: 1D torch.Tensor, IPT values (positive only)
        fsamp: sampling rate
        pulse_min_distance_ms: minimum inter-spike interval

    Returns:
        (spike_indices, sil_value)
    """
    vals = ipt_signal.cpu().numpy()
    n = len(vals)

    # K-means with k=2, init: (mean, 0)
    c1 = vals.mean()
    if c1 < 1e-10:
        return np.array([], dtype=int), 0.0

    c0 = 0.0
    prev_c1, prev_c0 = float("inf"), float("inf")

    for _ in range(20):
        d1 = np.abs(vals - c1)
        d0 = np.abs(vals - c0)
        labels = (d1 < d0).astype(int)
        new_c1 = vals[labels == 1].mean() if labels.sum() > 0 else c1
        new_c0 = vals[labels == 0].mean() if (1 - labels).sum() > 0 else c0
        if abs(new_c1 - prev_c1) < 1e-8 and abs(new_c0 - prev_c0) < 1e-8:
            break
        prev_c1, prev_c0 = new_c1, new_c0
        c1, c0 = new_c1, new_c0

    spike_class = 1 if c1 > c0 else 0
    spike_candidates = np.where(labels == spike_class)[0]

    if len(spike_candidates) == 0:
        return np.array([], dtype=int), 0.0

    # Remove duplicates within min distance (using remRepeatedInd logic)
    min_dist = int(fsamp * pulse_min_distance_ms / 1000)
    spikes = _rem_repeated_ind(vals, spike_candidates, min_dist)

    # Recompute centroids and compute SIL (silhouette)
    c_spike = vals[spikes].mean() if len(spikes) > 0 else c1
    noise_mask = np.ones(n, dtype=bool)
    noise_mask[spikes] = False
    c_noise = vals[noise_mask].mean() if noise_mask.any() else c0

    D_spike = np.abs(vals - c_spike)
    D_noise = np.abs(vals - c_noise)
    flags = np.ones(n)
    flags[spikes] = -1
    sil = np.mean(((D_spike - D_noise) * flags) / np.maximum(np.maximum(D_spike, D_noise), 1e-10))

    return spikes, sil


def _rate_of_agreement(pulses_a, pulses_b, lim, d_ipi):
    """Compute overlap ratio between two spike trains with lag search.

    Port of RoA.m. Searches for optimal lag +/- lim, then sums matches
    within jitter +/- d_ipi.

    Args:
        pulses_a, pulses_b: spike index arrays
        lim: search window for global lag alignment
        d_ipi: jitter window for pulse matching

    Returns:
        float [0, 1] overlap ratio
    """
    if len(pulses_a) == 0 or len(pulses_b) == 0:
        return 0.0

    # Cross-correlation of pulses
    # xcor(k+Lim+1) = length(intersect(Ind1,Ind2+k))
    set_a = set(pulses_a)
    xcor = np.zeros(2 * lim + 1, dtype=int)
    for k in range(-lim, lim + 1):
        # Optimized intersection count
        count = 0
        for p in pulses_b:
            if (p + k) in set_a:
                count += 1
        xcor[k + lim] = count

    # Find optimal lag
    max_overlap = xcor.max()
    if max_overlap == 0:
        return 0.0

    a_lags = np.where(xcor == max_overlap)[0]
    if len(a_lags) > 1:
        # Pick lag closest to zero
        a_opt = a_lags[np.argmin(np.abs(a_lags - lim))]
    else:
        a_opt = a_lags[0]

    # Sum matches within jitter around optimal lag
    matches = xcor[max(0, a_opt - d_ipi) : min(len(xcor), a_opt + d_ipi + 1)].sum()
    matches = min(matches, len(pulses_a), len(pulses_b))

    # MATLAB RoA.m: matches / (len(A) + len(B) - matches)
    denom = len(pulses_a) + len(pulses_b) - matches
    return float(matches) / max(denom, 1)


# ===========================================================================
# CKCDecomposer
# ===========================================================================


class CKCDecomposer:
    """CKC blind source separation: extract MU spike trains from extended EMG.

    Port of IPTExtraction_gCKC4GUI_v3.m to PyTorch with GPU acceleration.
    All core matrix operations run on GPU via cuBLAS.

    Usage:
        decomposer = CKCDecomposer(fsamp=2048, n_mu_max=40, device='cuda')
        decomposer.fit(eSig)   # eSig from CKCExtender.extend()
        print(f"MUs: {decomposer.mu_count}")
        decomposer.save('ckc_result.pt')
    """

    def __init__(
        self,
        fsamp=2048,
        ext_factor=10,
        n_mu_max=40,
        n_iter=45,
        cost_fcn=3,
        sil_threshold=0.8,
        activation_sigma=3.0,
        edge_point_factor=10,
        device="cuda",
    ):
        self.fsamp = fsamp
        self.ext_factor = ext_factor
        self.n_mu_max = n_mu_max
        self.n_iter = n_iter
        self.cost_fcn = cost_fcn
        self.sil_threshold = sil_threshold
        self.activation_sigma = activation_sigma
        self.edge_point = ext_factor * edge_point_factor
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and "cuda" in str(device)
            else torch.device("cpu")
        )

        # Results (populated by fit)
        self.mu_pulses = []
        self.separation_vectors = None  # [n_mus, M]
        self.ipts = []  # [n_mus, N] squared IPT signals
        self.pnrs = []  # [n_mus]
        self.sils = []  # [n_mus]
        self.centroids = None  # [n_mus, 2]
        self._inv_corr_sig = None  # [M, M]
        self._eSig = None

    @property
    def mu_count(self):
        return len(self.mu_pulses)

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def fit(self, eSig, ops=None):
        """Run CKC decomposition on extended EMG.

        Args:
            eSig: torch.Tensor [n_ext_ch, n_samples] already on device.
                  Typically output of CKCExtender.extend().
            ops: Optional Spark block operations; None keeps native Torch math.

        Returns:
            self
        """
        self._eSig = eSig
        M, N = eSig.shape
        edge = self.edge_point
        step_size = 1.0
        s_alpha = 0.1
        iterate_mod = 20

        print(f"  CKC: {M} extended channels, {N} samples")

        # 1. Correlation matrix + pseudoinverse (GPU)
        t0 = time.time()
        corr_sig = eSig @ eSig.T / N if ops is None else ops.correlation()
        self._inv_corr_sig = torch.linalg.pinv(corr_sig)
        print(f"    CorrSig+pinv: {time.time() - t0:.2f}s")

        # 2. Activity index
        t0 = time.time()
        act_whole = _activity_index(eSig, self._inv_corr_sig, block_size=5000)

        # Frequency-domain filtering: keep low frequencies (< 10% fsamp)
        # Match MATLAB: fftshift -> zero out <40% and >60% -> ifftshift
        act_fft = torch.fft.fftshift(torch.fft.fft(act_whole))
        n_fft = len(act_fft)
        lo = int(n_fft * 0.40)
        hi = int(n_fft * 0.60)
        act_fft[:lo] = 0
        act_fft[hi:] = 0
        act_resid = torch.fft.ifft(torch.fft.ifftshift(act_fft)).real

        # Zero edge points
        act_resid[:edge] = 0
        act_resid[-edge:] = 0

        # Threshold: remove below mean - N*sigma
        mean_act = act_whole.mean()
        std_act = act_whole.std()
        act_resid[act_resid < mean_act - self.activation_sigma * std_act] = 0

        # Remove peaks above mean + N*sigma
        tmp = torch.where(act_resid > mean_act + self.activation_sigma * std_act)[0]
        prohibited = list(tmp.cpu().numpy())
        for p in prohibited:
            lo_p, hi_p = _find_hill_limits(act_resid.cpu().numpy(), p)
            act_resid[lo_p : hi_p + 1] = 0

        print(f"    Activity index: {time.time() - t0:.2f}s")

        # 3. Iterative MU extraction
        self.mu_pulses = []
        self.pnrs = []
        self.sils = []
        sep_vecs = []
        centroids_list = []

        t_decomp = time.time()
        for run_ipt in range(self.n_mu_max):
            # Locate peak in residual activity
            n0 = act_resid.argmax().item()
            if act_resid[n0] <= 0:
                break

            _, peak_hi = _find_hill_limits(act_resid.cpu().numpy(), n0)
            act_resid[max(0, n0 - 10) : min(N, peak_hi + 10)] = 0

            # Init separation vector
            c_tx = eSig[:, n0].clone()
            c_tx = c_tx / torch.sqrt((c_tx**2).sum())
            a = c_tx @ self._inv_corr_sig
            t_new = (a @ eSig if ops is None else ops.project(a)).squeeze()
            t_new[:edge] = 0
            t_new[-edge:] = 0

            # Gradient descent loop
            similarity_flag = 0
            for run_c in range(self.n_iter):
                t_old = t_new.clone()

                # Cost function gradient
                weights = self._cost_weights(t_new)

                # Periodic spike extraction
                if run_c % iterate_mod == 0 and run_c > 0:
                    for _ in range(5):
                        t_new[: edge + 1] = 0
                        t_new[-edge + 1 :] = 0
                        t_new = t_new / t_new.abs().max()
                        tT = t_new.abs() * t_new
                        comp_ind, _ = self._evaluate_ipt(tT, edge)
                        if len(comp_ind) > 0:
                            c_tx = eSig[:, comp_ind + edge].sum(dim=1)
                            c_tx = c_tx / torch.sqrt((c_tx**2).sum())
                            a = c_tx @ self._inv_corr_sig
                            t_new = (a @ eSig if ops is None else ops.project(a)).squeeze()

                    tT = t_new.abs() * t_new
                    if -t_new.min() > t_new.max():
                        tT[t_new > 0] = 0
                    else:
                        tT[t_new < 0] = 0
                    tT = tT.abs()

                    spikes, sil = _kmeans_1d_spike_extraction(tT[edge:-edge], self.fsamp)
                    spikes = spikes + edge

                    # Require periodic structure: at least 3 spikes to reject random noise peaks
                    if sil > self.sil_threshold and len(spikes) >= 3:
                        # Remove extracted peaks from activity
                        for sp in spikes:
                            lo_s, hi_s = _find_hill_limits(act_resid.cpu().numpy(), sp)
                            act_resid[lo_s : hi_s + 1] = 0

                        noise_msk = torch.ones(len(tT), dtype=torch.bool, device=tT.device)
                        noise_msk[spikes] = False
                        noise_tT = tT[noise_msk]
                        pnr_val = (
                            (tT[spikes] ** 2).mean().item()
                            / max((noise_tT**2).mean().item(), 1e-10)
                            if len(noise_tT) > 0
                            else 1e-10
                        )
                        pnr = 10 * np.log10(max(pnr_val, 1e-10))

                        self.mu_pulses.append(spikes)
                        self.pnrs.append(pnr)
                        self.sils.append(sil)
                        self.ipts.append(tT.detach().cpu().numpy())
                        sep_vecs.append(c_tx.clone())
                        centroids_list.append(
                            [
                                tT[spikes].mean().item(),
                                tT[list(set(range(len(tT))) - set(spikes))].mean().item(),
                            ]
                        )

                    weights = torch.zeros_like(weights)
                    weights[spikes] = self._cost_weights(t_new)[spikes]

                # Remove prohibited regions from gradient
                for p in prohibited:
                    weights[max(0, p - 5) : min(N, p + 5)] = 0

                # Compute gradient (GEMV on GPU)
                valid = torch.arange(edge, N - edge, device=eSig.device)
                if ops is None:
                    dc = eSig[:, valid] @ weights[valid]
                else:
                    masked_weights = torch.zeros_like(weights)
                    masked_weights[valid] = weights[valid]
                    dc = ops.weighted_sum(masked_weights)
                dc = dc / torch.sqrt((dc**2).sum())

                # Cost function and step direction
                old_j = (t_new**3).mean() / 3 if self.cost_fcn == 3 else t_new.abs().mean()
                mode = 1.0 if old_j >= 0 else -1.0

                c_tx = c_tx + mode * step_size * s_alpha * dc
                # Normalize separation vector, then RECOMPUTE IPT from updated c_tx.
                # MATLAB: c_tx = c_tx/clen; t_new = c_tx' * invCorrSig * eSig
                # (NOT t_new += dt — accumulating gradients drifts from true IPT)
                clen = torch.sqrt((c_tx**2).sum())
                c_tx = c_tx / clen
                a = c_tx @ self._inv_corr_sig
                t_new = (a @ eSig if ops is None else ops.project(a)).squeeze()
                t_new[:edge] = 0
                t_new[-edge:] = 0

                # Convergence check
                similarity = (t_new - t_old).abs().sum() / t_new.abs().sum()
                if similarity < 0.01:
                    similarity_flag += 1
                    if similarity_flag >= 2:
                        break
                else:
                    similarity_flag = 0

            # Final spike extraction after gradient descent
            tT = t_new.abs() * t_new
            if -t_new.min() > t_new.max():
                tT[t_new > 0] = 0
            else:
                tT[t_new < 0] = 0
            tT = tT.abs()

            spikes, sil = _kmeans_1d_spike_extraction(tT[edge:-edge], self.fsamp)
            spikes = spikes + edge

            # Remove negative peak region
            n02 = (-t_new).argmax().item()
            lo_n, hi_n = _find_hill_limits(act_resid.cpu().numpy(), n02)
            prohibited.extend(range(lo_n, hi_n + 1))

            if sil > self.sil_threshold and len(spikes) >= 3:
                for sp in spikes:
                    lo_s, hi_s = _find_hill_limits(act_resid.cpu().numpy(), sp)
                    act_resid[lo_s : hi_s + 1] = 0

                noise_mask = torch.ones(len(tT), dtype=torch.bool, device=tT.device)
                noise_mask[spikes] = False
                noise_tT = tT[noise_mask]
                pnr_val = (
                    (tT[spikes] ** 2).mean().item() / max((noise_tT**2).mean().item(), 1e-10)
                    if len(noise_tT) > 0
                    else 1e-10
                )
                pnr = 10 * np.log10(max(pnr_val, 1e-10))

                self.mu_pulses.append(spikes)
                self.pnrs.append(pnr)
                self.sils.append(sil)
                self.ipts.append(tT.detach().cpu().numpy())
                sep_vecs.append(c_tx.clone())
                centroids_list.append(
                    [
                        tT[spikes].mean().item(),
                        tT[list(set(range(len(tT))) - set(spikes))].mean().item(),
                    ]
                )

        print(
            f"    Decomposition: {time.time() - t_decomp:.1f}s, extracted {len(self.mu_pulses)} MUs"
        )

        # 4. Deduplicate via RoA
        if len(self.mu_pulses) > 0:
            sep_vecs, centroids_list = self._deduplicate(sep_vecs, centroids_list)

        self.separation_vectors = (
            torch.stack(sep_vecs, dim=1).cpu().numpy() if sep_vecs else np.array([])
        )
        self.centroids = np.array(centroids_list)

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_weights(self, t_new):
        """Compute cost function gradient weights."""
        if self.cost_fcn == 3:
            return t_new**2
        elif self.cost_fcn == 5:
            return t_new.abs()
        else:
            return torch.log(1 + t_new**2)

    def _evaluate_ipt(self, tT, edge):
        """Find optimal MU firing pattern via threshold sweep and regularity cost.

        Port of evaluateIPT.m.
        """
        tT_np = tT[edge:-edge].cpu().numpy()
        if len(tT_np) == 0:
            return np.array([], dtype=int), 0.0

        max_pulse = tT_np.max()
        pulse_dist = int(self.fsamp / 100)
        freq_min, freq_max = 2.0, 50.0
        step = 0.05
        outliers_tsh = 2.0

        best_spikes = []
        best_cost = float("inf")

        # Search k ∈ [0.9, 0.05]
        for k in np.arange(0.9, 0.0, -step):
            thresh = k * max_pulse
            tmp_ind = np.where(tT_np > thresh)[0]
            if len(tmp_ind) < 2:
                continue

            # Local peak deduplication
            tmp_ind = _rem_repeated_ind(tT_np, tmp_ind, pulse_dist)

            if len(tmp_ind) > 5:
                # 1. First pass: find outliers in IPIs
                tmp2 = np.diff(tmp_ind)
                # polyfit(1:N, tmp2, 0) is just the mean
                i_ipi_mean = tmp2.mean()

                # Cut outliers (missed firings)
                non_outliers = np.where(
                    np.abs(tmp2 - i_ipi_mean) / abs(i_ipi_mean + 1e-12) <= outliers_tsh
                )[0]
                if len(non_outliers) > 6:
                    # MATLAB: tmp2(tmpInd2(3:end-2))
                    tmp2_clean = tmp2[non_outliers[2:-2]]
                else:
                    tmp2_clean = tmp2[non_outliers]

                if len(tmp2_clean) > 2:
                    # 2. Second pass: regularity analysis via 2nd-order polyfit
                    # x = 1:len(tmp2_clean)
                    x = np.arange(1, len(tmp2_clean) + 1)
                    p = np.polyfit(x, tmp2_clean, 2)
                    i_ipi_smooth = np.polyval(p, x)

                    i_freq = self.fsamp / (i_ipi_smooth + 1e-12)
                    f_mean = np.median(i_freq)
                    f_min, f_max = i_freq.min(), i_freq.max()

                    # Ffir_res: median relative deviation
                    f_res = np.median(
                        np.abs(tmp2_clean - i_ipi_smooth) / np.abs(i_ipi_smooth + 1e-12)
                    )

                    # 3. Cost calculation
                    tmp_cost = 1.0 if len(tmp2_clean) < 6 else 0.0

                    # Frequency penalties
                    if f_mean < freq_min or f_mean > freq_max:
                        tmp_cost += 100.0
                    if f_min < freq_min or f_min > freq_max:
                        tmp_cost += 100.0
                    if f_max < freq_min or f_max > freq_max:
                        tmp_cost += 100.0

                    # MaxNoPulses and CostPerPulse
                    max_no_pulses = f_mean * (tmp_ind[-1] - tmp_ind[0]) / self.fsamp
                    cost_per_pulse = 1.0 / (max_no_pulses + 1e-12)

                    # MATLAB: (tmpCostFun + 4.0*Ffir_res + 1*abs(1 - length(tmpInd)*CostPerPulse)) / 5
                    final_cost = (
                        tmp_cost + 4.0 * f_res + 1.0 * abs(1.0 - len(tmp_ind) * cost_per_pulse)
                    ) / 5.0

                    if final_cost < best_cost:
                        best_cost = final_cost
                        best_spikes = tmp_ind

        return np.array(best_spikes, dtype=int), best_cost

    def _deduplicate(self, sep_vecs=None, centroids_list=None):
        """Remove duplicate MU spike trains via RoA analysis.

        Args:
            sep_vecs: Optional list of separation vectors to filter.
            centroids_list: Optional list of centroid pairs to filter.

        Returns:
            (filtered_sep_vecs, filtered_centroids) or (None, None).
        """
        if len(self.mu_pulses) <= 1:
            return sep_vecs, centroids_list

        # Sort by SIL descending
        order = np.argsort(self.sils)[::-1]
        pulses = [self.mu_pulses[i] for i in order]
        pnrs = [self.pnrs[i] for i in order]
        sils = [self.sils[i] for i in order]
        ips = [self.ipts[i] for i in order]

        d_ipi = int(
            self.fsamp / 100
        )  # 20 samples at 2048 Hz (MATLAB PulseDistance = round(fsamp/100))
        lim = int(self.fsamp * 0.1)

        kept_indices = list(range(len(pulses)))
        i = 0
        while i < len(kept_indices):
            idx_i = kept_indices[i]
            ref_n = len(pulses[idx_i])
            if ref_n == 0:
                kept_indices.pop(i)
                continue

            j = i + 1
            while j < len(kept_indices):
                idx_j = kept_indices[j]
                test_n = len(pulses[idx_j])

                # MATLAB remRepMUST ratio check: only compare if counts are similar
                if test_n / ref_n < 0.3 or test_n / ref_n > 3.3:
                    j += 1
                    continue

                r = _rate_of_agreement(pulses[idx_i], pulses[idx_j], lim, d_ipi)
                if r > 0.3:
                    # Keep the one with higher SIL
                    if sils[idx_j] > sils[idx_i]:
                        kept_indices.pop(i)
                        i -= 1
                        break
                    else:
                        kept_indices.pop(j)
                else:
                    j += 1
            i += 1

        self.mu_pulses = [pulses[k] for k in kept_indices]
        self.pnrs = [pnrs[k] for k in kept_indices]
        self.sils = [sils[k] for k in kept_indices]
        self.ipts = [ips[k] for k in kept_indices]
        if sep_vecs is not None:
            sep_vecs = [sep_vecs[k] for k in kept_indices]
        if centroids_list is not None:
            centroids_list = [centroids_list[k] for k in kept_indices]
        print(f"    After dedup: {len(self.mu_pulses)} MUs kept")
        return sep_vecs, centroids_list

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path):
        """Save CKC decomposition results.

        Args:
            path: Output file path (e.g., 'ckc_result.pt').
        """
        state = {
            "mu_pulses": self.mu_pulses,
            "pnrs": self.pnrs,
            "sils": self.sils,
            "ipts": self.ipts,
            "separation_vectors": self.separation_vectors,
            "centroids": self.centroids,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device="cuda"):
        """Load serialized CKCDecomposer.

        Args:
            path: Path to the .pt file.
            device: Target device ('cuda' or 'cpu').

        Returns:
            CKCDecomposer instance.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(device=device)
        obj.mu_pulses = state["mu_pulses"]
        obj.pnrs = state["pnrs"]
        obj.sils = state["sils"]
        obj.ipts = state.get("ipts", [])
        obj.separation_vectors = state["separation_vectors"]
        obj.centroids = state["centroids"]
        return obj

    def __repr__(self):
        return (
            f"CKCDecomposer(n_mu_max={self.n_mu_max}, "
            f"device='{self.device}', mu_count={self.mu_count})"
        )


# ===========================================================================
# SegmentDecomposer
# ===========================================================================


class SegmentDecomposer:
    """Segment-wise CKC decomposition with whole-data back-projection.

    Port of segmentDecomp.m: decomposes long EMG in segments,
    back-projects segment templates onto whole data, merges results.

    Usage:
        decomposer = SegmentDecomposer(seg_duration_s=10, fsamp=2048, device='cuda')
        decomposer.fit(emg_array)  # emg: [n_channels, n_samples]
        print(f"MUs: {decomposer.mu_count}")
        decomposer.save('seg_decomp_result.pt')
    """

    def __init__(self, seg_duration_s=10, **ckc_kwargs):
        self.seg_duration_s = seg_duration_s
        self.ckc_kwargs = ckc_kwargs
        # Resolve device to best GPU
        dev = ckc_kwargs.get("device", "cpu")
        if dev in ("cuda", "auto"):
            self.device = _get_best_cuda_device()
        else:
            self.device = dev
        self.mu_pulses = []
        self.separation_vectors = None  # kept for CKCDecomposer compat
        self.mu_templates = []  # List[np.ndarray] per-MU separation vectors
        self.pnrs = []
        self.sils = []
        self.ipts = []
        self.centroids = None

    @property
    def mu_count(self):
        return len(self.mu_pulses)

    def fit(self, emg, emg_full=None):
        """Segment-wise CKC decomposition.

        Args:
            emg: numpy [n_channels, n_samples] preprocessed EMG for template extraction.
            emg_full: optional numpy [n_channels, n_samples_total] for back-projection.
                     If None, back-projects onto emg.

        Returns:
            self
        """
        from .ckc_extender import CKCExtender

        fsamp = self.ckc_kwargs.get("fsamp", 2048)
        ext_factor = self.ckc_kwargs.get("ext_factor", 10)
        device = self.device
        seg_samples = int(self.seg_duration_s * fsamp)
        n_samples = emg.shape[1]
        n_segs = max(1, n_samples // seg_samples)

        print(f"SegmentDecomposer: {n_segs} segments x {self.seg_duration_s}s each")

        extender = CKCExtender(ext_factor=ext_factor, device=device)

        # Step 1: Per-segment CKC decomposition
        segment_results = []
        for i_seg in range(n_segs):
            start = i_seg * seg_samples
            end = min(start + seg_samples, n_samples)
            seg_emg = emg[:, start:end]

            print(f"  Segment {i_seg + 1}/{n_segs} [{start}:{end}] ...")
            eSig = extender.extend(seg_emg)

            decomposer_kwargs = dict(self.ckc_kwargs)
            decomposer = CKCDecomposer(**decomposer_kwargs)
            decomposer.fit(eSig)
            segment_results.append(decomposer)
            gc.collect()
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 2: Whole-data back-projection (blocked to avoid GPU OOM)
        t_bp_start = time.time()
        emg_proj = emg_full if emg_full is not None else emg
        # CKCExtender.extend() trims ext_factor samples from both ends:
        #   eSig shape = [n_ch*ext_factor, N - ext_factor - 1]
        # n_proj_len must match the extended signal length, not the raw EMG length.
        n_proj_len = emg_proj.shape[1] - ext_factor - 1
        assert n_proj_len > 0, (
            f"Projection length {n_proj_len} <= 0 for {emg_proj.shape[1]} samples"
        )

        print(
            f"  Back-projecting templates onto {'full' if emg_full is not None else 'relevant'} data ({emg_proj.shape[1]} samples, proj_len={n_proj_len})..."
        )
        dev = torch.device(device)
        n_ext_ch = emg_proj.shape[0] * ext_factor
        edge = ext_factor * 10
        # Max samples per block: keep extended size well below GPU budget
        block_samp = max(20480, int(0.5 * 1024**3 / (n_ext_ch * 4)))

        # --- Blocked CorrSig accumulation ---
        corr_full = torch.zeros(n_ext_ch, n_ext_ch, dtype=torch.float32, device=dev)
        total_samp = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(0, n_proj_len, block_samp):
                end_k = min(k + block_samp, n_proj_len)
                b_emg = emg_proj[:, max(0, k - ext_factor + 1) : min(emg_proj.shape[1], end_k)]
                if b_emg.shape[1] == 0:
                    continue
                b_eSig = extender.extend(b_emg)
                blk_len = min(b_eSig.shape[1], end_k - k)
                blk = b_eSig[:, :blk_len]
                corr_full += blk @ blk.T
                total_samp += blk_len
                del b_eSig, blk, b_emg
        corr_full = corr_full / max(total_samp, 1)
        inv_corr_full = torch.linalg.pinv(corr_full)

        # --- Template projection ---
        # Collect per-MU projection vectors. If emg_full is available,
        # pre-extend all data once on GPU and project all MUs in a single GEMM.
        all_pulses = []
        all_pnrs = []
        all_sils = []
        all_ips = []
        all_templates = []

        if emg_full is not None:
            # --- First pass: collect per-MU projection vectors from all segments ---
            ctx_matrices = []  # list of 1D numpy arrays [ext_ch]
            mu_seg_map = []  # (segment_idx, local_mu_idx)

            for i_seg, decomposer in enumerate(segment_results):
                if decomposer.separation_vectors is None or decomposer.separation_vectors.size == 0:
                    continue
                sep_vecs = decomposer.separation_vectors
                if isinstance(sep_vecs, np.ndarray):
                    sep_vecs = torch.from_numpy(sep_vecs).to(dev)
                # A = ctx^T @ invCorr: [n_mus_seg, ext_ch]
                A = sep_vecs.T @ inv_corr_full  # [n_mus, ext_ch]

                for mu in range(A.shape[0]):
                    ctx_matrices.append(A[mu, :].cpu().numpy())
                    # Collect centroid for this MU (for template persistence)
                    if not hasattr(self, "_centroids_per_mu"):
                        self._centroids_per_mu = []
                    if decomposer.centroids is not None and mu < len(decomposer.centroids):
                        self._centroids_per_mu.append(decomposer.centroids[mu])
                    else:
                        self._centroids_per_mu.append(np.array([0.0, 0.0]))
                    mu_seg_map.append((i_seg, mu))

            self._mu_seg_map = mu_seg_map  # save for template persistence
            n_mus_global = len(ctx_matrices)

            if n_mus_global > 0:
                # --- Force GPU cleanup before backprojection ---
                if "cuda" in device and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(
                    f"  Back-projection start: {emg_proj.shape[1]} samples, "
                    f"{n_mus_global} templates from {n_segs} segments"
                )

                # --- GPU pre-extension (memory-safe, chunked if needed) ---
                # ponytail: extend full EMG once on GPU, project all MUs in
                # single GEMM. If estimated GPU memory exceeds 70% free,
                # split into 2 chunks to avoid OOM.
                extender_full = CKCExtender(ext_factor=ext_factor, device=device)
                n_ch = emg_proj.shape[0]
                ext_ch = n_ch * ext_factor
                n_full = emg_proj.shape[1]

                eSig_est = ext_ch * n_full * 4  # float32 bytes
                ipt_est = n_mus_global * n_full * 4
                total_est = int((eSig_est + ipt_est) * 1.5)

                if "cuda" in device and torch.cuda.is_available():
                    free_mem, _ = torch.cuda.mem_get_info(device.index)
                    max_usable = int(free_mem * 0.7)
                else:
                    max_usable = 16 * 1024**3

                if total_est > max_usable and n_full > 0:
                    n_chunks = 2
                    chunk_len = n_full // n_chunks
                    overlap = ext_factor * 2
                    print(
                        f"  Backprojection: {total_est / 1024**3:.1f}GB est > "
                        f"{max_usable / 1024**3:.1f}GB usable, splitting into {n_chunks} chunks"
                    )
                else:
                    n_chunks = 1
                    chunk_len = n_full
                    overlap = 0

                # --- Build ctx_all once (shared across chunks) ---
                ctx_all = np.stack(ctx_matrices, axis=0)  # [n_mus, ext_ch]
                self._ctx_all = ctx_all

                # --- Chunked projection + spike extraction ---
                for i_chunk in range(n_chunks):
                    if n_chunks > 1:
                        chunk_start = max(0, i_chunk * chunk_len - overlap)
                        chunk_end = min(n_full, (i_chunk + 1) * chunk_len + overlap)
                    else:
                        chunk_start = 0
                        chunk_end = n_full

                    t_chunk = time.time()
                    if n_chunks > 1:
                        print(
                            f"  Chunk {i_chunk + 1}/{n_chunks}: samples [{chunk_start}:{chunk_end}] "
                            f"({(chunk_end - chunk_start) / fsamp:.1f}s)"
                        )
                    emg_chunk = emg_proj[:, chunk_start:chunk_end]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    eSig_chunk = extender_full.extend(emg_chunk)
                    t_ext = time.time()
                    print(
                        f"    Extended: {eSig_chunk.shape[0]}ch x {eSig_chunk.shape[1]}samp "
                        f"({eSig_chunk.nbytes / 1024**3:.1f}GB, {t_ext - t_chunk:.1f}s)"
                    )

                    ctx_all_t = torch.from_numpy(ctx_all).to(eSig_chunk.device)
                    test_ipt_chunk = ctx_all_t @ eSig_chunk
                    t_gemm = time.time()
                    print(
                        f"    GEMM: {n_mus_global} MUs x {eSig_chunk.shape[1]}samp "
                        f"({t_gemm - t_ext:.1f}s)"
                    )

                    for global_mu, (i_seg, mu) in enumerate(mu_seg_map):
                        decomposer = segment_results[i_seg]
                        t_new = test_ipt_chunk[global_mu, :]
                        tT = t_new**2
                        if -t_new.min() > t_new.max():
                            tT[t_new > 0] = 0
                        else:
                            tT[t_new < 0] = 0

                        if decomposer.centroids is not None and mu < len(decomposer.centroids):
                            thresh = np.mean(decomposer.centroids[mu])
                        else:
                            thresh = tT[edge:-edge].mean().item()

                        tT_np = tT.cpu().numpy()
                        comp_ind = np.where(tT_np[edge:-edge] > thresh)[0] + edge

                        if len(comp_ind) == 0:
                            continue

                        min_dist = int(fsamp / 100)
                        cleaned = _rem_repeated_ind(tT_np, comp_ind, min_dist)

                        if len(cleaned) < 5:
                            continue

                        # PNR (local, on chunk data)
                        noise_mask = np.ones(len(tT_np), dtype=bool)
                        noise_mask[cleaned] = False
                        pnr_val = np.mean(tT_np[cleaned] ** 2) / max(
                            np.mean(tT_np[noise_mask] ** 2), 1e-10
                        )
                        pnr_db = 10 * np.log10(max(pnr_val, 1e-10))

                        # Convert to global indices
                        cleaned = cleaned + chunk_start

                        # Strip overlap regions for multi-chunk
                        if n_chunks > 1:
                            valid_start = i_chunk * chunk_len
                            valid_end = min(n_full, (i_chunk + 1) * chunk_len)
                            keep = (cleaned >= valid_start) & (cleaned < valid_end)
                            cleaned = cleaned[keep]

                        if len(cleaned) < 5:
                            continue

                        # SIL pre-filter: skip low-quality spike trains
                        sil = decomposer.sils[mu] if mu < len(decomposer.sils) else 0.0
                        min_sil = self.ckc_kwargs.get("sil_threshold", 0.8)
                        if sil >= min_sil:
                            all_pulses.append(cleaned)
                            all_pnrs.append(pnr_db)
                            all_sils.append(sil)
                            all_ips.append(tT_np)
                            # Store per-MU separation vector (numpy, CPU)
                            all_templates.append(decomposer.separation_vectors[:, mu])

                    # Per-MU progress (every 5 MUs)
                    if global_mu % 5 == 0 or global_mu == n_mus_global - 1:
                        t_mu = time.time()
                        n_done = global_mu + 1
                        print(f"    MU {n_done}/{n_mus_global} ({t_mu - t_gemm:.1f}s elapsed)")

                    del eSig_chunk, test_ipt_chunk, ctx_all_t
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # Original block-by-block projection (emg_full not provided)
            for i_seg, decomposer in enumerate(segment_results):
                if decomposer.separation_vectors is None or decomposer.separation_vectors.size == 0:
                    continue
                ctx = decomposer.separation_vectors
                if isinstance(ctx, np.ndarray):
                    ctx = torch.from_numpy(ctx).to(dev)
                # A = ctx^T @ invCorr: [n_mus_seg, M]
                A = ctx.T @ inv_corr_full  # [n_mus, M]

                # Blocked projection: test_ipt = A @ eSig (in blocks)
                n_mus_seg = A.shape[0]
                test_ipt = torch.zeros(n_mus_seg, n_proj_len, dtype=torch.float32, device=dev)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for k in range(0, n_proj_len, block_samp):
                        end_k = min(k + block_samp, n_proj_len)
                        b_emg = emg_proj[
                            :, max(0, k - ext_factor + 1) : min(emg_proj.shape[1], end_k)
                        ]
                        if b_emg.shape[1] == 0:
                            continue
                        b_eSig = extender.extend(b_emg)
                        blk_len = min(b_eSig.shape[1], end_k - k)
                        blk = b_eSig[:, :blk_len]
                        test_ipt[:, k : k + blk_len] = A @ blk
                        del b_eSig, blk, b_emg

                for mu in range(test_ipt.shape[0]):
                    t_new = test_ipt[mu, :]
                    tT = t_new**2
                    if -t_new.min() > t_new.max():
                        tT[t_new > 0] = 0
                    else:
                        tT[t_new < 0] = 0

                    # Threshold: strictly above centroid mean (MATLAB aligned)
                    if decomposer.centroids is not None and mu < len(decomposer.centroids):
                        thresh = np.mean(decomposer.centroids[mu])
                    else:
                        # Fallback if centroids not available
                        thresh = tT[edge:-edge].mean().item()

                    tT_np = tT.cpu().numpy()
                    comp_ind = np.where(tT_np[edge:-edge] > thresh)[0] + edge

                    if len(comp_ind) == 0:
                        continue

                    # MATLAB-aligned dedup: find local maxima in clusters
                    min_dist = int(fsamp / 100)
                    cleaned = _rem_repeated_ind(tT_np, comp_ind, min_dist)

                    if len(cleaned) < 5:  # require minimal periodic structure
                        continue

                    # MATLAB pnr.m: 10*log10(mean(spikes^2) / mean(noise^2))
                    # where input is already squared IPT (sj^2).
                    # So we calculate mean of sj^4.
                    noise_mask = np.ones(len(tT_np), dtype=bool)
                    noise_mask[cleaned] = False
                    pnr_val = np.mean(tT_np[cleaned] ** 2) / max(
                        np.mean(tT_np[noise_mask] ** 2), 1e-10
                    )
                    pnr_db = 10 * np.log10(max(pnr_val, 1e-10))

                    # SIL pre-filter: skip low-quality spike trains
                    sil = decomposer.sils[mu] if mu < len(decomposer.sils) else 0.0
                    min_sil = self.ckc_kwargs.get("sil_threshold", 0.8)
                    if sil >= min_sil:
                        all_pulses.append(cleaned)
                        all_pnrs.append(pnr_db)
                        all_sils.append(sil)
                        all_ips.append(tT_np)
                        # Store per-MU separation vector (numpy, CPU)
                        all_templates.append(ctx[:, mu].cpu().numpy())

        print(
            f"  Back-projection done: {time.time() - t_bp_start:.1f}s total, "
            f"{len(all_pulses)} MUs before dedup"
        )

        # Step 3: Global deduplication
        if all_pulses:
            self._global_dedup(all_pulses, all_pnrs, all_sils, all_ips, all_templates, fsamp)

        # --- Phase 5.1: Segment boundary integrity assertions ---
        for mu_idx, pulses in enumerate(self.mu_pulses):
            if len(pulses) > 0:
                p_min, p_max = pulses.min(), pulses.max()
                assert p_min >= 0, (
                    f"MU {mu_idx}: negative pulse index {p_min} (proj_len={n_proj_len})"
                )
                assert p_max < n_proj_len, f"MU {mu_idx}: pulse {p_max} >= proj_len {n_proj_len}"
        if self.mu_pulses:
            print(
                f"  Segment boundary check: {sum(len(p) for p in self.mu_pulses)} "
                f"pulses across {len(self.mu_pulses)} MUs, all within [0, {n_proj_len})"
            )

        return self

    def _global_dedup(self, pulses, pnrs, sils, ips, templates, fsamp):
        """Deduplicate across all segments via RoA."""
        if len(pulses) <= 1:
            self.mu_pulses = pulses
            self.pnrs = pnrs
            self.sils = sils
            self.ipts = ips
            self.mu_templates = templates
            return

        lim = int(fsamp * 0.1)
        d_ipi = int(fsamp / 100)  # 20 samples at 2048 Hz

        order = np.argsort(pnrs)[::-1]  # Sort by PNR
        n_total = len(pnrs)
        print(
            f"  Global dedup: sorting {n_total} MUs by PNR, "
            f"top PNR={pnrs[order[0]]:.1f}dB, "
            f"median PNR={np.median(pnrs):.1f}dB"
        )
        kept = list(range(len(pulses)))

        i = 0
        n_processed = 0
        while i < len(kept):
            n_processed += 1
            ref_n = len(pulses[kept[i]])
            if ref_n == 0:
                kept.pop(i)
                continue

            j = i + 1
            while j < len(kept):
                test_n = len(pulses[kept[j]])

                # Ratio check: only compare if firing counts are within similar scale
                if test_n / ref_n < 0.3 or test_n / ref_n > 3.3:
                    j += 1
                    continue

                r = _rate_of_agreement(pulses[kept[i]], pulses[kept[j]], lim, d_ipi)
                if r > 0.3:
                    if pnrs[kept[j]] > pnrs[kept[i]]:
                        kept.pop(i)
                        i -= 1
                        break
                    else:
                        kept.pop(j)
                else:
                    j += 1
            i += 1
            # Progress print every 20 MUs
            if n_processed % 20 == 0:
                print(f"    Dedup: {n_processed}/{n_total} MUs processed, {len(kept)} kept so far")

        print(f"    Dedup: {n_processed}/{n_total} MUs done, {len(kept)} kept final")

        self.mu_pulses = [pulses[k] for k in kept]
        self.pnrs = [pnrs[k] for k in kept]
        self.sils = [sils[k] for k in kept]
        self.ipts = [ips[k] for k in kept]
        self.mu_templates = [templates[k] for k in kept]
        print(f"  Global dedup: {len(self.mu_pulses)} MUs kept from {len(pulses)}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path):
        """Save SegmentDecomposer results.

        Args:
            path: Output file path (e.g., 'seg_decomp.pt').
        """
        state = {
            "mu_pulses": self.mu_pulses,
            "pnrs": self.pnrs,
            "sils": self.sils,
            "ipts": self.ipts,
            "mu_templates": self.mu_templates,
            "seg_duration_s": self.seg_duration_s,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, **ckc_kwargs):
        """Load serialized SegmentDecomposer.

        Args:
            path: Path to the .pt file.
            **ckc_kwargs: CKCDecomposer kwargs (for reconstruction).

        Returns:
            SegmentDecomposer instance.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(seg_duration_s=state["seg_duration_s"], **ckc_kwargs)
        obj.mu_pulses = state["mu_pulses"]
        obj.pnrs = state["pnrs"]
        obj.sils = state["sils"]
        obj.ipts = state.get("ipts", [])
        obj.mu_templates = state.get("mu_templates", [])
        return obj

    def save_templates(self, path):
        """Save projection matrix + centroids for cross-domain template transfer.

        Args:
            path: .npz file path.
        """
        import os

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(
            path,
            projection_matrix=self._ctx_all if hasattr(self, "_ctx_all") else np.array([]),
            centroids=np.array(self._centroids_per_mu)
            if hasattr(self, "_centroids_per_mu")
            else np.array([]),
            mu_seg_map=np.array(self._mu_seg_map) if hasattr(self, "_mu_seg_map") else np.array([]),
            n_mus=len(self.mu_pulses),
            ext_factor=self.ckc_kwargs.get("ext_factor", 10),
            fsamp=self.ckc_kwargs.get("fsamp", 2048),
            allow_pickle=True,
        )
        print(f"  Templates saved: {path} ({len(self.mu_pulses)} MUs)")

    @staticmethod
    def load_templates(path):
        """Load saved templates.

        Returns dict: projection_matrix, centroids, n_mus, ext_factor, fsamp
        """
        data = np.load(path, allow_pickle=True)
        return {
            "projection_matrix": data["projection_matrix"],
            "centroids": data["centroids"],
            "n_mus": int(data["n_mus"]),
            "ext_factor": int(data["ext_factor"]),
            "fsamp": int(data["fsamp"]),
        }

    def __repr__(self):
        return f"SegmentDecomposer(seg={self.seg_duration_s}s, mu_count={self.mu_count})"


def apply_templates(
    projection_matrix, centroids, emg_new, fsamp=2048, ext_factor=10, device="cuda",
    verbose=True, polarities=None,
):
    """Apply pre-learned templates to new EMG, extracting spike trains.

    Core: IPT = projection_matrix @ extend(emg_new)
          -> threshold by centroids -> spike extraction.

    Args:
        projection_matrix: np.ndarray [n_mus, ext_ch]
        centroids: np.ndarray [n_mus, 2] or None for fallback
        emg_new: np.ndarray [n_ch, n_samples] preprocessed EMG
        fsamp, ext_factor, device: self-explanatory
        verbose: print progress

    Returns:
        dict: mu_pulses, sils, pnrs, n_mus
    """
    n_mus = projection_matrix.shape[0]
    if n_mus == 0:
        return {"mu_pulses": [], "sils": [], "pnrs": [], "n_mus": 0}

    if polarities is not None:
        polarities = np.asarray(polarities)
        if polarities.shape != (n_mus,) or not np.isin(polarities, [-1, 1]).all():
            raise ValueError("Expected one fixed training polarity per MU")
        if centroids is None or np.asarray(centroids).shape != (n_mus, 2):
            raise ValueError("Fixed templates require two training centroids per MU")
        if not np.isfinite(centroids).all() or not np.all(np.mean(centroids, axis=1) > 0):
            raise ValueError("Fixed template thresholds must be positive and finite")

    from .ckc_extender import CKCExtender

    # Extend new EMG on GPU
    extender = CKCExtender(ext_factor=ext_factor, device=device)
    eSig_new = extender.extend(emg_new)
    N_new = eSig_new.shape[1]
    edge = ext_factor

    # Project all MUs
    ctx_t = torch.from_numpy(projection_matrix.astype(np.float32)).to(eSig_new.device)
    ipt_all = ctx_t @ eSig_new  # [n_mus, N_new]

    if verbose:
        print(f"  apply_templates: {n_mus} MUs -> {N_new} samples ({N_new / fsamp:.1f}s)")

    mu_pulses, sils, pnrs = [], [], []
    for mu in range(n_mus):
        t_new = ipt_all[mu, :].clone()
        t_new[:edge] = 0
        t_new[-edge:] = 0

        tT = t_new.abs() * t_new
        if polarities is not None:
            tT[t_new * int(polarities[mu]) < 0] = 0
        elif -t_new.min() > t_new.max():
            tT[t_new > 0] = 0
        else:
            tT[t_new < 0] = 0
        tT = tT.abs()

        tT_np = tT[edge:-edge].cpu().numpy()

        # Threshold from centroids or fallback
        if centroids is not None and len(centroids) > mu and len(centroids[mu]) >= 1:
            thresh = float(np.mean(centroids[mu]))
            if thresh <= 0:
                thresh = float(np.mean(tT_np)) * 1.5
        else:
            thresh = float(np.mean(tT_np)) * 1.5

        comp_ind = np.where(tT_np > thresh)[0] + edge
        if len(comp_ind) < 2:
            mu_pulses.append(np.array([], dtype=int))
            sils.append(0.0)
            pnrs.append(-10.0)
            continue

        limit = int(fsamp / 100)
        cleaned = _rem_repeated_ind(tT.cpu().numpy(), comp_ind, limit)
        if len(cleaned) < 3:
            mu_pulses.append(np.array([], dtype=int))
            sils.append(0.0)
            pnrs.append(-10.0)
            continue

        # SIL
        sv = tT_np[cleaned - edge]
        d_spike = np.abs(tT_np - sv.mean())
        d_noise = np.abs(tT_np - np.median(tT_np))
        sil = float(np.mean((d_noise - d_spike) / np.maximum(np.maximum(d_spike, d_noise), 1e-10)))

        # PNR
        noise_msk = np.ones(len(tT_np), dtype=bool)
        noise_msk[cleaned - edge] = False
        pnr_val = (
            np.mean(tT_np[cleaned - edge] ** 2) / max(np.mean(tT_np[noise_msk] ** 2), 1e-10)
            if noise_msk.sum() > 0
            else 1e-10
        )
        pnr = 10 * np.log10(max(pnr_val, 1e-10))

        mu_pulses.append(cleaned.astype(int))
        sils.append(sil)
        pnrs.append(pnr)

    del eSig_new, ipt_all, ctx_t
    if "cuda" in str(device):
        torch.cuda.empty_cache()

    return {"mu_pulses": mu_pulses, "sils": sils, "pnrs": pnrs, "n_mus": n_mus}
