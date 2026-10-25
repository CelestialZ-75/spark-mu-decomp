# src/models/v3/ckc_preprocess.py
"""CKC-specific EMG preprocessing for motor unit decomposition.

Port of PreProcess4GUI_v2.m preprocessing pipeline to Python.
Prepares raw aligned EMG for CKC blind source separation by applying:
    1. Channel quality filtering (exclude bad channels)
    2. Bandpass filtering (20-500 Hz, causal — preserves spike timing)
    3. Powerline notch filtering (50 Hz comb, causal)
    4. Output (no time_diff, no spatial bipolar — removed per advisor request)

Output: [n_channels, n_samples] float32 numpy array.
       Same channel count as input (256ch or filtered subset).

Usage:
    from mu_course.ckc.ckc_preprocess import preprocess_for_ckc, load_preprocessed

    # From .mat file
    data = preprocess_for_ckc(mat_path, cache_path='preprocessed.npy')

    # Load cached
    data = load_preprocessed('preprocessed.npy')
"""

import os

import numpy as np
from scipy import signal
from scipy.io import loadmat

# ===========================================================================
# Electrode layout — imported from visualization module
# ===========================================================================


def _load_elearr():
    """Load electrode arrangement matrix. Falls back to hardcoded copy."""
    # Use the original embedded layout without the research repository.
    _elearr = np.array(
        [
            [8, 16, 24, 32, 40, 48, 56, 64],
            [7, 15, 23, 31, 39, 47, 55, 63],
            [6, 14, 22, 30, 38, 46, 54, 62],
            [5, 13, 21, 29, 37, 45, 53, 61],
            [4, 12, 20, 28, 36, 44, 52, 60],
            [3, 11, 19, 27, 35, 43, 51, 59],
            [2, 10, 18, 26, 34, 42, 50, 58],
            [1, 9, 17, 25, 33, 41, 49, 57],
        ],
        dtype=int,
    )
    return _elearr, np.array([0, 64, 128, 192]), 4, 64


# ===========================================================================
# Signal processing steps
# ===========================================================================


def _apply_bandpass(data, fs=2048, lowcut=20, highcut=500, order=4):
    """Apply causal Butterworth bandpass filter per channel.

    Args:
        data: [n_channels, n_samples] float64
        fs: Sampling rate (Hz)
        lowcut, highcut: Corner frequencies (Hz)
        order: Filter order

    Returns:
        [n_channels, n_samples] float64, bandpass-filtered
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)  # prevent > nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    # Apply causal filter along time axis (axis=1)
    filtered = signal.lfilter(b, a, data, axis=1)
    return filtered


def _apply_notch_comb(data, fs=2048, base_freq=50, q=10):
    """Apply notch filters at powerline harmonics per channel.

    Filters out the fundamental frequency and all its harmonics up to Nyquist.
    Replaces scipy.signal.iircomb to avoid strict fs/f0 divisibility requirements.

    Args:
        data: [n_channels, n_samples] float64
        fs: Sampling rate (Hz)
        base_freq: Fundamental powerline frequency (Hz)
        q: Quality factor (bandwidth-related)

    Returns:
        [n_channels, n_samples] float64, notch-filtered
    """
    nyq = 0.5 * fs
    # Generate all harmonics up to Nyquist
    harmonics = np.arange(base_freq, nyq, base_freq)

    filtered_data = data.copy()
    for f0 in harmonics:
        w0 = f0 / nyq
        b, a = signal.iirnotch(w0, q)
        # Apply causal filter along time axis (axis=1)
        filtered_data = signal.lfilter(b, a, filtered_data, axis=1)

    return filtered_data


def _apply_time_diff(data):
    """Apply first-order temporal difference per channel.

    This emphasizes the rapid edges of MUAPs and suppresses slow
    baseline drift. Equivalent to MATLAB: diff(Data')'

    Args:
        data: [n_channels, n_samples] float64

    Returns:
        [n_channels, n_samples-1] float64
    """
    return np.diff(data, axis=1)


def _arrange_to_grid(data_2d, elearr, adapter_offsets, n_adapters=4, channels_per_adapter=64):
    """Arrange channels into 4 x (8x8) electrode grids.

    Each adapter's 64 channels are mapped to an 8x8 grid using the
    snake-column layout. Missing/bad channels become None.

    Args:
        data_2d: [n_channels, n_samples] (expected n_channels <= 256)
        elearr: [8, 8] local channel indices (1-indexed)
        adapter_offsets: [4] start index per adapter (0-indexed)
        n_adapters: Number of adapters (4)
        channels_per_adapter: Channels per adapter (64)

    Returns:
        List of 4 arrays, each [8, 8] cell array where each cell is
        a 1D signal or None (bad channel).
    """
    grids = []

    for a in range(n_adapters):
        offset = adapter_offsets[a]
        grid = [[None] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                local_ch = elearr[r, c]  # 1-indexed, 1..64
                global_idx = offset + local_ch - 1  # 0-indexed
                if global_idx < data_2d.shape[0]:
                    grid[r][c] = data_2d[global_idx, :]
        grids.append(grid)

    return grids


def _apply_spatial_bipolar(grid):
    """Apply longitudinal (proximal-distal) spatial bipolar filter.

    For each column in the 8x8 grid, computes:
        Y[r, c] = grid[r+1, c] - grid[r, c]

    This removes far-field common-mode signal and isolates local MUAPs.
    Output has 7 rows (one fewer than input).

    Args:
        grid: [8, 8] cell array of 1D signals or None

    Returns:
        [7, 8] cell array of 1D signals or None
    """
    n_rows, n_cols = len(grid), len(grid[0])
    bp_grid = [[None] * n_cols for _ in range(n_rows - 1)]

    for r in range(n_rows - 1):
        for c in range(n_cols):
            upper = grid[r][c]
            lower = grid[r + 1][c]
            if upper is not None and lower is not None:
                bp_grid[r][c] = lower - upper

    return bp_grid


def _flatten_grid(bp_grid):
    """Flatten bipolar grid to 2D array [n_channels, n_samples].

    Column-major traversal: for each column, iterate rows top-to-bottom.
    Matches MATLAB PreProcess4GUI_v2.m flatten order.

    Args:
        bp_grid: [7, 8] cell array of 1D signals or None

    Returns:
        np.ndarray [n_valid_channels, n_samples] float32
    """
    channels = []
    n_cols = len(bp_grid[0])
    n_rows = len(bp_grid)

    for c in range(n_cols):
        for r in range(n_rows):
            if bp_grid[r][c] is not None:
                channels.append(bp_grid[r][c].astype(np.float32))

    if not channels:
        return np.empty((0, 0), dtype=np.float32)

    return np.stack(channels, axis=0)


# ===========================================================================
# Main entry point
# ===========================================================================


def preprocess_for_ckc(
    mat_path,
    fs=2048,
    cache_path=None,
    bandpass=(20, 500),
    notch_base=50,
    notch_q=30,
    good_channels_only=True,
    verbose=True,
    target_adapters=None,
    time_slices=None,
):
    """Preprocess raw EMG from .mat file for CKC decomposition.

    Simplified preprocessing pipeline (per advisor request).
    Steps: bandpass -> notch -> output.

    Args:
        mat_path: Path to the .mat file (aligned_{Finger}.mat or similar).
        fs: Sampling rate (Hz). Default 2048.
        cache_path: If provided, save preprocessed data as .npy and skip
                    recomputation if file exists.
        bandpass: (lowcut, highcut) Hz. Default (20, 500).
        notch_base: Fundamental powerline frequency (Hz). Default 50.
        notch_q: Notch filter quality factor. Default 30.
        good_channels_only: If True, exclude channels marked bad in the
                            .mat file's good_channels field.
        verbose: Print progress.
        target_adapters: List of adapter indices to include (0-3).
                         None = all 4 adapters. E.g. [0,1] for flexor side.
        time_slices: List of (start_s, end_s) tuples for temporal windowing.
                     None = full signal. E.g. [(3,9), (13,19)] for plateaus.

    Returns:
        np.ndarray [n_channels, n_samples] float32, preprocessed for CKC.
        Same channel count as input (256ch or filtered subset).

    Raises:
        FileNotFoundError: If mat_path does not exist.
        KeyError: If emg_aligned field not found in .mat.
    """
    # --- Cache check ---
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"  Loading preprocessed data from cache: {cache_path}")
        return np.load(cache_path)

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f".mat file not found: {mat_path}")

    # --- Load raw EMG ---
    if verbose:
        print(f"  Loading: {os.path.basename(mat_path)}")
    mat_data = _load_mat(mat_path)

    if "emg_aligned" not in mat_data:
        raise KeyError(
            f"'emg_aligned' not found in {mat_path}. "
            f"Available fields: {list(mat_data.keys())}. "
            f"Run Stage 1 preprocessing first."
        )
    emg = mat_data["emg_aligned"].astype(np.float64)  # [256, N]
    n_ch_orig = emg.shape[0]  # save before adapter filtering

    # --- Adapter filtering (spatial prior) ---
    ch_per_adapter = 64
    if target_adapters is not None:
        keep_mask = np.zeros(emg.shape[0], dtype=bool)
        for a in target_adapters:
            start = a * ch_per_adapter
            end = start + ch_per_adapter
            keep_mask[start : min(end, emg.shape[0])] = True
        emg = emg[keep_mask, :]
        if verbose:
            print(f"  Adapters: {target_adapters} -> {emg.shape[0]} channels")

    # --- Temporal slicing (plateau prior) ---
    if time_slices is not None:
        slices_samp = []
        for t_start, t_end in time_slices:
            s0 = int(t_start * fs)
            s1 = int(t_end * fs)
            s1 = min(s1, emg.shape[1])
            if s1 > s0:
                slices_samp.append(emg[:, s0:s1])
        if slices_samp:
            emg = np.concatenate(slices_samp, axis=1)
            if verbose:
                print(
                    f"  Time slices: {len(time_slices)} windows -> "
                    f"{emg.shape[1]} samples ({emg.shape[1] / fs:.1f}s)"
                )
        else:
            return np.empty((0, 0), dtype=np.float32)

    # --- Channel quality filtering ---
    if good_channels_only and "good_channels" in mat_data:
        gc = mat_data["good_channels"]
        # Handle various storage formats
        if gc.ndim == 2:
            gc = gc.ravel()
        gc = gc.astype(bool)

        # Handle size mismatch with ORIGINAL channel count first
        n_gc = len(gc)
        if n_gc != n_ch_orig:
            if n_gc > n_ch_orig:
                gc = gc[:n_ch_orig]
            else:
                padded = np.ones(n_ch_orig, dtype=bool)
                padded[:n_gc] = gc
                gc = padded
            if verbose:
                print(f"  good_channels size mismatch ({n_gc} vs {n_ch_orig}), adjusted")

        # Apply adapter filter to good_channels (after size normalization)
        if target_adapters is not None:
            gc = gc[keep_mask]

        n_before = emg.shape[0]
        n_gc_now = len(gc)
        if n_gc_now != n_before:
            # Final defensive pad after adapter filtering
            if n_gc_now > n_before:
                gc = gc[:n_before]
            else:
                padded = np.ones(n_before, dtype=bool)
                padded[:n_gc_now] = gc
                gc = padded
        emg = emg[gc, :]
        if verbose:
            print(f"  Channels: {n_before} -> {emg.shape[0]} (good_channels filter)")

    n_ch, n_samples = emg.shape
    if verbose:
        dur_s = n_samples / fs
        print(f"  Signal: {n_ch} ch x {n_samples} samples ({dur_s:.1f}s)")

    # --- Step 1: Bandpass ---
    if verbose:
        print(f"  Bandpass {bandpass[0]}-{bandpass[1]} Hz ...")
    emg = _apply_bandpass(emg, fs, bandpass[0], bandpass[1])

    # --- Step 2: Notch comb ---
    if verbose:
        print(f"  Notch {notch_base} Hz comb ...")
    emg = _apply_notch_comb(emg, fs, notch_base, notch_q)

    # --- Step 3: Output (no time_diff, no spatial bipolar) ---
    # ponytail: removed time_diff and spatial bipolar per advisor request.
    # bandpass+notch covers frequency domain; CKC invCorrSig covers spatial whitening.
    # Keeping 256ch (vs 224ch after bipolar) gives CKC more spatial info.
    result = emg.astype(np.float32)

    if verbose:
        print(
            f"  Output: {result.shape[0]} ch x {result.shape[1]} samples "
            f"({result.nbytes / 1024**2:.0f} MB)"
        )

    # --- Save cache ---
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, result)
        if verbose:
            print(f"  Saved to: {cache_path}")

    return result


def load_preprocessed(cache_path):
    """Load previously saved preprocessed data.

    Args:
        cache_path: Path to .npy file.

    Returns:
        np.ndarray [n_channels, n_samples] float32
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Preprocessed file not found: {cache_path}")
    return np.load(cache_path)


# ===========================================================================
# .mat file loader (compatible with v5/v7/v7.3)
# ===========================================================================


def _load_mat(mat_path):
    """Load .mat file, supporting v7.3 (HDF5) and v5/v7 formats."""
    data = _try_loadmat(mat_path)
    # Strip scipy metadata
    for meta_key in ("__header__", "__version__", "__globals__"):
        data.pop(meta_key, None)
    return data


def _try_loadmat(mat_path):
    """Try scipy.io.loadmat, fall back to h5py for v7.3."""
    try:
        return loadmat(mat_path)
    except NotImplementedError as e:
        if "v7.3" in str(e):
            import h5py

            data = {}
            with h5py.File(mat_path, "r") as f:
                for key in f:
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        arr = item[()]
                        # Transpose back from column-major
                        if isinstance(arr, np.ndarray) and arr.ndim == 2:
                            arr = arr.T
                        data[key] = arr
            return data
        raise


# ===========================================================================
# Convenience: preprocess from raw numpy array (skip .mat I/O)
# ===========================================================================


def preprocess_array(
    emg_aligned,
    fs=2048,
    bandpass=(20, 500),
    notch_base=50,
    notch_q=30,
    verbose=False,
    target_adapters=None,
):
    """Preprocess raw EMG array for CKC (no .mat I/O).

    Args:
        emg_aligned: [n_channels, n_samples] float64 raw EMG.
        fs: Sampling rate (Hz).
        target_adapters: List of adapter indices to include (0-3). None = all.

    Returns:
        [n_channels_out, n_samples] float32 preprocessed data.
    """
    emg = emg_aligned.astype(np.float64)

    # Adapter filtering
    ch_per_adapter = 64
    if target_adapters is not None:
        keep_mask = np.zeros(emg.shape[0], dtype=bool)
        for a in target_adapters:
            start = a * ch_per_adapter
            end = start + ch_per_adapter
            keep_mask[start : min(end, emg.shape[0])] = True
        emg = emg[keep_mask, :]

    if verbose:
        print("  Bandpass ...")
    emg = _apply_bandpass(emg, fs, bandpass[0], bandpass[1])

    if verbose:
        print("  Notch ...")
    emg = _apply_notch_comb(emg, fs, notch_base, notch_q)

    # ponytail: removed time_diff and spatial bipolar per advisor request.
    return emg.astype(np.float32)
