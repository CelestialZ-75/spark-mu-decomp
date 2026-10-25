# src/models/v3/ckc_extender.py
"""CKC signal extension via delayed repetitions.

Port of example/CKC_MU_extraction/extend.m to PyTorch.
Creates ext_factor delayed copies of each EMG channel for
convolution kernel compensation.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F


class CKCExtender:
    """Extend EMG signals by adding delayed repetitions.

    For ext_factor=10 and 64 input channels, output has 640 channels.
    Each original channel is replicated 10 times with 0..9 sample delays.

    Output shape: [n_channels * ext_factor, n_samples + ext_factor - 1]
    """

    def __init__(self, ext_factor=10, device="cuda"):
        self.ext_factor = ext_factor
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and "cuda" in str(device)
            else torch.device("cpu")
        )

    def extend(self, emg):
        """Extend EMG signal.

        Args:
            emg: numpy float32 [n_channels, n_samples]

        Returns:
            torch.Tensor on self.device,
            [n_channels * ext_factor, n_samples + ext_factor - 1]
        """
        n_ch, n_samp = emg.shape
        ext_ch = n_ch * self.ext_factor
        ext_len = n_samp + self.ext_factor - 1

        # Estimate GPU memory: eSig + CorrSig + temp ≈ 3× the extended matrix
        est_gb = ext_ch * ext_len * 4 / (1024**3) * 3
        if self.device.type == "cuda" and est_gb > 3:
            gpu_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            warnings.warn(
                f"CKC extended signal estimated at {est_gb:.1f} GB GPU memory "
                f"(GPU has {gpu_total:.1f} GB). Consider using shorter segments "
                f"via SegmentDecomposer (seg_duration_s parameter)."
            )

        # --- GPU-vectorized extension ---
        # Build delayed copies via repeat + pad + roll.
        # Replaces 2560 sequential slice assignments with 10 batch operations,
        # reducing kernel launch overhead ~256x.
        emg_t = torch.from_numpy(np.asarray(emg, dtype=np.float32)).to(
            self.device
        )  # [n_ch, n_samp]

        # [n_ch, ext_factor, n_samp]
        eSig = emg_t.unsqueeze(1).repeat(1, self.ext_factor, 1)

        # Pad trailing zeros for delay shifts
        if self.ext_factor > 1:
            eSig = F.pad(
                eSig, (0, self.ext_factor - 1)
            )  # [n_ch, ext_factor, n_samp + ext_factor - 1]
            for m in range(self.ext_factor):
                eSig[:, m] = eSig[:, m].roll(m, dims=-1)  # shift row by m
            eSig[:, :, : self.ext_factor - 1] = 0  # clear rolled-in wrap-around

        eSig = eSig.reshape(ext_ch, ext_len)  # [ext_ch, ext_len]

        # Trim extension padding (matching MATLAB: eSig(:, extFactor+1:end-extFactor))
        return eSig[:, self.ext_factor : -self.ext_factor]

    def __repr__(self):
        return f"CKCExtender(ext_factor={self.ext_factor}, device='{self.device}')"
