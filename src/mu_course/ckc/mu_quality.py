# src/models/v3/mu_quality.py
"""MU separation quality assessment and visualization.

Computes quality metrics (SIL, PNR, CoV ISI, firing rate) for each
extracted motor unit, generates diagnostic raster plots and console
reports. Used both standalone and integrated into train_mu.py.

Metrics:
    - SIL (Silhouette): [-1, 1] K-means cluster separation quality
    - PNR (Pulse-to-Noise Ratio): dB, signal-to-noise at spike locations
    - CoV ISI: Firing regularity (lower = more regular)
    - Mean FR (Hz): Mean firing rate, physiological validity check

Usage:
    from mu_course.ckc.mu_quality import generate_quality_report

    filtered, metrics = generate_quality_report(
        mu_pulses, ipts, sils, pnrs, fsamp=2048,
        output_dir='results/quality',
        filter_flagged=True,
    )
"""

import json
import os

import numpy as np

# ===========================================================================
# MU-level quality metrics
# ===========================================================================


def compute_mu_metrics(mu_pulses, ipts, sils, pnrs, fsamp=2048):
    """Compute per-MU quality metrics.

    Args:
        mu_pulses: List[np.ndarray] spike sample indices per MU.
        ipts: List[np.ndarray] IPT signals (same length as EMG) per MU.
              Can be empty list if IPT not available.
        sils: List[float] silhouette values per MU.
        pnrs: List[float] PNR values (dB) per MU.
        fsamp: Sampling rate (Hz).

    Returns:
        List[dict] sorted by mean_fr_hz (slow -> fast, recruitment order).
        Each dict:
            mu_id, n_spikes, sil, pnr_db,
            cov_isi, mean_fr_hz, min_fr_hz, max_fr_hz,
            flagged, flag_reasons
    """
    metrics = []
    for mu_idx in range(len(mu_pulses)):
        pulses = mu_pulses[mu_idx]
        n_spikes = len(pulses)
        sil = sils[mu_idx] if mu_idx < len(sils) else 0.0
        pnr = pnrs[mu_idx] if mu_idx < len(pnrs) else 0.0

        # ISI-based metrics
        cov_isi, mean_fr, min_fr, max_fr = _compute_isi_metrics(pulses, fsamp)

        flagged, reasons = _flag_mu(sil, pnr, cov_isi, mean_fr, n_spikes)

        metrics.append(
            {
                "mu_id": mu_idx,  # original index (before sorting)
                "sort_idx": 0,  # placeholder, filled after sort
                "n_spikes": n_spikes,
                "sil": float(sil),
                "pnr_db": float(pnr),
                "cov_isi": float(cov_isi) if cov_isi is not None else None,
                "mean_fr_hz": float(mean_fr) if mean_fr is not None else None,
                "min_fr_hz": float(min_fr) if min_fr is not None else None,
                "max_fr_hz": float(max_fr) if max_fr is not None else None,
                "flagged": flagged,
                "flag_reasons": reasons,
            }
        )

    # Sort by mean firing rate (slow -> fast, matches recruitment order)
    metrics.sort(key=lambda m: m["mean_fr_hz"] if m["mean_fr_hz"] is not None else 999)
    # Assign sort order
    for i, m in enumerate(metrics):
        m["sort_idx"] = i

    return metrics


def _compute_isi_metrics(pulses, fsamp):
    """Compute ISI-based metrics for a single MU.

    Returns:
        (cov_isi, mean_fr, min_fr, max_fr) or (None, None, None, None) if
        insufficient spikes.
    """
    if len(pulses) < 3:
        return None, None, None, None

    isi = np.diff(np.sort(pulses.astype(np.float64)))
    # Remove outliers: ISI > 1 second (physiologically implausible)
    isi = isi[isi < fsamp]

    if len(isi) < 2:
        return None, None, None, None

    mean_isi = isi.mean()
    std_isi = isi.std(ddof=1)  # Match MATLAB std() which divides by N-1
    cov_isi = std_isi / mean_isi if mean_isi > 0 else 999.0

    ifr = fsamp / isi  # instantaneous firing rates
    mean_fr = fsamp / mean_isi
    min_fr = ifr.min()
    max_fr = ifr.max()

    return cov_isi, mean_fr, min_fr, max_fr


def _flag_mu(sil, pnr, cov_isi, mean_fr, n_spikes, sil_threshold=0.8, pnr_threshold=1.0):
    """Determine if a MU fails quality checks.

    Criteria:
        - SIL < sil_threshold: insufficient spike/noise separation
        - PNR < pnr_threshold: weak pulse relative to noise (threshold lowered to 1.0 dB to match MATLAB which does not filter by PNR)
        - Mean FR < 2 or > 50 Hz: adjusted for dynamic contractions
        - CoV ISI > 3.0: high variance allowed for intermittent bursting
        - Fewer than 15 spikes: insufficient data
    """
    reasons = []

    if sil < sil_threshold:
        reasons.append(f"SIL={sil:.3f}")
    if pnr < pnr_threshold:
        reasons.append(f"PNR={pnr:.1f}dB")
    if mean_fr is not None and (mean_fr < 2.0 or mean_fr > 50.0):
        reasons.append(f"FR={mean_fr:.1f}Hz")
    if cov_isi is not None and cov_isi > 3.0:
        reasons.append(f"CoV={cov_isi:.2f}")
    if n_spikes < 15:
        reasons.append(f"N={n_spikes}")

    return len(reasons) > 0, reasons


# ===========================================================================
# Console quality report
# ===========================================================================


def print_quality_report(metrics):
    """Print formatted MU quality report to console.

    Args:
        metrics: List[dict] from compute_mu_metrics().
    """
    if not metrics:
        print("No MUs to report.")
        return

    n_total = len(metrics)
    n_flagged = sum(1 for m in metrics if m["flagged"])

    print(f"\n{'=' * 80}")
    sep4 = "-" * 4
    sep6 = "-" * 6
    sep7 = "-" * 7
    sep10 = "-" * 10

    print(f"  MU Quality Report ({n_total} MUs extracted, {n_flagged} flagged)")
    print(f"{'=' * 80}")
    print(
        f"  {'MU#':>4s}  {'N':>6s}  {'SIL':>6s}  {'PNR':>7s}  "
        f"{'CoV':>7s}  {'FR(Hz)':>7s}  {'Status':s}"
    )
    print(f"  {sep4:>4s}  {sep6:>6s}  {sep6:>6s}  {sep7:>7s}  {sep7:>7s}  {sep7:>7s}  {sep10:>10s}")

    for m in metrics:
        sil_s = f"{m['sil']:.3f}" if m["sil"] is not None else "  N/A"
        pnr_s = f"{m['pnr_db']:.1f}" if m["pnr_db"] is not None else "  N/A"
        cov_s = f"{m['cov_isi']:.2f}" if m["cov_isi"] is not None else "  N/A"
        fr_s = f"{m['mean_fr_hz']:.1f}" if m["mean_fr_hz"] is not None else "  N/A"

        if m["flagged"]:
            status = " !! " + ", ".join(m["flag_reasons"])
        else:
            status = "  OK"

        print(
            f"  {m['mu_id']:4d}  {m['n_spikes']:6d}  {sil_s:>6s}  "
            f"{pnr_s:>7s}  {cov_s:>7s}  {fr_s:>7s}  {status}"
        )

    print(f"  {sep4:>4s}  {sep6:>6s}  {sep6:>6s}  {sep7:>7s}  {sep7:>7s}  {sep7:>7s}  {sep10:>10s}")

    # Summary statistics
    valid = [m for m in metrics if not m["flagged"]]
    if valid:
        avg_sil = np.mean([m["sil"] for m in valid])
        avg_pnr = np.mean([m["pnr_db"] for m in valid])
        avg_fr = np.mean([m["mean_fr_hz"] for m in valid if m["mean_fr_hz"] is not None])
        avg_spikes = np.mean([m["n_spikes"] for m in valid])
        print(
            f"  Valid MUs ({len(valid)}): mean SIL={avg_sil:.3f}, "
            f"PNR={avg_pnr:.1f}dB, FR={avg_fr:.1f}Hz, spikes={avg_spikes:.0f}"
        )
    print(f"{'=' * 80}\n")


# ===========================================================================
# Visualization
# ===========================================================================


def plot_mu_raster(
    mu_pulses, metrics, n_samples, fsamp=2048, save_path=None, title=None, show_ipt=False, ipts=None
):
    """Generate MU spike raster plot.

    Top panel: raster of spike trains (vertical ticks per spike, color-coded).
    Right margin: per-MU annotation (SIL | PNR | CoV | FR).
    MUs sorted by mean firing rate (slow -> fast).

    Args:
        mu_pulses: List[np.ndarray] spike sample indices per MU.
        metrics: List[dict] from compute_mu_metrics().
        n_samples: Total signal length in samples.
        fsamp: Sampling rate (Hz).
        save_path: If provided, save PNG to this path.
        title: Optional plot title.
        show_ipt: If True, overlay IPT waveforms (requires ipts).
        ipts: List[np.ndarray] IPT signals (needed if show_ipt=True).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_mus = len(mu_pulses)
    if n_mus == 0:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(
            0.5,
            0.5,
            "No MUs to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    time_axis = np.arange(n_samples) / fsamp

    # Color map: gradient from blue (slow) to red (fast)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_mus - 1, 1)) for i in range(n_mus)]

    # Figure dimensions scale with MU count
    fig_h = max(4, 1.5 + n_mus * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for mu_idx in range(n_mus):
        pulses = mu_pulses[mu_idx]
        m = metrics[mu_idx] if mu_idx < len(metrics) else {}
        color = colors[mu_idx]

        # Raster: vertical lines at each spike
        for sp in pulses:
            if 0 <= sp < n_samples:
                t_sp = sp / fsamp
                ax.plot(
                    [t_sp, t_sp],
                    [mu_idx + 0.2, mu_idx + 0.8],
                    color=color,
                    linewidth=0.6,
                    alpha=0.8,
                )

        # Optional: IPT waveform overlay
        if show_ipt and ipts and mu_idx < len(ipts):
            ipt = ipts[mu_idx]
            if len(ipt) > 0:
                ipt_norm = ipt / (ipt.max() + 1e-10) * 0.3
                ax.plot(
                    time_axis[: len(ipt)],
                    ipt_norm + mu_idx + 0.5,
                    color=color,
                    linewidth=0.5,
                    alpha=0.4,
                )

        # Right margin annotation
        sil_s = f"{m.get('sil', 0):.2f}" if m.get("sil") is not None else "?"
        pnr_s = f"{m.get('pnr_db', 0):.0f}" if m.get("pnr_db") is not None else "?"
        cov_s = f"{m.get('cov_isi', 0):.2f}" if m.get("cov_isi") is not None else "?"
        fr_s = f"{m.get('mean_fr_hz', 0):.0f}" if m.get("mean_fr_hz") is not None else "?"

        label = f"S{sil_s}|P{pnr_s}|C{cov_s}|F{fr_s}"
        flag_marker = " !!" if m.get("flagged") else ""
        ax.text(
            time_axis[-1] * 1.02,
            mu_idx + 0.5,
            label + flag_marker,
            fontsize=6,
            va="center",
            fontfamily="monospace",
            color="red" if m.get("flagged") else "gray",
        )

    ax.set_ylim(0, n_mus)
    ax.set_xlim(0, time_axis[-1] * 1.25)  # Leave room for annotations
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("MU # (sorted by firing rate)", fontsize=11)
    ax.set_yticks(np.arange(n_mus) + 0.5)
    ax.set_yticklabels([f"MU{m['mu_id']}" for m in metrics], fontsize=7)

    ttl = title or f"MU Spike Raster ({n_mus} MUs)"
    n_flag = sum(1 for m in metrics if m.get("flagged"))
    if n_flag > 0:
        ttl += f" -- {n_flag} flagged !!"
    ax.set_title(ttl, fontsize=12)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_firing_rate_hist(metrics, fsamp=2048, save_path=None, title=None):
    """Plot histogram of mean firing rates with physiological reference bands.

    Args:
        metrics: List[dict] from compute_mu_metrics().
        fsamp: Sampling rate (Hz).
        save_path: If provided, save PNG.
        title: Optional plot title.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frs = [m["mean_fr_hz"] for m in metrics if m["mean_fr_hz"] is not None]
    if not frs:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No firing rate data", ha="center", va="center", transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Physiological range shading
    ax.axvspan(5, 35, alpha=0.08, color="green", label="Physiological (5-35 Hz)")

    bins = max(5, min(20, len(frs)))
    _n_bins, bin_edges, patches = ax.hist(
        frs, bins=bins, edgecolor="white", linewidth=0.8, alpha=0.8
    )

    # Color bars by range
    for i, patch in enumerate(patches):
        center = (bin_edges[i] + bin_edges[i + 1]) / 2
        if 5 <= center <= 35:
            patch.set_facecolor("#2ca02c")  # green
        elif center < 5 or center > 45:
            patch.set_facecolor("#d62728")  # red
        else:
            patch.set_facecolor("#ff7f0e")  # orange

    # Mark flagged MU FRs
    flagged_frs = [m["mean_fr_hz"] for m in metrics if m["flagged"] and m["mean_fr_hz"] is not None]
    if flagged_frs:
        ax.plot(
            flagged_frs,
            [0] * len(flagged_frs),
            "rx",
            markersize=8,
            markeredgewidth=1.5,
            label="Flagged",
        )

    # Median line
    median_fr = np.median(frs)
    ax.axvline(
        median_fr, color="black", linestyle="--", linewidth=1, label=f"Median: {median_fr:.1f} Hz"
    )

    ax.set_xlabel("Mean Firing Rate (Hz)", fontsize=11)
    ax.set_ylabel("MU Count", fontsize=11)
    ax.set_title(title or f"MU Firing Rate Distribution ({len(frs)} MUs)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ===========================================================================
# All-in-one convenience
# ===========================================================================


def generate_quality_report(
    mu_pulses,
    ipts,
    sils,
    pnrs,
    fsamp=2048,
    output_dir=None,
    filter_flagged=False,
    prefix="",
    sil_threshold=0.8,
    pnr_threshold=1.0,
):
    """Compute metrics, print report, save plots, optionally filter MUs.

    This is the primary entry point. Call after CKC decomposition.

    Args:
        mu_pulses: List[np.ndarray] spike indices per MU.
        ipts: List[np.ndarray] IPT signals per MU (can be [] if unavailable).
        sils: List[float] silhouette values.
        pnrs: List[float] PNR values (dB).
        fsamp: Sampling rate (Hz).
        output_dir: If provided, save raster PNG, histogram PNG, metrics JSON.
        filter_flagged: If True, return only non-flagged MUs.
        prefix: Prefix for output filenames (e.g., 'Middle_flexion_').
        sil_threshold: SIL below which MU is flagged (default 0.8, matching MATLAB).
        pnr_threshold: PNR (dB) below which MU is flagged (default 5.0).
                       MATLAB does not filter by PNR.

    Returns:
        (filtered_mu_pulses, metrics_list)
    """
    if not mu_pulses:
        print("No MUs to assess.")
        return ([], []) if filter_flagged else (mu_pulses, [])

    metrics = compute_mu_metrics(mu_pulses, ipts, sils, pnrs, fsamp)

    # Console report
    print_quality_report(metrics)

    # Save artifacts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Raster and histogram now generated by v3_viz.generate_quality_preview
        # (panel_a_raster.png + panel_d_fr_dist.png) with force boundary annotations.
        # Original code preserved for reference:
        # raster_path = os.path.join(output_dir, f'{prefix}mu_raster.png')
        # plot_mu_raster(mu_pulses, metrics, _estimate_n_samples(ipts, mu_pulses),
        #                fsamp=fsamp, save_path=raster_path,
        #                title=f'{prefix}MU Spike Raster')
        # hist_path = os.path.join(output_dir, f'{prefix}firing_rate_hist.png')
        # plot_firing_rate_hist(metrics, fsamp=fsamp, save_path=hist_path,
        #                       title=f'{prefix}MU Firing Rates')

        # JSON metrics
        json_path = os.path.join(output_dir, f"{prefix}mu_metrics.json")
        # Convert numpy types for JSON serialization
        json_metrics = []
        for m in metrics:
            jm = {}
            for k, v in m.items():
                if isinstance(v, (np.integer,)):
                    jm[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    jm[k] = float(v)
                elif isinstance(v, (np.ndarray,)):
                    jm[k] = v.tolist()
                else:
                    jm[k] = v
            json_metrics.append(jm)
        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"  Quality artifacts saved to: {output_dir}")

    # Filter
    if filter_flagged:
        flagged_original_ids = {m["mu_id"] for m in metrics if m["flagged"]}
        filtered_pulses = [
            mu_pulses[i] for i in range(len(mu_pulses)) if i not in flagged_original_ids
        ]
        n_removed = len(mu_pulses) - len(filtered_pulses)
        if n_removed > 0:
            print(f"  Filtered: {n_removed} flagged MUs removed, {len(filtered_pulses)} kept")
        return filtered_pulses, metrics

    return mu_pulses, metrics


def _estimate_n_samples(ipts, mu_pulses):
    """Estimate total signal length from available data."""
    if ipts and len(ipts) > 0 and len(ipts[0]) > 0:
        return len(ipts[0])
    if mu_pulses:
        all_spikes = np.concatenate([p for p in mu_pulses if len(p) > 0])
        return int(all_spikes.max()) + 1 if len(all_spikes) > 0 else 1000
    return 1000


# ===========================================================================
# Force Encoding Score (FES) — Phase 4.2
# ===========================================================================


def _pearson_r(x, y):
    """Pearson correlation coefficient (pure numpy, no scipy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    if denom < 1e-15:
        return 0.0
    return float((xm * ym).sum() / denom)


def compute_fes(mu_pulses, cfr, force, fsamp=2048):
    """Compute Force Encoding Score (FES) for each MU.

    FES = |Pearson(CFR_i, force)| * (1 - CoV/100) * (FR_range / FR_mean)

    Components:
        - |Pearson|: how well firing rate tracks force (0-1)
        - 1 - CoV/100: penalize irregular firing (CoV > 100 = noise)
        - FR_range / FR_mean: reward wide dynamic range

    Args:
        mu_pulses: List[np.ndarray] spike sample indices per MU.
        cfr: np.ndarray [n_samples, n_mus] continuous firing rate.
        force: np.ndarray [n_samples] force target signal.
        fsamp: Sampling rate (Hz).

    Returns:
        np.ndarray [n_mus] FES scores (higher = better force encoder).
    """
    n_mus = len(mu_pulses)
    fes = np.zeros(n_mus, dtype=np.float64)

    for mu_idx in range(n_mus):
        pulses = mu_pulses[mu_idx]
        if len(pulses) < 3:
            fes[mu_idx] = -1.0  # insufficient data
            continue

        # 1. |Pearson| between CFR and force
        cfr_i = cfr[: len(force), mu_idx].astype(np.float64)
        pearson = abs(_pearson_r(cfr_i, force))

        # 2. CoV penalty
        isi = np.diff(np.sort(pulses.astype(np.float64)))
        isi = isi[isi < fsamp]  # remove implausible gaps (>1s)
        if len(isi) < 2:
            fes[mu_idx] = pearson * 0.1  # low confidence
            continue

        mean_isi = isi.mean()
        std_isi = isi.std(ddof=1)
        cov = std_isi / mean_isi if mean_isi > 0 else 999.0
        cov_penalty = max(0.0, 1.0 - cov / 100.0)

        # 3. FR dynamic range reward
        ifr = fsamp / isi
        fr_mean = ifr.mean()
        fr_range = ifr.max() - ifr.min()
        dynamic_range = fr_range / fr_mean if fr_mean > 0 else 0.0

        fes[mu_idx] = pearson * cov_penalty * dynamic_range

    return fes


def select_mus_by_fes(mu_pulses, fes_scores, top_k=20, max_roa=0.3, fsamp=2048):
    """Select top-K MUs by FES, removing redundant ones via RoA.

    Args:
        mu_pulses: List[np.ndarray] spike indices per MU.
        fes_scores: np.ndarray [n_mus] FES scores.
        top_k: Maximum number of MUs to select.
        max_roa: Maximum Rate of Agreement for deduplication.
        fsamp: Sampling rate (Hz).

    Returns:
        (selected_pulses: List[np.ndarray], selected_indices: List[int])
    """
    from .ckc_decomposer import _rate_of_agreement

    n_mus = len(mu_pulses)
    if n_mus == 0:
        return [], []

    # Sort by FES descending, keep only positive scores
    order = np.argsort(fes_scores)[::-1]
    valid_mask = fes_scores[order] > 0
    order = order[valid_mask]

    if len(order) == 0:
        print("  [FES] No MUs with positive FES score")
        return [], []

    selected = [int(order[0])]
    d_ipi = int(fsamp / 100)
    lim = int(fsamp * 0.1)

    for idx in order[1:]:
        if len(selected) >= top_k:
            break
        # Check RoA against all already-selected MUs
        duplicate = False
        for sel_idx in selected:
            r = _rate_of_agreement(mu_pulses[sel_idx], mu_pulses[idx], lim, d_ipi)
            if r > max_roa:
                duplicate = True
                break
        if not duplicate:
            selected.append(int(idx))

    selected_pulses = [mu_pulses[i] for i in selected]
    print(
        f"  [FES] Selected {len(selected)}/{n_mus} MUs "
        f"(top FES={fes_scores[selected[0]]:.4f}, "
        f"min FES={fes_scores[selected[-1]]:.4f})"
    )
    return selected_pulses, selected
