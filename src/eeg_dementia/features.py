"""
Spectral feature extraction for preprocessed EEG epochs (OpenNeuro ds004504).
 
Consolidates the band-power extraction logic developed interactively in
notebooks/03_feature_extraction.ipynb into reusable functions, so it can be
applied consistently across many subjects without copy-pasting notebook cells.
"""
 
import numpy as np
import pandas as pd
 
# Standard EEG frequency band boundaries (Hz). Theta is this project's primary
# focus, motivated by the "EEG slowing" pattern described in dementia EEG
# literature (see project reference docs for citations/rationale).
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}
 
 
def compute_psd(epochs, fmin=0.5, fmax=45.0, n_fft=1024, verbose=False):
    """
    Compute a power spectral density estimate via Welch's method.
 
    Thin wrapper around epochs.compute_psd(), centralizing the parameter
    choices established and justified during Stage 4 (see reference docs):
    n_fft=1024 gives a ~2.048s effective window (Δf ≈ 0.488 Hz), comfortably
    finer resolution than needed to separate the bands in BANDS.
    """
    return epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=verbose
    )
 
 
def extract_relative_band_power(psd, bands=None):
    """
    Compute relative (normalized) band power per channel from a PSD.
 
    Relative power — each band's power as a fraction of that same channel's
    own total power across all bands — is used instead of absolute power
    because absolute EEG power is not directly comparable across subjects
    (confounded by non-neural factors like electrode contact, skull
    thickness, and cap fit). See Stage 4 reference docs for the investigation
    that motivated this choice.
 
    Parameters
    ----------
    psd : mne.time_frequency.EpochsSpectrum
        PSD object from compute_psd(), shape (n_epochs, n_channels, n_freqs).
    bands : dict, optional
        Mapping of band name -> (fmin, fmax). Defaults to BANDS.
 
    Returns
    -------
    dict
        {band_name: np.ndarray of shape (n_channels,)}, relative power per
        channel, averaged across epochs.
    """
    if bands is None:
        bands = BANDS
 
    data = psd.get_data()          # (n_epochs, n_channels, n_freqs)
    freqs = psd.freqs
 
    mean_psd = data.mean(axis=0)   # average across epochs -> (n_channels, n_freqs)
 
    band_absolute = {}
    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        band_absolute[band_name] = mean_psd[:, freq_mask].mean(axis=1)  # (n_channels,)
 
    total_power_per_channel = sum(band_absolute[b] for b in bands)
 
    band_relative = {
        band_name: band_absolute[band_name] / total_power_per_channel
        for band_name in bands
    }
    return band_relative
 
 
def build_feature_row(sub_id, group, epochs, bands=None):
    """
    Build one subject's feature row: relative band power for every
    (band, channel) combination, plus identifying columns.
 
    Returns
    -------
    dict
        {'subject_id': ..., 'group': ..., 'delta_Fp1': ..., 'theta_Fp1': ..., ...}
        Ready to be collected into a list and passed to pd.DataFrame().
    """
    psd = compute_psd(epochs)
    rel_power = extract_relative_band_power(psd, bands=bands)
 
    row = {"subject_id": sub_id, "group": group}
    for band_name, values_per_channel in rel_power.items():
        for ch_name, value in zip(psd.ch_names, values_per_channel):
            row[f"{band_name}_{ch_name}"] = value
    return row
 
 
def build_feature_table(subjects_epochs, bands=None):
    """
    Build the full subject x feature table from multiple subjects' epochs.
 
    Parameters
    ----------
    subjects_epochs : dict
        {subject_id: (group_label, epochs)} for each subject to include.
    bands : dict, optional
        Mapping of band name -> (fmin, fmax). Defaults to BANDS.
 
    Returns
    -------
    pd.DataFrame
        One row per subject; columns are subject_id, group, and
        {band}_{channel} for every band/channel combination.
    """
    rows = []
    for sub_id, (group, epochs) in subjects_epochs.items():
        rows.append(build_feature_row(sub_id, group, epochs, bands=bands))
    return pd.DataFrame(rows)
 