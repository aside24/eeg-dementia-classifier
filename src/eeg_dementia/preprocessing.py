"""
Preprocessing pipeline for resting-state EEG (OpenNeuro ds004504).

Consolidates pipeline in notebooks/02_preprocessing.ipynb into a single reusable function
"""

import mne
import numpy as np


def preprocess_subject(
    raw_path,
    l_freq=0.5,
    h_freq=45.0,
    bad_segment_window_sec=1.0,
    bad_segment_percentile=95,
    ica_n_components=15,
    ica_random_state=42,
    eog_proxy_channel="Fp1",
    epoch_duration_sec=4.0,
    verbose=False,
):
    """
    Run the preprocessing pipeline on a single subject's raw EEG file.

    Steps: filtering -> bad channel check (informational only) -> bad segment
    annotation -> ICA eye-artifact removal -> average re-referencing -> fixed-length
    epoching with bad-segment rejection.

    Returns
    -------
    epochs : mne.Epochs
        \ epoched data ready for feature extraction.
    info : dict
        Summary diagnostics (channel z-scores, bad segment count, excluded ICA
        component, epoch counts) for sanitycheck each subject's run.
    """
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=verbose)

    # Step 1: Filtering
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # Step 2: Bad channel detection (statistical check; informational — see
    # notes on why elevated frontal channels are generally an ICA target, not an
    # interpolation target, unless a channel is extreme relative to its own recording)
    data = raw_filtered.get_data()
    channel_variances = np.var(data, axis=1)
    z_scores = (channel_variances - channel_variances.mean()) / channel_variances.std()
    channel_z = dict(zip(raw_filtered.ch_names, z_scores))

    # Step 3: Bad segment detection (empirically thresholded peak2peak amplitude)
    sfreq = raw_filtered.info["sfreq"]
    window_samples = int(bad_segment_window_sec * sfreq)
    n_samples = data.shape[1]

    all_ptps = []
    for start in range(0, n_samples - window_samples, window_samples):
        window = data[:, start : start + window_samples]
        all_ptps.append((window.max(axis=1) - window.min(axis=1)).max())
    all_ptps = np.array(all_ptps)
    threshold = np.percentile(all_ptps, bad_segment_percentile)

    bad_onsets, bad_durations = [], []
    for start in range(0, n_samples - window_samples, window_samples):
        window = data[:, start : start + window_samples]
        ptp = window.max(axis=1) - window.min(axis=1)
        if np.any(ptp > threshold):
            bad_onsets.append(start / sfreq)
            bad_durations.append(bad_segment_window_sec)

    annotations = mne.Annotations(
        onset=np.array(bad_onsets),
        duration=np.array(bad_durations),
        description=["BAD_amplitude"] * len(bad_onsets),
    )
    raw_filtered.set_annotations(annotations)

    # Step 4: ICA eyea  rtifact removal
    raw_for_ica = raw_filtered.copy().filter(l_freq=1.0, h_freq=None, verbose=verbose)
    ica = mne.preprocessing.ICA(
        n_components=ica_n_components, random_state=ica_random_state, max_iter="auto"
    )
    ica.fit(raw_for_ica, verbose=verbose)

    eog_indices, eog_scores = ica.find_bads_eog(
        raw_for_ica, ch_name=eog_proxy_channel, verbose=verbose
    )

    # Fallback: find_bads_eog()'s internal automatic threshold can miss a real
    # artifact component when more than one component shows elevated correlation
    # (it compresses the z-score of the true artifact relative to the group). If
    # the automatic pass found nothing, fall back to a manual correlation-magnitude
    # cutoff rather than silently proceeding with zero components excluded.
    manual_correlation_cutoff = 0.5
    if len(eog_indices) == 0:
        eog_indices = [
            i for i, s in enumerate(eog_scores) if abs(s) > manual_correlation_cutoff
        ]

    ica.exclude = eog_indices

    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean, verbose=verbose)

    # Step 5: Rereferencing
    raw_reref = raw_clean.copy().set_eeg_reference("average", verbose=verbose)

    # Step 6: Epoching
    epochs = mne.make_fixed_length_epochs(
        raw_reref,
        duration=epoch_duration_sec,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True,
        verbose=verbose,
    )

    info = {
        "channel_z_scores": channel_z,
        "bad_segment_threshold_uv": threshold * 1e6,
        "n_bad_segments": len(bad_onsets),
        "excluded_ica_components": eog_indices,
        "eog_correlation_scores": eog_scores,
        "n_epochs_total": len(epochs.drop_log),
        "n_epochs_kept": len(epochs),
    }

    return epochs, info