import numpy as np
import mne
from mne.connectivity import spectral_connectivity
def compute_coherence(data, sfreq, bands, fmin=None, fmax=None, method='coh', mode='multitaper', tmin=None, tmax=None):
    """
    Computes coherence for given EEG data.
    Parameters:
    - data: MNE Raw or Epochs object.
    - sfreq: Sampling frequency.
    - bands: Dictionary of frequency bands (e.g., {'alpha': (8, 13)}).
    - fmin, fmax: Frequency range for connectivity calculation. If None, uses band ranges.
    - method: Connectivity measure (e.g., 'coh', 'pli', 'wpli').
    - mode: Spectral estimation mode (e.g., 'multitaper', 'fourier').
    - tmin, tmax: Time window for connectivity calculation.
    Returns:
    - connectivity_features: Dictionary of coherence values for each band and channel pair.
    """
    if isinstance(data, np.ndarray):
        info = mne.create_info(ch_names=[f'EEG {i:03d}' for i in range(data.shape[0])], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        epochs = mne.Epochs(raw, events=np.array([[0, 0, 1]]), tmin=0, tmax=(data.shape[1]-1)/sfreq, preload=True, baseline=None)
    elif isinstance(data, mne.io.BaseRaw):
        epochs = mne.Epochs(data, events=np.array([[0, 0, 1]]), tmin=0, tmax=(data.n_times-1)/sfreq, preload=True, baseline=None)
    elif isinstance(data, mne.Epochs):
        epochs = data
    else:
        raise ValueError("Data must be a numpy array, MNE Raw, or MNE Epochs object.")
    ch_names = epochs.ch_names
    n_channels = len(ch_names)
    indices = mne.connectivity.seed_target_indices(ch_names, ch_names, exclude='self')
    connectivity_features = {}
    for band_name, (b_fmin, b_fmax) in bands.items():
        current_fmin = fmin if fmin is not None else b_fmin
        current_fmax = fmax if fmax is not None else b_fmax
        con = spectral_connectivity(
            epochs, method=method, mode=mode, sfreq=sfreq,
            fmin=current_fmin, fmax=current_fmax,
            tmin=tmin, tmax=tmax,
            indices=indices,
            faverage=True, # Average across frequencies in the band
            verbose=False
        )
        connectivity_features[band_name] = con.get_data(output='dense')[0, :] # (n_connections,)
    return connectivity_features
