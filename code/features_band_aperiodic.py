import numpy as np
from scipy.signal import welch, spectrogram
from scipy.stats import linregress
from sklearn.feature_selection import mutual_info_regression
import mne
from features_riemann import concat_and_tangent
def compute_psd(x, fs, nperseg=512, method='welch'):
    """Enhanced PSD computation with multiple methods"""
    if method == 'welch':
        f, pxx = welch(x, fs=fs, nperseg=nperseg)
    elif method == 'multitaper':
        if x.ndim == 1:
            x = x[np.newaxis, :]
        f, pxx = [], []
        for ch_data in x:
            psd = mne.time_frequency.psd_array_multitaper(
                ch_data[np.newaxis, :], fs, fmin=1, fmax=40, 
                bandwidth=2, n_jobs=1, verbose=False
            )
            if len(f) == 0:
                f = psd[1]
            pxx.append(psd[0][0])
        pxx = np.array(pxx)
    return f, pxx
def bandpower_from_psd(freqs, pxx, band):
    """Enhanced band power with relative power option"""
    fmin, fmax = band
    idx = (freqs >= fmin) & (freqs < fmax)
    if pxx.ndim == 1:
        pxx = pxx[np.newaxis, :]
    
    abs_power = np.log(np.mean(pxx[:, idx] + 1e-12, axis=1))
    
    total_power = np.sum(pxx + 1e-12, axis=1)
    band_power = np.sum(pxx[:, idx] + 1e-12, axis=1)
    rel_power = np.log(band_power / total_power + 1e-12)
    
    return abs_power, rel_power
def aperiodic_fit(freqs, pxx, exclude=(8,13)):
    """Enhanced aperiodic fitting with FOOOF-style approach"""
    x = np.log10(freqs + 1e-9)
    y = np.log10(pxx + 1e-12)
    mask = (freqs >= 1) & (freqs <= 40)
    if exclude:
        mask &= ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    xfit = x[mask]
    slopes, offsets, r_squared = [], [], []
    
    for ch in range(y.shape[0]):
        yfit = y[ch, mask]
        slope, intercept, r_val, _, _ = linregress(xfit, yfit)
        slopes.append(slope)
        offsets.append(intercept)
        r_squared.append(r_val**2)
    
    return np.array(slopes), np.array(offsets), np.array(r_squared)
def compute_time_resolved_band_power(data, fs, bands, win_sec, win_overlap):
    """
    Computes time-resolved band power using a sliding window.
    Returns a dictionary where keys are band names and values are 2D arrays (channels x windows).
    """
    n_channels, n_times = data.shape
    win_samples = int(win_sec * fs)
    step_samples = int(win_samples * (1 - win_overlap))
    
    # Ensure at least one window
    if n_times < win_samples:
        # Return zeros if data is too short for even one window
        time_resolved_features = {band_name: np.zeros((n_channels, 0)) for band_name in bands.keys()}
        return time_resolved_features
        
    n_windows = (n_times - win_samples) // step_samples + 1
    
    time_resolved_features = {band_name: [] for band_name in bands.keys()}
    
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + win_samples
        
        window_data = data[:, start_idx:end_idx]
        freqs, pxx = compute_psd(window_data, fs)
        
        for band_name, band_range in bands.items():
            abs_pow, rel_pow = bandpower_from_psd(freqs, pxx, band_range)
            time_resolved_features[band_name].append(abs_pow) # Append absolute power for each channel
            
    # Convert lists of arrays to 2D arrays (channels x windows)
    for band_name in bands.keys():
        time_resolved_features[band_name] = np.array(time_resolved_features[band_name]).T
        
    return time_resolved_features
def compute_cross_frequency_coupling(data, fs, bands):
    """
    Simplified cross-frequency coupling - just alpha-gamma correlation per channel
    """
    try:
        from scipy.signal import butter, filtfilt, hilbert
        
        n_channels = data.shape[0]
        alpha_band = bands.get('alpha', (8, 13))
        gamma_band = bands.get('gamma', (30, 40))
        
        def bandpass_filter(data, lowcut, highcut, fs, order=2):
            nyq = 0.5 * fs
            low = max(lowcut / nyq, 0.01)
            high = min(highcut / nyq, 0.99)
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data, axis=-1)
        
        coupling_strength = np.zeros(n_channels)
        
        for ch in range(n_channels):
            try:
                alpha_filtered = bandpass_filter(data[ch:ch+1], alpha_band[0], alpha_band[1], fs)
                gamma_filtered = bandpass_filter(data[ch:ch+1], gamma_band[0], gamma_band[1], fs)
                
                alpha_phase = np.angle(hilbert(alpha_filtered[0]))
                gamma_amp = np.abs(hilbert(gamma_filtered[0]))
                
                coupling_strength[ch] = np.corrcoef(alpha_phase, gamma_amp)[0, 1]
                if not np.isfinite(coupling_strength[ch]):
                    coupling_strength[ch] = 0.0
                    
            except:
                coupling_strength[ch] = 0.0
        
        return {'alpha_gamma_coupling': coupling_strength}
        
    except:
        return {'alpha_gamma_coupling': np.zeros(data.shape[0])}
def compute_spectral_entropy(data, fs, bands):
    """Compute spectral entropy for each frequency band"""
    freqs, pxx = compute_psd(data, fs)
    entropy_features = {}
    
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        band_pxx = pxx[:, idx]
        
        prob_pxx = band_pxx / (np.sum(band_pxx, axis=1, keepdims=True) + 1e-12)
        
        entropy = -np.sum(prob_pxx * np.log(prob_pxx + 1e-12), axis=1)
        entropy_features[f'{band_name}_entropy'] = entropy
    
    return entropy_features
def compute_visual_attention_features(data, fs, ch_names):
    """
    Simplified visual attention features
    """
    try:
        features = {}
        
        occipital_chs = [i for i, ch in enumerate(ch_names) if ch.startswith(('O', 'PO'))]
        parietal_chs = [i for i, ch in enumerate(ch_names) if ch.startswith(('P', 'CP'))]
        
        if len(occipital_chs) > 0 and len(parietal_chs) > 0:
            freqs, pxx = compute_psd(data, fs)
            alpha_idx = (freqs >= 8) & (freqs <= 13)
            
            occipital_alpha = np.mean(pxx[occipital_chs][:, alpha_idx])
            parietal_alpha = np.mean(pxx[parietal_chs][:, alpha_idx])
            
            if parietal_alpha > 1e-12:
                alpha_ratio = np.log(max(occipital_alpha, 1e-12) / parietal_alpha)
            else:
                alpha_ratio = 0.0
                
            features['occ_par_alpha_ratio'] = alpha_ratio
            
            if len(occipital_chs) >= 1 and len(parietal_chs) >= 1:
                try:
                    occ_signal = np.mean(data[occipital_chs], axis=0)
                    par_signal = np.mean(data[parietal_chs], axis=0)
                    connectivity = np.corrcoef(occ_signal, par_signal)[0, 1]
                    if not np.isfinite(connectivity):
                        connectivity = 0.0
                    features['occ_par_connectivity'] = connectivity
                except:
                    features['occ_par_connectivity'] = 0.0
        else:
            features['occ_par_alpha_ratio'] = 0.0
            features['occ_par_connectivity'] = 0.0
        
        return features
        
    except:
        return {'occ_par_alpha_ratio': 0.0, 'occ_par_connectivity': 0.0}
def extract_all_features(window_data, fs, bands, exclude_aperiodic, win_sec, win_overlap, ch_names=None):
    """
    Extracts comprehensive features and returns them in a structured dictionary.
    """
    try:
        n_channels = window_data.shape[0]
        
        freqs, pxx = compute_psd(window_data, fs, method='welch')
        
        structured_features = {
            'traditional_band_powers': {},
            'relative_band_powers': {},
            'aperiodic_slope': np.zeros(n_channels),
            'aperiodic_offset': np.zeros(n_channels),
            'aperiodic_r_squared': np.zeros(n_channels),
            'cross_frequency_coupling': {},
            'spectral_entropy': {},
            'visual_attention': {},
            'time_resolved_band_powers': {},
            'riemann_features': np.zeros(n_channels * (n_channels + 1) // 2) # Placeholder for Riemann features
        }
        
        # Band Powers
        for band_name, band_range in bands.items():
            abs_pow, rel_pow = bandpower_from_psd(freqs, pxx, band_range)
            structured_features['traditional_band_powers'][band_name] = abs_pow
            structured_features['relative_band_powers'][band_name] = rel_pow
        
        # Aperiodic Fit
        slope, offset, r_squared = aperiodic_fit(freqs, pxx, exclude=exclude_aperiodic)
        structured_features['aperiodic_slope'] = slope
        structured_features['aperiodic_offset'] = offset
        structured_features['aperiodic_r_squared'] = r_squared
        
        # Time-Resolved Band Powers
        time_features = compute_time_resolved_band_power(window_data, fs, bands, win_sec, win_overlap)
        structured_features['time_resolved_band_powers'] = time_features
        
        # Cross-Frequency Coupling
        cfc_features = compute_cross_frequency_coupling(window_data, fs, bands)
        structured_features['cross_frequency_coupling'] = cfc_features
        
        # Spectral Entropy
        entropy_features = compute_spectral_entropy(window_data, fs, bands)
        structured_features['spectral_entropy'] = entropy_features
        
        # Visual Attention Features
        if ch_names is not None:
            visual_features = compute_visual_attention_features(window_data, fs, ch_names)
            structured_features['visual_attention'] = visual_features
        
        # Riemann Features
        try:
            riemann_feats = concat_and_tangent(window_data)
            structured_features['riemann_features'] = riemann_feats
        except Exception as riemann_e:
            print(f"Warning: Could not compute Riemann features: {riemann_e}")
            structured_features['riemann_features'] = np.zeros(n_channels * (n_channels + 1) // 2) # Default to zeros
        
        # Handle NaNs/Infinities for all values in structured_features
        def handle_nans_recursive(obj):
            if isinstance(obj, dict):
                return {k: handle_nans_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [handle_nans_recursive(elem) for elem in obj]
            elif isinstance(obj, np.ndarray):
                return np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
            elif isinstance(obj, (float, int)):
                return obj if np.isfinite(obj) else 0.0
            else:
                return obj
        
        structured_features = handle_nans_recursive(structured_features)
        
        return structured_features
        
    except Exception as e:
        n_channels = window_data.shape[0] if window_data.ndim > 1 else 1
        
        minimal_structured_features = {
            'traditional_band_powers': {band_name: np.zeros(n_channels) for band_name in bands.keys()},
            'relative_band_powers': {band_name: np.zeros(n_channels) for band_name in bands.keys()},
            'aperiodic_slope': np.zeros(n_channels),
            'aperiodic_offset': np.zeros(n_channels),
            'aperiodic_r_squared': np.zeros(n_channels),
            'cross_frequency_coupling': {'alpha_gamma_coupling': np.zeros(n_channels)},
            'spectral_entropy': {band_name: np.zeros(n_channels) for band_name in bands.keys()},
            'visual_attention': {'occ_par_alpha_ratio': 0.0, 'occ_par_connectivity': 0.0},
            'time_resolved_band_powers': {band_name: np.zeros((n_channels, 2)) for band_name in bands.keys()}, # Assuming 2 windows for simplified
            'riemann_features': np.zeros(n_channels * (n_channels + 1) // 2) # Placeholder for Riemann features
        }
        
        return minimal_structured_features
