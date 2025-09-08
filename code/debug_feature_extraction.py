import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from utils_io import load_raw_set
from features_band_aperiodic import extract_all_features
from config import BANDS, EXCLUDE_APERIODIC, WIN_SEC, WIN_OVERLAP, FS_TARGET
def print_dict_structure(d, indent=0):
    """Recursively prints the structure of a dictionary, showing keys and value shapes/lengths."""
    for key, value in d.items():
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        elif isinstance(value, (np.ndarray, list)):
            value_arr = np.array(value)
        else:
def flatten_features(feat_dict):
    """Flattens the feature dictionary into a 1D numpy array."""
    flattened = []
    for key in sorted(feat_dict.keys()):
        value = feat_dict[key]
        if isinstance(value, dict):
            for sub_key in sorted(value.keys()):
                sub_value = value[sub_key]
                if isinstance(sub_value, (np.ndarray, list)):
                    flattened.extend(np.array(sub_value).flatten())
                else:
                    flattened.append(float(sub_value))
        elif isinstance(value, (np.ndarray, list)):
            flattened.extend(np.array(value).flatten())
        else:
            flattened.append(float(value))
    return np.array(flattened)
def main():
    """Main function to run the debug comparison."""
    base_dir = Path("/Users/hyeongsuk/Library/CloudStorage/OneDrive-개인/HS_논문작성/DB/EEG/step3_cleaned")
    normal_file = base_dir / "135_FA00101R_2_processed_ica_cleaned.set"
    problem_file = base_dir / "136_2_processed_ica_cleaned.set"
    try:
        raw_normal = load_raw_set(normal_file, fs_target=FS_TARGET)
        
        window_normal = raw_normal.get_data(start=0, stop=int(WIN_SEC * raw_normal.info['sfreq']))
        
        features_normal = extract_all_features(
            window_normal,
            fs=raw_normal.info['sfreq'],
            bands=BANDS,
            exclude_aperiodic=EXCLUDE_APERIODIC,
            win_sec=WIN_SEC,
            win_overlap=WIN_OVERLAP,
            ch_names=raw_normal.ch_names
        )
        
        print_dict_structure(features_normal)
        flat_normal = flatten_features(features_normal)
    except Exception as e:
