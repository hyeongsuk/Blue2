import re, glob, os
from pathlib import Path
import pandas as pd
import mne
def list_set_files(data_dir, pattern="*_cleaned.set"):
    return sorted(glob.glob(str(Path(data_dir) / pattern)))
def read_seq_map(csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index('Subjects')
    return df
def lens_of(seq_map, subject_id, passage_idx):
    subject_id_int = int(subject_id)
    sequence = seq_map.loc[subject_id_int, 'Sequence']
    passage_idx = int(passage_idx)
    if sequence == 1:
        if 1 <= passage_idx <= 3:
            return 0  # Normal
        elif 4 <= passage_idx <= 6:
            return 1  # BLF
    elif sequence == 2:
        if 1 <= passage_idx <= 3:
            return 1  # BLF
        elif 4 <= passage_idx <= 6:
            return 0  # Normal
            
    raise ValueError(f"Invalid passage_idx '{passage_idx}' or sequence '{sequence}' for subject '{subject_id}'")
def subject_from_filename(filepath):
    name = Path(filepath).stem
    m = re.match(r'^([A-Za-z0-9]+)', name)
    return m.group(1) if m else name
def load_raw_set(filepath, fs_target=None, drop_eog=True):
    raw = mne.io.read_raw_eeglab(filepath, preload=True)
    if fs_target:
        raw.resample(fs_target)
    if drop_eog:
        eog_like = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
        if eog_like:
            raw.drop_channels(eog_like)
    return raw
