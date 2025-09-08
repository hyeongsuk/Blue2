import re
import numpy as np
ROI_ORDER = [
    ['Oz','POz','O1','O2'],
    ['Pz','P1','P2','P3','P4'],
    ['CPz','CP1','CP2','Cz','C1','C2'],
]
def pick_by_roi(ch_names, target_n):
    chosen = []
    for group in ROI_ORDER:
        for ch in group:
            if ch in ch_names and ch not in chosen:
                chosen.append(ch)
                if len(chosen) == target_n:
                    return chosen
    pats = ['^O', '^PO', '^P', '^CP', '^C']
    for pat in pats:
        for ch in ch_names:
            if re.match(pat, ch) and ch not in chosen:
                chosen.append(ch)
                if len(chosen) == target_n:
                    return chosen
    return chosen
