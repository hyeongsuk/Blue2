import numpy as np
def slice_passages(raw, win_sec=4.0, overlap=0.5):
    """S1_Start..S6_End 주석을 이용해 각 2분 구간을 윈도로 분할.
    반환: list of dict(passage, t_elapsed, data(ndarray: n_ch x n_samp_window))
    """
    ann = raw.annotations
    srate = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)
    win_len = int(win_sec * srate)
    hop = int(win_len * (1.0 - overlap))
    out = []
    for i in range(1,7):
        start_key = f"S{i}_Start"; end_key = f"S{i}_End"
        starts = [a['onset'] for a in ann if a['description']==start_key]
        ends   = [a['onset'] for a in ann if a['description']==end_key]
        if len(starts)==0 or len(ends)==0:
            continue
        t0 = starts[0]; t1 = ends[0]
        i0 = np.searchsorted(times, t0)
        i1 = np.searchsorted(times, t1)
        seg = data[:, i0:i1]
        for s in range(0, seg.shape[1]-win_len+1, hop):
            w = seg[:, s:s+win_len]
            t_elapsed = (s + win_len) / srate
            out.append(dict(passage=i, t_elapsed=t_elapsed, data=w))
    return out
