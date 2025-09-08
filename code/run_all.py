import os, sys, json
import numpy as np
import pandas as pd
from config import DATA_DIR, SEQ_CSV, FS_TARGET, WIN_SEC, WIN_OVERLAP, BANDS, EXCLUDE_APERIODIC, T_LIST, MDT_THRESH, OUTPUT_DIR, FIG_DIR, TIER_STRATEGY, TIERS
from utils_io import list_set_files, read_seq_map, subject_from_filename, load_raw_set, lens_of
from windowing import slice_passages
from features_band_aperiodic import extract_all_features
from time_to_detection import auc_vs_time_band, compute_mdt, bootstrap_ci_subject, auc_vs_time_enhanced, analyze_feature_importance
from channel_selection import pick_by_roi
from plots import plot_auc_vs_time
from ensemble_optimization import run_ensemble_optimization_integrated
def main():
    files = list_set_files(DATA_DIR)
    if not files:
        sys.exit(1)
    seq_map = read_seq_map(SEQ_CSV)
    import mne
    tmp = mne.io.read_raw_eeglab(files[0], preload=False)
    all_ch = tmp.ch_names
    tier_defs = []
    for n in TIERS:
        if n >= len(all_ch):
            tier_defs.append((f"all{len(all_ch)}", None))
        else:
            if TIER_STRATEGY == 'roi':
                sel = pick_by_roi(all_ch, n)
                tier_defs.append((f"roi{n}", sel))
            else:
                sel = pick_by_roi(all_ch, n)
                tier_defs.append((f"data{n}", sel))
    summary_rows = []
    time_resolved_features_collection = []
    for tier_name, sel_chs in tier_defs:
        rows = []
        for fp in files:
            sid = subject_from_filename(fp)
            raw = load_raw_set(fp, fs_target=FS_TARGET, drop_eog=True)
            if sel_chs:
                keep = [ch for ch in raw.ch_names if ch in sel_chs]
                raw.pick_channels(keep)
            win_list = slice_passages(raw, win_sec=WIN_SEC, overlap=WIN_OVERLAP)
            for w in win_list:
                ch_names = raw.ch_names if sel_chs is None else sel_chs
                all_feats_dict = extract_all_features(w['data'], fs=raw.info['sfreq'], bands=BANDS,
                                                    exclude_aperiodic=EXCLUDE_APERIODIC,
                                                    win_sec=WIN_SEC, win_overlap=WIN_OVERLAP,
                                                    ch_names=ch_names)
                feat_list = []
                
                def ensure_1d(arr):
                    """Ensure array is 1D for concatenation"""
                    arr = np.asarray(arr)
                    if arr.ndim == 0:  # scalar
                        return np.array([arr])
                    elif arr.ndim == 1:
                        return arr
                    else:  # 2D or higher
                        return arr.flatten()
                
                for band_name in BANDS.keys():
                    feat_list.append(ensure_1d(all_feats_dict['traditional_band_powers'][band_name]))
                    if f'{band_name}_mt' in all_feats_dict['traditional_band_powers']:
                        feat_list.append(ensure_1d(all_feats_dict['traditional_band_powers'][f'{band_name}_mt']))
                
                for band_name in BANDS.keys():
                    feat_list.append(ensure_1d(all_feats_dict['relative_band_powers'][band_name]))
                    if f'{band_name}_mt' in all_feats_dict['relative_band_powers']:
                        feat_list.append(ensure_1d(all_feats_dict['relative_band_powers'][f'{band_name}_mt']))
                
                feat_list.append(ensure_1d(all_feats_dict['aperiodic_slope']))
                feat_list.append(ensure_1d(all_feats_dict['aperiodic_offset']))
                feat_list.append(ensure_1d(all_feats_dict['aperiodic_r_squared']))
                
                for cfc_name, cfc_val in all_feats_dict['cross_frequency_coupling'].items():
                    feat_list.append(ensure_1d(cfc_val))
                
                for entropy_name, entropy_val in all_feats_dict['spectral_entropy'].items():
                    feat_list.append(ensure_1d(entropy_val))
                
                for vis_name, vis_val in all_feats_dict['visual_attention'].items():
                    feat_list.append(ensure_1d(vis_val))
                
                for band_name, tr_powers in all_feats_dict['time_resolved_band_powers'].items():
                    if tr_powers.size > 0:
                        feat_list.append(ensure_1d(np.mean(tr_powers, axis=1)))  # mean across time
                        feat_list.append(ensure_1d(np.std(tr_powers, axis=1)))   # std across time
                        feat_list.append(ensure_1d(np.min(tr_powers, axis=1)))   # min across time
                        feat_list.append(ensure_1d(np.max(tr_powers, axis=1)))   # max across time
                
                feat_list = [f for f in feat_list if f.size > 0]
                feat = np.concatenate(feat_list, axis=0)
                rows.append(dict(subject=sid, passage=w['passage'], t_elapsed=w['t_elapsed'], feat=feat))
                time_resolved_features_collection.append(dict(
                    subject=sid,
                    passage=w['passage'],
                    t_elapsed=w['t_elapsed'],
                    time_resolved_band_powers=all_feats_dict['time_resolved_band_powers']
                ))
        df = pd.DataFrame(rows)
        df['y'] = df.apply(lambda r: lens_of(seq_map, r['subject'], r['passage']), axis=1)
        curve_baseline = auc_vs_time_band(df, T_LIST)
        curve_baseline.to_csv(OUTPUT_DIR / f"auc_curve_baseline_{tier_name}.csv", index=False)
        mdt_baseline = compute_mdt(curve_baseline, thr=MDT_THRESH)
        
        model_types = ['ensemble', 'rf', 'logistic', 'svm']
        best_auc = 0
        best_model = 'baseline'
        best_curve = curve_baseline
        best_mdt = mdt_baseline
        
        
        for model_type in model_types:
            try:
                curve_enhanced = auc_vs_time_enhanced(df, T_LIST, model_type=model_type)
                if not curve_enhanced.empty:
                    curve_enhanced.to_csv(OUTPUT_DIR / f"auc_curve_{model_type}_{tier_name}.csv", index=False)
                    mdt_enhanced = compute_mdt(curve_enhanced, thr=MDT_THRESH)
                    max_auc = curve_enhanced['AUC'].max()
                    
                    
                    if max_auc > best_auc:
                        best_auc = max_auc
                        best_model = model_type
                        best_curve = curve_enhanced
                        best_mdt = mdt_enhanced
                        
            except Exception as e:
        
        
        lo, hi = bootstrap_ci_subject(df, lambda D, TL: auc_vs_time_enhanced(D, TL, model_type=best_model), 
                                    T_LIST, thr=MDT_THRESH, B=200)
        summary_rows.append(dict(
            tier=tier_name, 
            best_model=best_model,
            mdt=best_mdt, 
            mdt_ci_lo=lo, 
            mdt_ci_hi=hi,
            max_auc=best_auc,
            baseline_mdt=mdt_baseline,
            baseline_max_auc=curve_baseline['AUC'].max()
        ))
        
        try:
            importance_analysis = analyze_feature_importance(df, T=60, model_type='rf')
            if importance_analysis:
                np.save(OUTPUT_DIR / f"feature_importance_{tier_name}.npy", importance_analysis)
        except Exception as e:
        plot_auc_vs_time([best_curve], [f"{tier_name}_{best_model}"], 
                        fig_path=FIG_DIR / f"auc_vs_time_best_{tier_name}.png")
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary_mdt.csv", index=False)
if __name__ == "__main__":
    main()
