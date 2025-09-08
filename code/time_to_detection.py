import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from enhanced_models import auc_vs_time_enhanced, analyze_feature_importance
from ensemble_optimization import run_ensemble_optimization_integrated
def aggregate_up_to_T_band(df, T):
    sub = df[df['t_elapsed'] <= T].copy()
    if len(sub)==0:
        return None, None, None
    grouped = []
    for (s,p,y), g in sub.groupby(['subject','passage','y']):
        feats = np.stack(g['feat'].values, axis=0)
        grouped.append(dict(subject=s, passage=p, y=y, feat=feats.mean(axis=0)))
    G = pd.DataFrame(grouped)
    X = np.stack(G['feat'].values, axis=0)
    y = G['y'].values
    groups = G['subject'].values
    return X, y, groups
def auc_vs_time_ensemble(df_all, T_list, C=1.0):
    """앙상블 기법을 사용한 AUC vs Time 분석"""
    out = []
    for T in T_list:
        X, y, groups = aggregate_up_to_T_band(df_all, T)
        if X is None:
            out.append(dict(T=T, AUC=0.5))
            continue
            
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, C=C))
        ])
        
        gkf = GroupKfold(n_splits=5)
        baseline_scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe.fit(X_train, y[train_idx])
            y_pred_proba = pipe.predict_proba(X_test)[:, 1]
            baseline_scores.append(roc_auc_score(y_test, y_pred_proba))
        baseline_auc = np.mean(baseline_scores)
        
        try:
            ensemble_results = run_ensemble_optimization_integrated(X, y, baseline_auc)
            best_auc = ensemble_results['best_auc']
            improvement = ensemble_results['improvement']
        except:
            best_auc = baseline_auc
            improvement = 0
            
        out.append(dict(
            T=T, 
            AUC=best_auc,
            baseline_AUC=baseline_auc,
            improvement=improvement
        ))
    return out
def auc_vs_time_band(df_all, T_list, C=1.0):
    out = []
    for T in T_list:
        X, y, groups = aggregate_up_to_T_band(df_all, T)
        if X is None: 
            continue
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        y_true, y_prob = [], []
        for tr, te in gkf.split(X, y, groups):
            model = Pipeline([('scaler', StandardScaler()), 
                              ('clf', LogisticRegression(max_iter=2000, C=C, solver='lbfgs'))])
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[te])[:,1]
            y_true.extend(y[te]); y_prob.extend(prob)
        auc = roc_auc_score(y_true, y_prob)
        out.append((T, auc))
    return pd.DataFrame(out, columns=['T','AUC'])
def compute_mdt(auc_curve, thr=0.70):
    ok = auc_curve[auc_curve['AUC'] >= thr]
    return float(ok['T'].min()) if len(ok) else float('inf')
def bootstrap_ci_subject(df_all, fn_auc_curve, T_list, thr=0.70, B=500, seed=42):
    rng = np.random.RandomState(seed)
    subjects = df_all['subject'].unique()
    mdt_list = []
    for _ in range(B):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        boot = df_all[df_all['subject'].isin(pick)].copy()
        curve = fn_auc_curve(boot, T_list)
        mdt = compute_mdt(curve, thr)
        if np.isfinite(mdt):
            mdt_list.append(mdt)
    lo, hi = np.percentile(mdt_list, [2.5, 97.5]) if len(mdt_list)>0 else (np.nan, np.nan)
    return lo, hi