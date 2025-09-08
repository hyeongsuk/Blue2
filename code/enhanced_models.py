import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_model(model_type='ensemble', n_features='auto'):
    """
    Create enhanced models with feature selection
    
    Parameters:
    - model_type: 'logistic', 'rf', 'ensemble', 'svm'
    - n_features: number of features to select, 'auto' for automatic selection
    """
    
    if model_type == 'logistic':
        if n_features == 'auto':
            selector = SelectKBest(score_func=mutual_info_classif, k=50)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', selector),
            ('clf', LogisticRegression(max_iter=2000, C=0.1, solver='liblinear', class_weight='balanced'))
        ])
    
    elif model_type == 'rf':
        if n_features == 'auto':
            selector = SelectKBest(score_func=mutual_info_classif, k=100)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', selector),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, 
                                         min_samples_split=5, min_samples_leaf=2,
                                         class_weight='balanced', random_state=42))
        ])
    
    elif model_type == 'svm':
        if n_features == 'auto':
            selector = SelectKBest(score_func=mutual_info_classif, k=30)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', selector),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', 
                       class_weight='balanced', probability=True, random_state=42))
        ])
    
    elif model_type == 'ensemble':
        logistic_model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_classif, k=50)),
            ('clf', LogisticRegression(max_iter=2000, C=0.1, solver='liblinear', 
                                     class_weight='balanced', random_state=42))
        ])
        
        rf_model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_classif, k=100)),
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=8,
                                         min_samples_split=5, class_weight='balanced', 
                                         random_state=42))
        ])
        
        gb_model = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_classif, k=75)),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=6,
                                              learning_rate=0.1, random_state=42))
        ])
        
        model = VotingClassifier(
            estimators=[
                ('logistic', logistic_model),
                ('rf', rf_model), 
                ('gb', gb_model)
            ],
            voting='soft'
        )
    
    return model

def auc_vs_time_enhanced(df_all, T_list, model_type='ensemble', n_features='auto'):
    """
    Enhanced AUC vs time computation with advanced models
    """
    out = []
    for T in T_list:
        sub = df_all[df_all['t_elapsed'] <= T].copy()
        if len(sub) == 0:
            continue
            
        grouped = []
        for (s,p,y), g in sub.groupby(['subject','passage','y']):
            feats = np.stack(g['feat'].values, axis=0)
            grouped.append(dict(subject=s, passage=p, y=y, feat=feats.mean(axis=0)))
        
        G = pd.DataFrame(grouped)
        X = np.stack(G['feat'].values, axis=0)
        y = G['y'].values
        groups = G['subject'].values
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        if len(np.unique(y)) < 2 or len(X) < 5:
            continue
            
        model = create_enhanced_model(model_type, n_features)
        
        gkf = GroupKFold(n_splits=min(len(np.unique(groups)), 10))
        y_true, y_prob = [], []
        
        for tr, te in gkf.split(X, y, groups):
            try:
                model.fit(X[tr], y[tr])
                prob = model.predict_proba(X[te])
                if prob.shape[1] == 2:
                    prob = prob[:, 1]
                else:
                    prob = prob[:, 0]
                y_true.extend(y[te])
                y_prob.extend(prob)
            except Exception as e:
                print(f"Fold error: {e}")
                continue
        
        if len(y_true) > 0 and len(set(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
                out.append((T, auc))
            except Exception as e:
                print(f"AUC calculation error: {e}")
    
    return pd.DataFrame(out, columns=['T','AUC'])

def analyze_feature_importance(df_all, T=60, model_type='rf'):
    """
    Analyze which features are most important for classification
    """
    sub = df_all[df_all['t_elapsed'] <= T].copy()
    if len(sub) == 0:
        return None
        
    grouped = []
    for (s,p,y), g in sub.groupby(['subject','passage','y']):
        feats = np.stack(g['feat'].values, axis=0)
        grouped.append(dict(subject=s, passage=p, y=y, feat=feats.mean(axis=0)))
    
    G = pd.DataFrame(grouped)
    X = np.stack(G['feat'].values, axis=0)
    y = G['y'].values
    
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X, y)
        importances = model.feature_importances_
    else:
        importances = mutual_info_classif(X, y, random_state=42)
    
    feature_idx = np.argsort(importances)[::-1]
    
    return {
        'importances': importances,
        'feature_ranking': feature_idx,
        'top_10_importance': importances[feature_idx[:10]]
    }