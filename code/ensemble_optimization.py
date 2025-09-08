"""
Day 1: Advanced Ensemble Methods Implementation
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from tqdm import tqdm
from checkpoint_manager import save_step_checkpoint, load_step_checkpoint
from config import OUTPUT_DIR
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
class WeightedVotingClassifier(BaseEstimator, ClassifierMixin):
    """동적 가중치 계산을 통한 고급 Voting Classifier"""
    
    def __init__(self, estimators, cv_folds=5):
        self.estimators = estimators
        self.cv_folds = cv_folds
        self.weights_ = None
        self.fitted_estimators_ = {}
        
    def fit(self, X, y):
        """각 모델의 CV 성능을 기반으로 최적 가중치 계산"""
        
        cv_scores = {}
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, estimator in tqdm(self.estimators, desc="Training base models"):
            scores = cross_val_score(estimator, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            cv_scores[name] = scores.mean()
            
            estimator.fit(X, y)
            self.fitted_estimators_[name] = estimator
        
        scores_array = np.array(list(cv_scores.values()))
        exp_scores = np.exp(scores_array * 5)  # 온도 파라미터로 차이 증폭
        self.weights_ = exp_scores / exp_scores.sum()
        
        return self
    
    def predict_proba(self, X):
        """가중 평균을 통한 확률 예측"""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet")
        
        probas = []
        for name, _ in self.estimators:
            estimator = self.fitted_estimators_[name]
            proba = estimator.predict_proba(X)[:, 1]  # positive class probability
            probas.append(proba)
        
        weighted_proba = np.average(probas, weights=self.weights_, axis=0)
        
        result = np.column_stack([1 - weighted_proba, weighted_proba])
        return result
    
    def predict(self, X):
        """가중 평균 기반 예측"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
def create_base_estimators():
    """다양한 기본 분류기들 생성"""
    
    estimators = [
        ('lr', LogisticRegression(random_state=42, max_iter=1000, C=0.1)),
        ('rf', RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, n_jobs=-1)),
        ('svm', SVC(random_state=42, probability=True, C=1.0, gamma='scale', cache_size=1000)),
        ('nb', GaussianNB())
    ]
    
    return estimators
def implement_advanced_voting(X_train, y_train, X_test, y_test):
    """고급 Voting Classifier 구현"""
    
    
    base_estimators = create_base_estimators()
    
    standard_voting = VotingClassifier(
        estimators=base_estimators,
        voting='soft'
    )
    standard_voting.fit(X_train, y_train)
    standard_pred = standard_voting.predict_proba(X_test)[:, 1]
    standard_auc = roc_auc_score(y_test, standard_pred)
    
    
    weighted_voting = WeightedVotingClassifier(base_estimators)
    weighted_voting.fit(X_train, y_train)
    weighted_pred = weighted_voting.predict_proba(X_test)[:, 1]
    weighted_auc = roc_auc_score(y_test, weighted_pred)
    
    
    return {
        'standard_voting': standard_voting,
        'weighted_voting': weighted_voting,
        'standard_auc': standard_auc,
        'weighted_auc': weighted_auc,
        'improvement': weighted_auc - standard_auc
    }
def implement_stacking_ensemble(X_train, y_train, X_test, y_test):
    """Stacking Ensemble 구현"""
    
    
    level0_estimators = [
        ('lr', LogisticRegression(random_state=42, C=0.5)),
        ('rf', RandomForestClassifier(random_state=42, n_estimators=50, max_depth=6)),
        ('svm', SVC(random_state=42, probability=True, C=2.0)),
        ('nb', GaussianNB()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    meta_learners = {
        'LogisticRegression': LogisticRegression(random_state=42, C=10),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for meta_name, meta_learner in tqdm(meta_learners.items(), desc="Training stacking models"):
        stacking_clf = StackingClassifier(
            estimators=level0_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        stacking_clf.fit(X_train, y_train)
        stacking_pred = stacking_clf.predict_proba(X_test)[:, 1]
        stacking_auc = roc_auc_score(y_test, stacking_pred)
        
        results[meta_name] = {
            'model': stacking_clf,
            'auc': stacking_auc,
            'predictions': stacking_pred
        }
        
    
    best_meta = max(results.keys(), key=lambda k: results[k]['auc'])
    best_auc = results[best_meta]['auc']
    
    
    return results, best_meta, best_auc
def ensemble_cross_validation(X, y, ensemble_models):
    """앙상블 모델들의 교차 검증"""
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in tqdm(ensemble_models.items(), desc="Cross-validation"):
        if hasattr(model, 'predict_proba'):
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            cv_results[name] = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'scores': cv_scores
            }
    
    return cv_results
def run_ensemble_optimization():
    """Day 1 앙상블 최적화 실행"""
    
    
    try:
        data_path = Path("outputs/quick_test_summary.csv")
        if not data_path.exists():
            return None
            
        df = pd.read_csv(data_path)
        
        X = df.drop(['label', 'subject_id'], axis=1, errors='ignore').values
        y = df['label'].values if 'label' in df.columns else df.iloc[:, -1].values
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        baseline_lr = LogisticRegression(random_state=42)
        baseline_lr.fit(X_train, y_train)
        baseline_pred = baseline_lr.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, baseline_pred)
        
        
        voting_results = implement_advanced_voting(X_train, y_train, X_test, y_test)
        
        stacking_results, best_stacking, best_stacking_auc = implement_stacking_ensemble(
            X_train, y_train, X_test, y_test
        )
        
        ensemble_models = {
            'Baseline_LR': baseline_lr,
            'Standard_Voting': voting_results['standard_voting'],
            'Weighted_Voting': voting_results['weighted_voting'],
            f'Best_Stacking_{best_stacking}': stacking_results[best_stacking]['model']
        }
        
        cv_results = ensemble_cross_validation(X_scaled, y, ensemble_models)
        
        
        all_aucs = {
            'baseline': baseline_auc,
            'standard_voting': voting_results['standard_auc'],
            'weighted_voting': voting_results['weighted_auc'],
            'best_stacking': best_stacking_auc
        }
        
        best_method = max(all_aucs.keys(), key=lambda k: all_aucs[k])
        best_auc_final = all_aucs[best_method]
        
        
        results_summary = {
            'day': 1,
            'method': 'Ensemble Optimization',
            'baseline_auc': baseline_auc,
            'best_method': best_method,
            'best_auc': best_auc_final,
            'improvement': best_auc_final - baseline_auc,
            'all_results': all_aucs
        }
        
        return results_summary
        
    except Exception as e:
        return None
def run_ensemble_optimization_integrated(X, y, baseline_auc=None):
    """기존 파이프라인과 통합된 앙상블 최적화"""
    
    
    if baseline_auc:
        pass
    
    
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    if X.ndim == 1:
        return None
    elif X.ndim != 2:
        return None
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if baseline_auc is None:
        baseline_lr = LogisticRegression(random_state=42)
        baseline_lr.fit(X_train, y_train)
        baseline_pred = baseline_lr.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, baseline_pred)
    
    voting_results = implement_advanced_voting(X_train, y_train, X_test, y_test)
    
    stacking_results, best_stacking, best_stacking_auc = implement_stacking_ensemble(
        X_train, y_train, X_test, y_test
    )
    
    all_aucs = {
        'baseline': baseline_auc,
        'weighted_voting': voting_results['weighted_auc'],
        'best_stacking': best_stacking_auc
    }
    
    best_method = max(all_aucs.keys(), key=lambda k: all_aucs[k])
    best_auc_final = all_aucs[best_method]
    improvement = best_auc_final - baseline_auc
    
    best_model = baseline_lr if best_method == 'baseline' else (
        voting_results['weighted_voting'] if best_method == 'weighted_voting' else
        stacking_results[best_stacking]['model']
    )
    y_pred = best_model.predict(X_test)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('Original Features Distribution')
    plt.boxplot(X_scaled)
    plt.subplot(122)
    plt.title('Scaled Features Distribution')
    plt.boxplot(X_scaled)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png')
    plt.close()
    
    feature_importances = np.abs(best_model.coef_[0]) if hasattr(best_model, 'coef_') else None
    
    if feature_importances is not None:
        plt.figure(figsize=(15, 5))
        plt.title('Feature Contributions')
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Coefficient')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_contributions.png')
        plt.close()
    
    
    results = {
        'best_method': best_method,
        'best_auc': best_auc_final,
        'improvement': improvement,
        'all_aucs': all_aucs,
        'voting_results': voting_results,
        'stacking_results': {
            'best_stacking': best_stacking,
            'best_stacking_auc': best_stacking_auc,
            'all_results': stacking_results
        },
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return results
if __name__ == "__main__":
    results = run_ensemble_optimization()
    if results:
        pass
