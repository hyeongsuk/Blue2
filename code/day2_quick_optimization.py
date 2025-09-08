"""
Day 2: Quick Hyperparameter Optimization
빠른 그리드 서치를 통한 효율적 최적화
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
class QuickHyperparameterOptimizer:
    """빠른 하이퍼파라미터 최적화"""
    
    def __init__(self, X, y, cv_folds=3):  # CV folds 줄임
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.best_params = {}
        self.best_scores = {}
        
    def optimize_logistic_regression(self):
        """Logistic Regression 빠른 최적화"""
        
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        clf = LogisticRegression(random_state=42, max_iter=500)
        grid_search = GridSearchCV(
            clf, param_grid, cv=self.cv_folds, 
            scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(self.X_scaled, self.y)
        
        self.best_params['LogisticRegression'] = grid_search.best_params_
        self.best_scores['LogisticRegression'] = grid_search.best_score_
        
        
        return grid_search.best_params_, grid_search.best_score_
    
    def optimize_random_forest(self):
        """Random Forest 빠른 최적화"""
        
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            clf, param_grid, cv=self.cv_folds,
            scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(self.X_scaled, self.y)
        
        self.best_params['RandomForest'] = grid_search.best_params_
        self.best_scores['RandomForest'] = grid_search.best_score_
        
        
        return grid_search.best_params_, grid_search.best_score_
    
    def optimize_svm(self):
        """SVM 빠른 최적화"""
        
        
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        clf = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            clf, param_grid, cv=self.cv_folds,
            scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(self.X_scaled, self.y)
        
        self.best_params['SVM'] = grid_search.best_params_
        self.best_scores['SVM'] = grid_search.best_score_
        
        
        return grid_search.best_params_, grid_search.best_score_
    
    def optimize_knn(self):
        """KNN 빠른 최적화"""
        
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
        
        clf = KNeighborsClassifier()
        grid_search = GridSearchCV(
            clf, param_grid, cv=self.cv_folds,
            scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(self.X_scaled, self.y)
        
        self.best_params['KNN'] = grid_search.best_params_
        self.best_scores['KNN'] = grid_search.best_score_
        
        
        return grid_search.best_params_, grid_search.best_score_
    
    def create_optimized_ensemble(self):
        """최적화된 파라미터로 앙상블 생성"""
        
        
        optimized_estimators = []
        
        lr_params = self.best_params.get('LogisticRegression', {})
        lr = LogisticRegression(random_state=42, max_iter=500, **lr_params)
        optimized_estimators.append(('lr_opt', lr))
        
        rf_params = self.best_params.get('RandomForest', {})
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params)
        optimized_estimators.append(('rf_opt', rf))
        
        svm_params = self.best_params.get('SVM', {})
        svm = SVC(probability=True, random_state=42, **svm_params)
        optimized_estimators.append(('svm_opt', svm))
        
        knn_params = self.best_params.get('KNN', {})
        knn = KNeighborsClassifier(**knn_params)
        optimized_estimators.append(('knn_opt', knn))
        
        optimized_estimators.append(('nb', GaussianNB()))
        
        voting_ensemble = VotingClassifier(
            estimators=optimized_estimators,
            voting='soft'
        )
        
        meta_learner = LogisticRegression(
            random_state=42, max_iter=500, **lr_params
        )
        
        stacking_ensemble = StackingClassifier(
            estimators=optimized_estimators,
            final_estimator=meta_learner,
            cv=3,  # CV folds 줄임
            stack_method='predict_proba'
        )
        
        return voting_ensemble, stacking_ensemble
    
    def run_quick_optimization(self):
        """빠른 최적화 프로세스 실행"""
        
        
        baseline_lr = LogisticRegression(random_state=42)
        baseline_scores = cross_val_score(
            baseline_lr, self.X_scaled, self.y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        baseline_auc = baseline_scores.mean()
        
        self.optimize_logistic_regression()
        self.optimize_random_forest()
        self.optimize_svm()
        self.optimize_knn()
        
        voting_ensemble, stacking_ensemble = self.create_optimized_ensemble()
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        
        voting_scores = cross_val_score(voting_ensemble, self.X_scaled, self.y, cv=cv, scoring='roc_auc')
        stacking_scores = cross_val_score(stacking_ensemble, self.X_scaled, self.y, cv=cv, scoring='roc_auc')
        
        voting_auc = voting_scores.mean()
        stacking_auc = stacking_scores.mean()
        
        
        if stacking_auc > voting_auc:
            best_ensemble = stacking_ensemble
            best_auc = stacking_auc
            best_type = 'Optimized Stacking'
            improvement = stacking_auc - baseline_auc
        else:
            best_ensemble = voting_ensemble
            best_auc = voting_auc
            best_type = 'Optimized Voting'
            improvement = voting_auc - baseline_auc
        
        
        return {
            'best_model': best_ensemble,
            'best_auc': best_auc,
            'best_type': best_type,
            'improvement': improvement,
            'baseline_auc': baseline_auc,
            'voting_auc': voting_auc,
            'stacking_auc': stacking_auc,
            'optimized_params': self.best_params,
            'individual_scores': self.best_scores
        }
def run_day2_quick_optimization(X, y):
    """Day 2 빠른 하이퍼파라미터 최적화 실행"""
    
    optimizer = QuickHyperparameterOptimizer(X, y)
    results = optimizer.run_quick_optimization()
    
    return results
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    
    X, y = make_classification(
        n_samples=3540, n_features=180, n_informative=50,
        n_redundant=30, class_sep=0.6, random_state=42
    )
    
    
    results = run_day2_quick_optimization(X, y)
    
    if results:
        pass
