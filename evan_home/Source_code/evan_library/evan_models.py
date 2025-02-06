import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class EvanModels:
    def __init__(self, data, all_features_dict, level='l1'):
        self.data = data
        self.all_features_dict = all_features_dict
        self.level = level  # Set to 'l1' or 'l2'
        self.label = self.data.obs[f'celltype.{self.level}'].tolist()
        self.types = np.unique(self.label).tolist()

    def LR_kfold(self, celltype, feature_set, k=5):
        # Subset data to celltype features
        X = self.data[:, feature_set].X
        # Binary label based on the specified level
        y = [1 if i == celltype else 0 for i in self.label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        lr_clf = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
        lr_clf.fit(X_train, y_train)

        # K-fold cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1_score': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',  # PR AUC
            'mcc': make_scorer(matthews_corrcoef)
        }

        print('Cross-validation...')
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = cross_validate(lr_clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=32)
        mean_metrics = [np.mean(cv_results[f'test_{metric}']) for metric in scoring.keys()]

        # Likelihood
        likelihood_all = lr_clf.predict_proba(X)[:, 1]
        print('likelihood > 0.5:', sum(likelihood_all > 0.5))  # decision_scores > 0 的有幾個
        
        return lr_clf, mean_metrics, cv_results, likelihood_all

    def run_LR_kfold_for_types(self, save_path=None):
        cols = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC', 'MCC']
        all_metrics = pd.DataFrame(columns=cols)
        cv_results_dict = {}
        likelihood_dict = {}
        
        for celltype in self.types:
            print('====================')
            print('K-fold CV for:', celltype)
            if isinstance(self.all_features_dict[celltype], list):
                print('is a list')
                features_celltype = self.all_features_dict[celltype]
            else:
                features_celltype = self.all_features_dict[celltype]['Gene'].tolist()
            lr_clf, metrics, cv_results, likelihood = self.LR_kfold(celltype, features_celltype, k=5)  # metrics is a list
            print(metrics)
            
            # Append metrics to all_metrics
            all_metrics.loc[celltype] = metrics
            # Record CV result
            cv_results_dict[celltype] = self.convert_arrays_to_lists(cv_results)
            # Record likelihood
            likelihood_dict[celltype] = likelihood

            # Save the model as a pickle file
            if save_path:
                filename = f'LRclassifier_{celltype}_{self.level}.pkl'
                with open(os.path.join(save_path, filename), 'wb') as f:
                    pickle.dump(lr_clf, f)
        return all_metrics, cv_results_dict, likelihood_dict

    def SVM_kfold(self, celltype, feature_set, svm_kernel='rbf', k=5):
        X = self.adata[:, feature_set].X
        y = [1 if i == celltype else 0 for i in self.label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        print('Fitting SVM model...')
        svm_clf = SVC(kernel=svm_kernel, decision_function_shape='ovr', class_weight='balanced', max_iter=10000)
        svm_clf.fit(X_train, y_train)

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1_score': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',  # PR AUC
            'mcc': make_scorer(matthews_corrcoef)
        }

        print('Cross-validation...')
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = cross_validate(svm_clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=32)

        mean_metrics = [np.mean(cv_results[f'test_{metric}']) for metric in scoring.keys()]
        return svm_clf, mean_metrics, cv_results

    def run_SVM_kfold_for_types(self, save_path=None):
        cols = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC', 'MCC']
        all_metrics = pd.DataFrame(columns=cols)
        cv_results_dict = {}
        for celltype in self.types:
            print('====================')
            print('K-fold CV for:', celltype)
            if isinstance(self.all_features_dict[celltype], list):
                features_celltype = self.all_features_dict[celltype]
            else:
                features_celltype = self.all_features_dict[celltype]['Gene'].tolist()

            svm_clf, metrics, cv_results = self.SVM_kfold(celltype, features_celltype, k=5)
            print(metrics)
            cv_results_dict[celltype] = cv_results
            all_metrics.loc[celltype] = metrics

            # Save the SVM model as a pickle file
            # filename = f'SVM_{celltype}_rbf_StandardScale_{self.level}_DEG.pkl'
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(svm_clf, f)

        return all_metrics, cv_results_dict

    def XGB_kfold(self, celltype, feature_set, k=5):
        # Subset data to celltype features
        X = self.data[:, feature_set].X
        # Binary label
        y = [1 if i == celltype else 0 for i in self.label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        xgb_clf = xgb.XGBClassifier(n_estimators=1000, 
                                    objective='binary:logistic', 
                                    # eval_metric='logloss',  # or 'auc', 'error'
                                    # early_stopping_rounds=50, 
                                    reg_lambda=1, 
                                    device='cuda', 
                                    verbose=True)
        xgb_clf.fit(X_train, y_train)

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1_score': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',
            'mcc': make_scorer(matthews_corrcoef)
        }

        print('Cross-validation...')
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=32)
        mean_metrics = [np.mean(cv_results[f'test_{metric}']) for metric in scoring.keys()]

        # Likelihood
        likelihood_all = xgb_clf.predict_proba(X)[:, 1]
        print('likelihood > 0.5:', sum(likelihood_all > 0.5))  # decision_scores > 0 的有幾個

        return xgb_clf, mean_metrics, cv_results, likelihood_all

    def run_XGB_kfold_for_types(self, save_path=None):
        cols = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC', 'MCC']
        all_metrics = pd.DataFrame(columns=cols)
        cv_results_dict = {}
        likelihood_dict = {}

        for celltype in self.types:
            print('====================')
            print('K-fold CV for:', celltype)
            if isinstance(self.all_features_dict[celltype], list):
                features_celltype = self.all_features_dict[celltype]
            else:
                features_celltype = self.all_features_dict[celltype]['Gene'].tolist()
            xgb_clf, metrics, cv_results, likelihood = self.XGB_kfold(celltype, features_celltype, k=5)
            print(metrics)

            # Record CV results fold-by-fold
            cv_results_dict[celltype] = self.convert_arrays_to_lists(cv_results)

            # Append metrics to all_metrics
            all_metrics.loc[celltype] = metrics

            # Record likelihood
            likelihood_dict[celltype] = likelihood

            # Save the model as a pickle file
            if save_path:
                filename = f'XGBclassifier_{celltype}_{self.level}.pkl'
                with open(os.path.join(save_path, filename), 'wb') as f:
                    pickle.dump(xgb_clf, f)

        return all_metrics, cv_results_dict, likelihood_dict

    def assign_likelihoods(self, likelihood_dict):
        likelihood_df = pd.DataFrame(likelihood_dict)
        largest_values, largest_columns, assignments = [], [], []

        for index, row in likelihood_df.iterrows():
            # find largest value and their corresponding columns
            largest_value = row.max()
            largest_column = row.idxmax()
            largest_values.append(largest_value)
            largest_columns.append(largest_column)
            assignments.append(largest_column)

        result_df = pd.DataFrame({'Largest Value': largest_values, 'Largest Column': largest_columns, 'Assignment': assignments})

        true_labels = self.label
        predicted_labels = result_df['Assignment'].tolist()

        # Create the confusion matrix
        cm = pd.crosstab(true_labels, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=False)
        # replace NaN with 0
        cm = cm.fillna(0)
        cm = cm.astype(int)

        # Evaluate metrics
        self.evaluate_multiclass_metrics(true_labels, predicted_labels)

        return cm

    def evaluate_multiclass_metrics(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        print("Accuracy:", accuracy)

        method = ['micro', 'macro', 'weighted']
        for m in method:
            precision = precision_score(true_labels, predicted_labels, average=m)
            print(f"{m} Precision:", precision)
            f1 = f1_score(true_labels, predicted_labels, average=m)
            print(f"{m} F1 Score:", f1)
        # return accuracy

    def plot_confusion_matrix(self, cm, figsize=(8, 8), title="Confusion Matrix", save_path=None):
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=300)

    def plot_confusion_matrix_proportion(self, cm, figsize=(8, 8), title="Confusion Matrix (Proportion %)", save_path=None):
        row_sum = cm.sum(axis=1)
        cm_proportion = cm.div(row_sum, axis=0) * 100
        plt.figure(figsize=figsize)
        sns.heatmap(cm_proportion, fmt=".1f", annot=True, cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=300)

    @staticmethod
    def convert_arrays_to_lists(data):
        if isinstance(data, dict):
            return {key: EvanModels.convert_arrays_to_lists(value) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
