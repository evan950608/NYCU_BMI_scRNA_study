# %%
import numpy as np
import pandas as pd
import time
import os
import scipy
from scipy.sparse import *
import copy
from collections import deque
import warnings

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class ADlasso2():
    def __init__(self, lmbd=1e-5, max_iter=100, tol=1e-4, alpha=0.1, gamma=0.1, device='cpu', echo=False):
        super().__init__()
        self.lmbd = lmbd
        self.max_iter = int(max_iter)
        self.tol = tol
        self.alpha = alpha
        self.gamma = gamma
        self.memory = 10
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCELoss()
        self.UsingDevice = device
        self.echo = echo
        self.classes_ = None
        self.n_iter_ = None
        self.loss_ = None
        self.convergence_ = None
        self.w = None
        self.b = None
        self.feature_set = None
        self.feature_sort = None
        
    def np2tensor(self, X):
        X = torch.from_numpy(X).float()
        self.n_samples, self.n_features = X.shape
        bias = torch.ones(self.n_samples).reshape(-1, 1) # create a all 1 vector for bias
        X_data = torch.cat((X, bias), 1).to(self.UsingDevice)        # append the all 1 column into X for representing bias
        return X_data, X_data.T
    
    def coo2sparse_tensor(self, X):
        self.n_samples, self.n_features = X.shape
        bias = np.ones(self.n_samples).reshape(-1, 1)  # create a all 1 vector for bias
        X = scipy.sparse.hstack((X, bias))                         # append the all 1 column into X for representing bias
        values = X.data; v = torch.FloatTensor(values)
        indices = np.vstack((X.row, X.col)); i = torch.LongTensor(indices)
        Xshape = X.shape
        X_data = torch.sparse_coo_tensor(i, v, size=Xshape, dtype=torch.float32, requires_grad=False).to(self.UsingDevice) 
        X_dataT = torch.transpose(X_data, 0, 1).to(self.UsingDevice)
        return X_data, X_dataT
        
    def initialize(self, X, Y, pvl):
        class_content = np.unique(Y)
        if len(class_content) != 2:
            raise ValueError("This solver needs samples of at only 2 classes, try to use MultiClassADlasso.")
        
        if self.UsingDevice not in ['cpu','cuda']:
            raise ValueError("Wrong device assignment.")

        if self.UsingDevice == 'cuda':
            if not torch.cuda.is_available():
                print("your GPU is not available, ADlasso is running with CPU.")
                self.UsingDevice = 'cpu'
        
        if type(X) is np.ndarray:
            X, XT = self.np2tensor(X)
        elif type(X) is pd.DataFrame:
            X, XT = self.np2tensor(X.to_numpy())
        elif isspmatrix_csr(X):
            X, XT = self.coo2sparse_tensor(X.tocoo())
        elif isspmatrix_csc(X):
            X, XT = self.coo2sparse_tensor(X.tocoo())
        elif isspmatrix_coo(X):
            X, XT = self.coo2sparse_tensor(X)
        else :
            raise ValueError("X is unrecognizable data type")
            
        if len(Y) != self.n_samples:
            raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])

        y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
        y = torch.from_numpy(y).float().reshape(self.n_samples, 1).to(self.UsingDevice)
        self.classes_ = {class_content[0] : 0, class_content[1] : 1}
        
        if pvl.shape[0] != self.n_features:
            raise ValueError("Found input prevalence vector with inconsistent numbers of features: %r" % [self.n_features, pvl.shape[0]])
        pvl = np.append(pvl,1)     # append 1 for representing bias prevalence
        pvl = torch.from_numpy(pvl).float().reshape(-1, 1).to(self.UsingDevice)
        
        weight = torch.zeros(self.n_features+1, requires_grad = False).reshape(-1, 1).to(self.UsingDevice)
        return X, XT, y, pvl, weight
    
    def logistic_gradient(self, X, XT, y, w):
        resid = self.sigmoid(X.mm(w)) - y
        gradient = XT.mm(resid)/self.n_samples
        return gradient
    
    def pseudo_grad(self, x, grad, thres):
        grad_r = thres * x.sign()
        grad_right = grad + torch.where(x != 0, grad_r, thres)
        grad_left = grad + torch.where(x != 0, grad_r, -thres)
        grad_pseudo = torch.zeros_like(x)
        grad_pseudo = torch.where(grad_left > 0, grad_left, grad_pseudo)
        grad_pseudo = torch.where(grad_right < 0, grad_right, grad_pseudo)
        return grad_pseudo

    def lbfgs_2_recursion(self, q, H, hess_0):
        if len(H):
            s, y = H.pop()
            q_i = q - (s.T.mm(q)) / (y.T.mm(s)) * y
            r = self.lbfgs_2_recursion(q_i, H, hess_0)
            return r + ((s.T.mm(q) - y.T.mm(r)) / (y.T.mm(s))) * s
        return hess_0 * q

    def lbfgs_direction(self, s_y_pairs, grad_k):
        if len(s_y_pairs) == 0:
            hess = torch.zeros((self.n_features+1,self.n_features+1))
            hess.diagonal().add_(1)
            return hess.mm(grad_k)
        s_prev, y_prev = s_y_pairs[-1]
        hess_0 = y_prev.T.mm(s_prev) / (y_prev.T.mm(y_prev))
        hess_0 = torch.nan_to_num(hess_0) 
        return self.lbfgs_2_recursion(grad_k, copy.deepcopy(s_y_pairs), hess_0)

    def project(self, u, v):
        return u.masked_fill(u.sign() != v.sign(), 0)

    def glance(self, current_w, p, pvl, lr, eta):
        updated_w = self.project(current_w + lr*pvl*p, eta)
        return updated_w

    def backtracking(self, X, y, curr_w, p, pvl, lr, eta, curr_loss, neg_pseudo_grad, thres, decay=0.95, maxiter=500):
        for n_iter in range(1, maxiter + 1):
            updated_w = self.glance(curr_w, p, pvl, lr, eta)
            updated_loss = self.BCE(self.sigmoid(X.mm(updated_w)), y) + (thres * updated_w).norm(p=1)
            if updated_loss <= curr_loss + self.gamma*neg_pseudo_grad.T.mm(updated_w - curr_w):
                break
            lr = lr * decay
        else:
            warnings.warn('line search did not converge.')
        return updated_w, n_iter

    def fit(self, X_input, Y, prevalence):
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X_input : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            The normalized or transformed data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : {array-like, list} of shape (n_samples,)
            The label list relative to X.
        prevalence : array of shape (n_features,)
            The prevalence vector relative to each feature in X.
        echo : bool
            If True, will print the final training result.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        X, Xt, y, prevalence, weight = self.initialize(X_input, Y, prevalence)
        thres = self.lmbd/(prevalence + 1e-8)
        s_y_pairs = []
        mini_iter = 0; mini_loss = 1e+10; mini_diff = 1e+10
        for k in range(self.max_iter):
            prev_w = weight
            grad = self.logistic_gradient(X, Xt, y, weight)
            vk = -self.pseudo_grad(weight, grad, thres)
            dk = self.lbfgs_direction(s_y_pairs, vk)
            pk = self.project(dk, vk)
            etak = torch.where(weight == 0, vk.sign(), weight.sign())
            curr_loss = self.BCE(self.sigmoid(X.mm(weight)), y) + (thres * weight).norm(p=1)
            weight, btsiter = self.backtracking(X, y, weight, pk, prevalence, self.alpha, etak, curr_loss, vk, thres)
            diff_w = torch.norm(weight - prev_w, p=2)
            
            #updating
            sk = weight - prev_w
            yk = self.logistic_gradient(X, Xt, y, weight) - grad
            s_y_pairs.append((sk, yk))
            if len(s_y_pairs) > self.memory:
                s_y_pairs = s_y_pairs[-self.memory:]
            
            if mini_loss > curr_loss:
                mini_iter = k+1; mini_loss = curr_loss; mini_diff = diff_w; best_w = weight
      
            if diff_w <= self.tol:
                #print('QWL-QN convergence')
                #print('minimum epoch = ', mini_iter, '; minimum lost = ', mini_loss, '; diff weight = ', mini_diff)
                break
                
            if yk.norm(p=1) == 0:
                #print('Hessian initialization fail, gradient diff = 0')
                #print('minimum epoch = ', mini_iter, '; minimum lost = ', mini_loss, '; diff weight = ', mini_diff)
                break
                
            if yk.T.mm(yk) == 0:
                #print('Hessian initialization fail, lower H0 = 0')
                #print('minimum epoch = ', mini_iter, '; minimum lost = ', mini_loss, '; diff weight = ', mini_diff)
                break
        #else:
        #    print('QWL-QN did not convergence')

        if self.UsingDevice == 'cuda':
            best_w = best_w.cpu().numpy().reshape(-1)
        else:
            best_w = best_w.numpy().reshape(-1)
        
        self.n_iter_ = mini_iter; self.loss_ = mini_loss.item(); self.convergence_ = mini_diff.item()

        #if self.echo:
        #    print('minimum epoch = ', self.n_iter_, '; minimum lost = ', self.loss_, '; diff weight = ', self.convergence_)

        self.w = best_w[0:self.n_features]; self.b = best_w[self.n_features]
        self.feature_set = np.where(self.w != 0, 1, 0)
        weight_abs = np.abs(self.w)
        self.feature_sort= np.argsort(-weight_abs)

    def score(self, X, y):
        """
        Goodness of fit estimation.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array of shape (n_samples,)
        
        Returns
        -------
        score : dictionary of measurement (AUC, AUPR, MCC, precision, recall)
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        y_true = self.get_y_array(y)
        auroc = roc_auc_score(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        aupr = metrics.auc(recall, precision)
        mcc = matthews_corrcoef(y_true, y_pred)
        return {"AUC" : auroc, "AUPR" : aupr, "MCC" : mcc, "Precision" : precision, "Recall" : recall}
    
    def predict_proba(self, X):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        Returns
        -------
        pred_proba : array of shape (n_samples,)
            Returns the probability of the sample for binary class in the model,
            where classes are show in ``self.classes_``.
        """
        z = np.exp(-(X.dot(self.w)+self.b))
        pred_proba = 1 / (1 + z)
        return pred_proba
    
    def predict(self, X):
        """
        Prediction.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        Returns
        -------
        pred : array of shape (n_samples,)
            Returns the prediction result for binary classifier,
            where classes are show in ``self.classes_``.
        """
        z = np.exp(-(X.dot(self.w)+self.b))
        pred_proba = 1 / (1 + z)
        pred = np.where(pred_proba > 0.5, 1, 0)
        return pred
    
    def get_y_array(self, label_list):
        """
        Get the corresponding label array in this model.
        
        Parameters
        ----------
        label_list : list of shape (n_samples)
        
        Returns
        -------
        y : array of shape (n_samples,)
        """
        return  np.array([self.classes_[yi] for yi in label_list])


def get_prevalence(X_raw, sample_idx):
    """
    Get the feature prevalence vector according to the given sample set.
        
    Parameters
    ----------
    X_raw : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            The original, count data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    sample_idx : {array-like} of shape (n_samples,)
            The label list relative to X.
            
    Returns
    -------
    prevalence vector : array of shape (n_features,)
    """
    if type(X_raw) is pd.DataFrame:
        X = X_raw.to_numpy()
        X = X[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
    elif type(X_raw) is np.ndarray:
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
    elif isspmatrix_csr(X_raw) or isspmatrix_csc(X_raw) or isspmatrix_coo(X_raw):
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
        pvl_vecter = pvl_vecter.tolist()[0]
        pvl_vecter = np.array(pvl_vecter)
    else :
        raise ValueError("Unrecognizable data types")
    return pvl_vecter

def auto_scale(X_input, X_raw, Y, step=50, device='cpu', training_echo=False,
               max_iter=100, tol=1e-4, alpha=0.1, gamma=0.1): 
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This solver needs samples of at only 2 classe.")
    
    n_samples_i, n_features_i = X_input.shape
    n_samples_r, n_features_r = X_raw.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])
        
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    pvl = get_prevalence(X_raw, np.arange(X_raw.shape[0]))
    
    exam_range = [1/(10**i) for i in np.arange(10,-1,-1)]
    select_number = []
    for lmbd in exam_range:
        exam_res = ADlasso2(lmbd=lmbd, device=device, echo=training_echo,
                                 max_iter=max_iter, tol=tol, alpha=alpha, gamma=gamma)
        exam_res.fit(X_input, y, pvl)
        select_number.append(np.sum(exam_res.feature_set))
    upper  = np.nan
    for i in range(len(exam_range)):
        if np.isnan(upper):
            if select_number[i] < n_features_i*0.9:
                upper  = exam_range[i]
        if select_number[i] < 10:
            lower  = exam_range[i]
            break
    return np.linspace(np.log(upper), np.log(lower), step)

def lambda_tuning(X_input, X_raw, Y, lmbdrange, k_fold, outdir, device='cpu', training_echo=False,
                  max_iter=100, tol=1e-4, alpha=0.1, gamma=0.1):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This procedure allows only 2 classes.")
    
    n_samples_i, n_features_i = X_input.shape
    n_samples_r, n_features_r = X_raw.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    if os.path.exists(outdir) == False :
        os.mkdir(outdir)
    os.chdir(outdir)
    
    if type(X_input) is pd.DataFrame:
        X_input = X_input.to_numpy()
    if type(X_raw) is pd.DataFrame:
        X_raw = X_raw.to_numpy()
    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    prevalence = get_prevalence(X_raw, np.arange(X_raw.shape[0]))
    n_lambda = len(lmbdrange)
    Z = np.zeros((n_lambda, k_fold, n_features_i), dtype=np.int8)
    
    metrics = ['Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC']
    metrics_dict = dict()
    for m in metrics :
        if m == 'pairwiseMCC' :
            metrics_dict[m] = np.zeros((n_lambda, int(k_fold*(k_fold-1)/2)))
        else : 
            metrics_dict[m] = np.zeros((n_lambda, k_fold))
    
    for i in range(n_lambda):
        start = time.time()
        kfold = StratifiedKFold(n_splits=k_fold, shuffle=True)
        kcount = 0
        for train_ix, test_ix in kfold.split(X_input, y):
            train_X, test_X = X_input[train_ix], X_input[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            train_pvl = get_prevalence(X_raw, train_ix)
            lambd = lmbdrange[i]
            examined_lambda = ADlasso2(lmbd = lambd, device = device, echo = training_echo)
                                            #max_iter=max_iter, tol=tol, alpha=alpha, gamma=gamma)
            examined_lambda.fit(train_X, train_y, train_pvl)
            selected_set = examined_lambda.feature_set
            Z[i, kcount, :] = selected_set
            metrics_dict['loss_history'][i,kcount] = examined_lambda.loss_
            metrics_dict['error_history'][i,kcount] = examined_lambda.convergence_
            if np.sum(selected_set) > 1:
                metrics_dict['Feature_number'][i,kcount] = np.sum(selected_set)
                metrics_dict['Percentage'][i,kcount] = np.sum(selected_set)/n_features_i                
                metrics_dict['Prevalence'][i,kcount] = np.median(prevalence[selected_set != 0])
                norm_LR = LogisticRegression(penalty='none')
                perf = evaluation(train_X, train_y, test_X, test_y, examined_lambda, norm_LR)
                metrics_dict['AUC'][i,kcount] = perf['AUC']
                metrics_dict['AUPR'][i,kcount] = perf['AUPR']
                metrics_dict['MCC'][i,kcount] = perf['MCC']
            else:
                metrics_dict['Feature_number'][i,kcount] = 0
                metrics_dict['Percentage'][i,kcount] = 0
                metrics_dict['Prevalence'][i,kcount] = 0
                metrics_dict['AUC'][i,kcount] = 0; 
                metrics_dict['AUPR'][i,kcount] = 0
                metrics_dict['MCC'][i,kcount] = -1
            kcount += 1
        metrics_dict['pairwiseMCC'][i] = stability_measurement(Z[i])
        end = time.time()
        print('lambda is : {lmb}, cost : {tm} min'.format(lmb = lambd, tm = round((end - start)/60, 3)))
    
    metrics_dict['log_lambda_range'] = np.log(lmbdrange)
    for m in list(metrics_dict.keys()) :
        metric_out = str(m) + '.dat'
        np.savetxt(metric_out, metrics_dict[m])
    return metrics_dict

def stability_measurement(feature_set_collection) :
    """
    Calculates Matthews correlation coefficient to estimate the stability of the selected feature set at different runs.
    
    Parameters
    ----------
    feature_set_collection : list of shape (n_run, n_features)
            The original, count data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
    Returns
    -------
    MCC result : array of shape (C(n_run,2),)
    
    References
    ----------
    Jiang, L., Haiminen, N., Carrieri, A. P., Huang, S., Vázquez‐Baeza, Y., Parida, L., ... & Natarajan, L. (2021).
    Utilizing stability criteria in choosing feature selection methods yields reproducible results in microbiome data.
    Biometrics.
    """
    n_selected = feature_set_collection.shape[0]
    i = 0; MCC = []
    while i < n_selected :
        for k in range(i+1, n_selected) :
            mcc_ = matthews_corrcoef(feature_set_collection[i,:], feature_set_collection[k,:])
            MCC.append(mcc_)
        i += 1
    return MCC

def evaluation(x_train, y_train, x_test, y_test, AD_object, classifier):
    """
    Examine the Goodness of selected feature set by user-specified classifier
    
    Parameters
    ----------
    x_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    y_train : array-like of shape (n_samples,)
            Target vector relative to X.
    x_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    y_test : array-like of shape (n_samples,)
            Testing vector relative to X.
    AD_object : A training ADlasso class
    classifier : A scikit-learn classifier.
            A parameterized scikit-learn estimators for examine the performance of feature set selected by ADlasso.
    
    Returns
    -------
    dict : a dict of performance measurement {AUC, AUPR, MCC}
    """
    x_subtrain = x_train[:, AD_object.feature_set != 0]
    x_subtest = x_test[:, AD_object.feature_set != 0]
    classifier.fit(x_subtrain, y_train)
    y_pred_proba = classifier.predict_proba(x_subtest)
    y_pred = classifier.predict(x_subtest)
    auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    aupr = metrics.auc(recall, precision)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    return {"AUC" : auroc, "AUPR" : aupr, "MCC" : mcc}

def get_tuning_result(result_path):
    os.chdir(result_path)
    measurement = ['Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC', 'log_lambda_range']
    result_dict = dict()
    for m in measurement:
        measurement_in = str(m) + '.dat'
        if measurement_in not in os.listdir():
            raise ValueError("No storage results.")
        res = np.loadtxt(measurement_in, dtype=float)
        result_dict[m] = res
    return result_dict

def lambda_tuning_viz(result_dict, metric, savepath=None, fig_width=8, fig_height=4):
    lmbd_range = np.exp(result_dict['log_lambda_range'])
    metrics_recode = result_dict[metric]
    pvl_recode = result_dict['Prevalence']
    m_mean = np.mean(metrics_recode, 1); m_err = np.std(metrics_recode, 1)
    pvl_mean = np.mean(pvl_recode, 1); pvl_err = np.std(pvl_recode, 1)
    pvl_mean = pvl_mean * 100 

    fig, ax1 = plt.subplots(figsize = (fig_width,fig_height))
    ax2 = ax1.twinx()
    ln1 = ax1.errorbar(lmbd_range, m_mean, yerr=m_err, marker='o', c = 'b', linestyle='--', label = metric); ax1.legend(loc='upper left')
    ln2 = ax2.errorbar(lmbd_range, pvl_mean, yerr=pvl_err, marker='o', c = 'r', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set_xlabel("lambda"); ax1.set_ylabel(metric); ax2.set_ylabel("Prevalence (%)")
    ax1.set(xscale="log")
    if metric in ['Feature_number', 'loss_history', 'error_history']:
        ax1.set(yscale="log")
    if savepath:
        plt.savefig(savepath,dpi =300)
    return fig

def lambda_decision(result_dict, k, savepath=None, fig_width=8, fig_height=4):
    lmbd_range = np.exp(result_dict['log_lambda_range'])
    loss_recode = result_dict['loss_history']
    pvl_recode = result_dict['Prevalence']
    loss_mean = np.mean(loss_recode, 1); loss_err = np.std(loss_recode, 1)
    pvl_mean = np.mean(pvl_recode, 1); pvl_err = np.std(pvl_recode, 1)
    pvl_mean = pvl_mean * 100 
        
    xs = np.log(lmbd_range); ys = np.log(loss_mean)
    fig, ax1 = plt.subplots(figsize = (fig_width,fig_height))
    ax2 = ax1.twinx()
    dys = np.gradient(ys, xs)
    rgr = DecisionTreeRegressor(max_leaf_nodes = k).fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
    dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()
    ys_sl = np.ones(len(xs)) * np.nan
    for y in np.unique(dys_dt):
        msk = dys_dt == y
        lin_reg = LinearRegression()
        lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
        ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
        ax1.plot(np.exp([xs[msk][0], xs[msk][-1]]), np.exp([ys_sl[msk][0], ys_sl[msk][-1]]), color='r', zorder=5, linewidth = 2)
    
    
    segth = []; count = 0
    for i in range(len(dys_dt)):
        if dys_dt[i] not in segth:
            segth.append(dys_dt[i])
            count += 1
        if count == 1:
            selected_lambda = xs[i]
    
    ax1.errorbar(lmbd_range, loss_mean, marker='o', c='#33CCFF', linestyle='--', label ='BCE loss'); ax1.legend(loc='upper left')
    ax2.errorbar(lmbd_range, pvl_mean, marker='o', c='#FFAA33', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set(xscale="log"); ax1.set(yscale="log")
    ax1.set_xlabel("lambda"); ax1.set_ylabel("loss"); ax2.set_ylabel("Prevalence (%)")
    selected_lambda = np.exp(selected_lambda); plt.axvline(x=selected_lambda, color = 'black', linestyle=':')
    plt.show()
    if savepath:
        plt.savefig(savepath,dpi =300)
    return selected_lambda, fig

def scipySparseVars(a, axis=None):
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def featureProperty(X, y, AD_object):
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    y = AD_object.get_y_array(y)
    
    if type(X) is np.ndarray:
        nonzeroSamples = X.sum(axis=1) != 0
        X_ = X[nonzeroSamples,:]
        y_ = y[nonzeroSamples]
        
        XF = X_/X_.sum(axis=1)[:,None]
        RA = XF.mean(axis = 0)
        Var = X_.std(axis = 0)**2

    elif isspmatrix_csr(X) or isspmatrix_csc(X) or isspmatrix_coo(X):
        rowSum = X.sum(axis=1).reshape(-1)
        nonzeroSamples = np.array(rowSum != 0)[0]
        X_ = X[nonzeroSamples,:]
        y_ = y[nonzeroSamples]
        RA = X_.mean(axis = 0); RA = np.array(RA).reshape(-1) #XF
        Var = scipySparseVars(X_, axis = 0); Var = np.array(Var).reshape(-1)
        
    else :
        raise ValueError("X is unrecognizable data types")
        
    selection = ["ADlasso" if i == 1 else "No selected" for i in AD_object.feature_set]
    classIdx = {v: k for k, v in AD_object.classes_.items()}
    class0Idx = np.array([i for i, la in enumerate(y_) if la == 0])
    class1Idx = np.array([i for i, la in enumerate(y_) if la == 1])
    wholePvl = get_prevalence(X_, np.arange(X_.shape[0]))
    class0Pvl = get_prevalence(X_, class0Idx)
    class1Pvl = get_prevalence(X_, class1Idx)
    C0head = 'prevalence_' + str(list(AD_object.classes_.keys())[0])
    C1head = 'prevalence_' + str(list(AD_object.classes_.keys())[1])
    plotdf = pd.DataFrame({'meanAbundance' : RA,'Variance' : Var, 'select' : selection, 'prevalence' : wholePvl, C0head : class0Pvl, C1head : class1Pvl})
    return plotdf



# %% 
url_1 = 'https://raw.githubusercontent.com/YinchengChen23/ADlasso/main/data/crc_zeller/ASV_vst.txt'
url_2 = 'https://raw.githubusercontent.com/YinchengChen23/ADlasso/main/data/crc_zeller/ASV_table.txt'
url_3 = 'https://raw.githubusercontent.com/YinchengChen23/ADlasso/main/data/crc_zeller/metadata.txt'
Data = pd.read_csv(url_1, sep = "\t"); Data = Data.T           # Variance-stabilizing transformation was conducted by DESeq2
Data_std = scipy.stats.zscore(Data, axis=0, ddof=0)            # we using z-normalization data as input-data
RawData = pd.read_csv(url_2, sep = "\t"); RawData = RawData.T  # Raw count data, was used as an assessment of prevalence
Cohort = pd.read_csv(url_3, sep = "\t")                        # Metadata
Label = Cohort['Class'].tolist()

print('This data contains', Data_std.shape[0], 'samples and', Data_std.shape[1], 'features')
print(Label[0:10], np.unique(Label))

# %%
pvl = get_prevalence(RawData, np.arange(RawData.shape[0]))
res = ADlasso2(lmbd = 1e-5, alpha=0.9, echo= True)
start = time.time()
res.fit(Data_std, Label, pvl)  # .fit(X, y, prevalence)
# minimum epoch =  9999 ; minimum lost =  6.27363842795603e-05 ; diff weight =  0.002454951871186495
end = time.time()

print('median of selected prevalence :',np.median([pvl[i]  for i, w in enumerate(res.feature_set) if w != 0]))
# median of prevalence : 0.3023255813953488

print('total selected feature :',np.sum(res.feature_set))
# total selected feature : 480

print("Total cost：%f sec" % (end - start))
# Total cost：0.231663 sec

# %%
import matplotlib.pyplot as plt

selected_pvl = [pvl[i]  for i, w in enumerate(res.feature_set) if w != 0]
plt.hist(selected_pvl)

# %%
# importance ranking
for ix in res.feature_sort:  # feature_sort: The list of sorted features indices by importance.
    print(Data_std.columns[ix], res.w[ix])

# %%
# We also can do the split training and testing with ADlasso
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(Data_std, Label, test_size=0.2)
train_ix = np.array([ix for ix, sample in enumerate(Data_std.index) if sample in train_X.index])
pvl = get_prevalence(RawData, train_ix)  # sample_idx = train_ix
res = ADlasso2(lmbd = 1e-6, alpha=0.9, echo= True)
res.fit(train_X, train_y, pvl)

print('median of prevalence :',np.median([pvl[i]  for i, w in enumerate(res.feature_set) if w != 0]))
print('total selected feature :',np.sum(res.feature_set))

# %%
metrics_dict = res.score(test_X, test_y)  # no attribute 'score'
print(metrics_dict)

fig, ax = plt.subplots()
ax.plot(metrics_dict['Recall'], metrics_dict['Precision'], color='purple')
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.show()
# You may notice that the accuracy of ADlasso is not very high as
# it is a feature selector rather than a classifier.

# %%
### Lambda tuning
log_lmbd_range = auto_scale(Data_std, RawData, Label, step=50)

lmbd_range = np.exp(log_lmbd_range)
print(lmbd_range)
# %%
# k_fold: split data into k parts for cross validation. 
k_fold = 5
outPath_dataSet = './LmbdTuning'
result_dict =lambda_tuning(Data_std, RawData, Label, lmbd_range, k_fold, outPath_dataSet)  # error
# Use GPU
# result_dict = lambda_tuning(Data_std, RawData, Label, lmbd_range, k_fold, outPath_dataSet ,device='cuda')

# %%
# Lambda tuning visualization
Fig = lambda_tuning_viz(result_dict, 'Feature_number')
# Fig = lambda_tuning_viz(result_dict, 'AUC')
# Fig = lambda_tuning_viz(result_dict, 'loss_history')
# Fig = lambda_tuning_viz(result_dict, 'error_history')

# %%
# Find optimal lambda
opt_lmbd, fig = lambda_decision(result_dict, 5)
print(opt_lmbd)

# %%
# Use optimal lambda in ADlasso()
pvl = get_prevalence(RawData, np.arange(RawData.shape[0]))
opt_res = ADlasso2(lmbd = opt_lmbd, alpha=0.9, echo= True)
start = time.time()
opt_res.fit(Data_std, Label, pvl)
# minimum epoch =  9999 ; minimum lost =  0.00022168313444126397 ; diff weight =  0.002862341469153762
end = time.time()

print('median of prevalence :',np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0]))
# median of prevalence : 0.5271317829457365
print('total selected feature :',np.sum(opt_res.feature_set))
print("Total cost：%f sec" % (end - start))

# %%
# Selection profile
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

LR = LogisticRegression()
SVM = SVC(probability = True)
RF = RandomForestClassifier()

X_raw = Data_std.to_numpy()
y_array = opt_res.get_y_array(Label)  # Get the corresponding binary array in this model.

train_X, test_X, train_y, test_y = train_test_split(X_raw, y_array, test_size=0.2)

# Examine the Goodness of selected feature set by user-specified classifier
# Logistic Regression
perf = evaluation(train_X, train_y, test_X, test_y, opt_res, LR)
print('Logistic Regression :', perf)

# SVM
perf = evaluation(train_X, train_y, test_X, test_y, opt_res, SVM)
print('SVM :', perf)

# Random Forest
perf = evaluation(train_X, train_y, test_X, test_y, opt_res, RF)
print('RandomForest :', perf)

# %%
Prop = featureProperty(RawData, Label, opt_res)  # get the properties of each features
Prop['featureID'] = Data_std.columns
Prop


# %% lambda decision breakdown
lmbd_range = np.exp(result_dict['log_lambda_range'])
loss_recode = result_dict['loss_history']
pvl_recode = result_dict['Prevalence']
loss_mean = np.mean(loss_recode, 1); loss_err = np.std(loss_recode, 1)
pvl_mean = np.mean(pvl_recode, 1); pvl_err = np.std(pvl_recode, 1)
pvl_mean = pvl_mean * 100 

# %%
xs = np.log(lmbd_range); ys = np.log(loss_mean)
fig, ax1 = plt.subplots(figsize = (8,4))
ax2 = ax1.twinx()
dys = np.gradient(ys, xs)
rgr = DecisionTreeRegressor(max_leaf_nodes = 5).fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()
ys_sl = np.ones(len(xs)) * np.nan

for y in np.unique(dys_dt):
    msk = dys_dt == y
    lin_reg = LinearRegression()
    lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
    ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
    ax1.plot(np.exp([xs[msk][0], xs[msk][-1]]), np.exp([ys_sl[msk][0], ys_sl[msk][-1]]), color='r', zorder=5, linewidth = 2)


segth = []; count = 0
for i in range(len(dys_dt)):
    if dys_dt[i] not in segth:
        segth.append(dys_dt[i])
        count += 1
    if count == 1:
        selected_lambda = xs[i]

ax1.errorbar(lmbd_range, loss_mean, marker='o', c='#33CCFF', linestyle='--', label ='BCE loss'); ax1.legend(loc='upper left')
ax2.errorbar(lmbd_range, pvl_mean, marker='o', c='#FFAA33', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
ax1.set(xscale="log"); ax1.set(yscale="log")
ax1.set_xlabel("lambda"); ax1.set_ylabel("loss"); ax2.set_ylabel("Prevalence (%)")
selected_lambda = np.exp(selected_lambda); plt.axvline(x=selected_lambda, color = 'black', linestyle=':')
plt.show()
# %%
