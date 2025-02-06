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
                norm_LR = LogisticRegression(penalty=None)
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