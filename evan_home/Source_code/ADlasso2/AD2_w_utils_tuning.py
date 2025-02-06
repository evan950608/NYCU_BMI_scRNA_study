import numpy as np
import pandas as pd
import time
import os
import scipy
from scipy.sparse import *
import copy
from collections import deque
import warnings
import random
import multiprocessing as mp

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class ADlasso2():
    def __init__(self, lmbd=1e-5, max_iter=1000, tol=1e-4, alpha=0.01, gamma=0.1, device='cpu', echo=True):
        # original alpha = 0.1
        super().__init__()
        self.lmbd = lmbd
        self.max_iter = int(max_iter)
        self.tol = tol
        self.alpha = alpha
        self.loss_history = {}
        self.weight_diff_history = {}
        self.stop_iter = 0
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
        bias = torch.ones(self.n_samples).reshape(-1, 1)
        X_data = torch.cat((X, bias), 1).to(self.UsingDevice)
        return X_data, X_data.T
    
    def coo2sparse_tensor(self, X):
        self.n_samples, self.n_features = X.shape
        bias = np.ones(self.n_samples).reshape(-1, 1)
        X = scipy.sparse.hstack((X, bias))
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
#         pvl = np.append(pvl,1)     # append 1 for representing bias prevalence
        pvl = np.append(pvl, np.min(pvl))  # append the minimum prevalence for representing bias prevalence
        pvl = torch.from_numpy(pvl).float().reshape(-1, 1).to(self.UsingDevice)
        
        weight = torch.zeros(self.n_features+1, requires_grad = False).reshape(-1, 1).to(self.UsingDevice)
        return X, XT, y, pvl, weight
    
    def logistic_gradient(self, X, XT, y, w):
        resid = self.sigmoid(X.mm(w)) - y
        gradient = XT.mm(resid)/self.n_samples
        return gradient
    
    def pseudo_grad(self, w, grad, thres):
        grad_r = thres * w.sign()
        grad_right = grad + torch.where(w != 0, grad_r, thres)
        grad_left = grad + torch.where(w != 0, grad_r, -thres)
        grad_pseudo = torch.zeros_like(w)
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

    def lbfgs_direction(self, s_y_pairs, grad):
        if len(s_y_pairs) == 0:
            ## GPU edit
            # hess = torch.zeros((self.n_features+1,self.n_features+1)).to(self.UsingDevice)
            # hess.diagonal().add_(1)
            # return hess.mm(grad)
            return grad
        s_prev, y_prev = s_y_pairs[-1]
        hess_0 = y_prev.T.mm(s_prev) / (y_prev.T.mm(y_prev))
        hess_0 = torch.nan_to_num(hess_0)
        return self.lbfgs_2_recursion(grad, copy.deepcopy(s_y_pairs), hess_0)

    def project(self, u, v):
        return u.masked_fill(u.sign() != v.sign(), 0)

    def glance(self, current_w, p, lr, eta):
        updated_w = self.project(current_w.add(p, alpha=lr), eta)
        return updated_w

    def backtracking(self, X, y, curr_w, p, lr, eta, curr_loss, neg_pseudo_grad, thres, decay=0.95, maxiter=100):
        for n_iter in range(1, maxiter + 1):
            updated_w = self.glance(curr_w, p, lr, eta)
            updated_loss = self.BCE(self.sigmoid(X.mm(updated_w)), y) + (thres * updated_w).norm(p=1)
            if updated_loss <= curr_loss + self.gamma*neg_pseudo_grad.T.mm(updated_w - curr_w):
                break
            lr = lr * decay
        else:
            warnings.warn('line search did not converge.')
        return updated_w, n_iter

    
    def fit(self, X_input, Y, prevalence):
        # yincheng 20230804 edit .fit()
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X_input : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            The normalized or transformed data, where n_samples is the number of samples and
            n_features is the number of features.
        y : {array-like, list} of shape (n_samples,)
            The label list relative to X.
        prevalence : array of shape (n_features,)
            The prevalence vector relative to each feature in X.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        # print('Fitting ADlasso2, learning rate alpha = ', self.alpha)
        status = ""

        X, Xt, y, prevalence, weight = self.initialize(X_input, Y, prevalence)
        thres = self.lmbd/(prevalence + 1e-8)
        s_y_pairs = []
        mini_iter = 0; mini_loss = 1e+10; mini_diff = 1e+10
        for k in range(self.max_iter):
            # print('iter:', k)
            prev_weight = weight
            grad = self.logistic_gradient(X, Xt, y, weight)
            vk = -self.pseudo_grad(weight, grad, thres)
            dk = self.lbfgs_direction(s_y_pairs, vk)
            pk = self.project(dk, vk)
            etak = torch.where(weight == 0, vk.sign(), weight.sign())
            curr_loss = self.BCE(self.sigmoid(X.mm(weight)), y) + (thres * weight).norm(p=1)
            weight, btsiter = self.backtracking(X, y, weight, pk, self.alpha, etak, curr_loss, vk, thres)
            diff_w = torch.norm(weight - prev_weight, p=2)
            
            #updating
            sk = weight - prev_weight
            yk = self.logistic_gradient(X, Xt, y, weight) - grad
            s_y_pairs.append((sk, yk))
            if len(s_y_pairs) > self.memory:
                s_y_pairs = s_y_pairs[-self.memory:]
            
            if mini_loss > curr_loss:
                # print('set best weight')
                mini_iter = k+1; mini_loss = curr_loss; mini_diff = diff_w; best_w = weight
    
            # document loss and weight difference by iteration
            self.loss_history[k] = curr_loss.item()
            self.weight_diff_history[k] = diff_w.item()

            # Convengence with weight difference
            if abs(diff_w) <= self.tol and k > 100:
                self.stop_iter = k
                status = 'QWL-QN convergence'
                print(f'At iteration {k},', status)
                break
                
            if yk.norm(p=1) == 0:
                status = 'Hessian initialization fail, gradient diff = 0'
                print(status)
                break
                
            if yk.T.mm(yk) == 0:
                status = 'Hessian initialization fail, lower H0 = 0'
                print(status)
                break
                
            if yk.T.mm(sk) == 0:
                status = 'Hessian approximation fail, yTs = 0'
                print(status)
                break
            if yk.T.mm(sk)/(yk.T.mm(yk)) > 1e+30:
                status = 'H0 so large'
                print(status)
                break
            # if torch.sum(torch.abs(weight)) == 0:
            #     status = 'All weight is zero'
            #     print(status)
            #     break
        else:
            status = 'QWL-QN did not convergence'
            print(status)
                
                
        if self.UsingDevice == 'cuda':
            best_w = best_w.cpu().numpy().reshape(-1)
        else:
            best_w = best_w.numpy().reshape(-1)
        
        self.n_iter_ = mini_iter; self.loss_ = mini_loss.item(); self.convergence_ = mini_diff.item()

        if self.echo:
            print(status)
            print('minimum epoch = ', self.n_iter_, '; minimum lost = ', self.loss_, '; diff weight = ', self.convergence_)

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
    
    def writeList(self, outpath=None, featureNameList=None):
        """
        Export the selection result.
        
        Parameters
        ----------
        outpath : str
                  Absolute path for output file.
        featureNameList : list or array of shape (n_features,)
                  A list contains feature name.
        
        Returns
        -------
        File : first column : Name or index of selected feature.
               second column : Weight of each feauture.
               third column : Tendency of each feature.
        """
        if featureNameList is not None:
            if len(self.feature_set) != len(featureNameList):
                raise ValueError("Found input feature list with inconsistent numbers of features: %r" % [len(self.feature_set), len(featureNameList)])

        dirpath = os.path.dirname(outpath)
        if not dirpath:
            raise ValueError("The folder you assigned does not exist.")

        classes = {v: k for k, v in self.classes_.items()}
        w = open(outpath,'w')
        for ix, wi in enumerate(self.w):
            if wi != 0:
                featureID = featureNameList[ix] if featureNameList is not None else ix
                tendency = classes[0] if wi < 0 else classes[1]
                w.writelines(str(featureID) + "\t" + str(wi) + "\t" + str(tendency) + '\n')
        w.close()


def get_prevalence(X_raw, sample_idx):
    # print('Getting prevalance')
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
        # Convert a dataframe to numpy array
        X = X_raw.to_numpy()
        X = X[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
    elif type(X_raw) is np.ndarray:
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]  # calculates the proportion of non-zero elements in each column
    elif isspmatrix_csr(X_raw) or isspmatrix_csc(X_raw) or isspmatrix_coo(X_raw):
        # print('isspmatrix_csr prevalence')
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
        pvl_vecter = pvl_vecter.tolist()[0]
        pvl_vecter = np.array(pvl_vecter)
    else :
        raise ValueError("Unrecognizable data types")
    return pvl_vecter

def auto_scale(X_norm, Y, step=50, device='cpu', training_echo=False,
               max_iter=1000, tol=1e-4, alpha=0.01, gamma=0.1): 
    # Combine X_input and X_raw into X_norm
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This solver needs samples of at only 2 classe.")
    
    n_samples_i, n_features_i = X_norm.shape
    n_samples_r, n_features_r = X_norm.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        # raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [n_samples_i, len(Y)])
        
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    # Modify sample_index for prevalence calculation
    celltype_indices = [idx for idx, label in enumerate(Y) if label == 1]
    # pvl = get_prevalence(X_norm, np.arange(X_norm.shape[0]))
    pvl = get_prevalence(X_norm, celltype_indices)
    
    exam_range = [1/(10**i) for i in np.arange(10,-1,-1)]  # 1e-10 to 1e-1
    select_number = []
    for lmbd in exam_range:
        print('Lambda:', lmbd)
        exam_res = ADlasso2(lmbd=lmbd, device=device, echo=training_echo,
                                 max_iter=max_iter, tol=tol, alpha=alpha, gamma=gamma)
        exam_res.fit(X_norm, y, pvl)
        select_number.append(np.sum(exam_res.feature_set))
    
    upper  = np.nan
    lower = exam_range[-1]  # initialize as the last element in exam_range
    for i in range(len(exam_range)):
        if np.isnan(upper):
            if select_number[i] < n_features_i*0.9:
                upper  = exam_range[i]
        if select_number[i] < 10:
            lower  = exam_range[i]
            break

    return np.linspace(np.log(upper), np.log(lower), step)  # evenly spaced numbers on a log scale


def lambda_tuning_evan(X_norm, Y, lmbdrange, device='cpu', training_echo=False,
                  max_iter=1000, tol=1e-4, alpha=0.01, gamma=0.1):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This procedure allows only 2 classes.")
    
    n_samples_i, n_features_i = X_norm.shape
    n_samples_r, n_features_r = X_norm.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [n_samples_i, len(Y)])
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    
    if type(X_norm) is pd.DataFrame:
        X_norm = X_norm.to_numpy()

    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    # Local prevalence: prevalence with one cell type
    celltype_indices = [idx for idx, label in enumerate(Y) if label == 1]
    prevalence = get_prevalence(X_norm, celltype_indices)
    # Global prevalence: prevalence with all cells
    # prevalence_global = get_prevalence(X_norm, np.arange(n_samples_i))  # np.arange(X_norm.shape[0])
    n_lambda = len(lmbdrange)
    
    # Remove pairwiseMCC
    metrics = ['Percentage', 'Prevalence', 'Train_prevalence', 'Train_prevalence_global', 
               'Feature_number', 'AUC', 'AUPR', 'MCC', 'Precision', 'F1 score', 'loss_history', 'error_history']
    metrics_dict = dict()
    for m in metrics :
        # metrics_dict[m] = np.zeros((n_lambda, k_fold))
        metrics_dict[m] = np.zeros(n_lambda)
    
    # iteratively test different lambda, ignore the k-fold cross validation
    for i in range(n_lambda):
        start = time.time()
        # Train test split with index
        # train_X, test_X, train_y, test_y = train_test_split(X_input, y, test_size=0.2)
        train_test_index = [True] * int(X_norm.shape[0] * 0.8) + [False] * (X_norm.shape[0] - int(X_norm.shape[0] * 0.8))
        random.shuffle(train_test_index)
        train_index = np.where(train_test_index)[0]
        test_index = np.where(np.logical_not(train_test_index))[0]
        train_X, test_X = X_norm[train_index], X_norm[test_index]
        train_y, test_y = y[train_index], y[test_index]

        # Calculate new prevalence
        # Training local prevalence: Get the prevalence of one cell type in training set
        train_celltype_indices = [idx for idx, label in enumerate(train_y) if label == 1]
        train_pvl = get_prevalence(train_X, train_celltype_indices)
        # this_pvl = get_prevalence(X_raw, celltype_indices)

        # Training global prevalence: Get the prevalence of all cells in training set
        train_pvl_global = get_prevalence(train_X, np.arange(train_X.shape[0]))


        # Create new ADlasso object
        lambd = lmbdrange[i]
        print('Lambda:', lambd)
        examined_lambda = ADlasso2(lmbd = lambd, device = device, echo = training_echo, alpha=alpha, max_iter=max_iter, tol=tol, gamma=gamma)  # set alpha, gamma, max_iter, tol?
        examined_lambda.fit(train_X, train_y, train_pvl)
        selected_set = examined_lambda.feature_set
        metrics_dict['loss_history'][i] = examined_lambda.loss_
        metrics_dict['error_history'][i] = examined_lambda.convergence_
        if np.sum(selected_set) > 1:
            metrics_dict['Feature_number'][i] = np.sum(selected_set)
            metrics_dict['Percentage'][i] = np.sum(selected_set)/n_features_i                
            metrics_dict['Prevalence'][i] = np.median(prevalence[selected_set != 0])  # WHY MEDIAN?
            metrics_dict['Train_prevalence'][i] = np.median(train_pvl[selected_set != 0])
            metrics_dict['Train_prevalence_global'][i] = np.median(train_pvl_global[selected_set != 0])
            # new LR model and evaluation
            norm_LR = LogisticRegression(penalty='none')
            perf = evaluation(train_X, train_y, test_X, test_y, examined_lambda, norm_LR)
            metrics_dict['AUC'][i] = perf['AUC']
            metrics_dict['AUPR'][i] = perf['AUPR']
            metrics_dict['MCC'][i] = perf['MCC']
            metrics_dict['Precision'][i] = perf['Precision']
            # metrics_dict['Precision met'][i] = perf['Precision met']
            metrics_dict['F1 score'][i] = perf['F1 score']
        else:
            metrics_dict['Feature_number'][i] = 0
            metrics_dict['Percentage'][i] = 0
            metrics_dict['Prevalence'][i] = 0
            metrics_dict['AUC'][i] = 0; 
            metrics_dict['AUPR'][i] = 0
            metrics_dict['MCC'][i] = -1
            metrics_dict['Precision'][i] = 0
            metrics_dict['F1 score'][i] = 0

        end = time.time()
        print('lambda is : {lmb}, cost : {tm} min'.format(lmb = lambd, tm = round((end - start)/60, 3)))
        print('==========')
    
    metrics_dict['log_lambda_range'] = np.log(lmbdrange)
    # for m in list(metrics_dict.keys()) :
    #     metric_out = str(m) + '.dat'
    #     np.savetxt(metric_out, metrics_dict[m])

    return metrics_dict

def lambda_tuning_para_ttsplit(X_norm, Y, lmbdrange, device='cpu', training_echo=False,
                                max_iter=1000, tol=1e-4, alpha=0.01, gamma=0.1, n_jobs=1):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This procedure allows only 2 classes.")
    
    n_samples_i, n_features_i = X_norm.shape
    n_samples_r, n_features_r = X_norm.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [n_samples_i, len(Y)])
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    if type(X_norm) is pd.DataFrame:
        X_norm = X_norm.to_numpy()

    # Calculate prevalence
    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    # Local prevalence: prevalence with one cell type
    celltype_indices = [idx for idx, label in enumerate(y) if label == 1]  # from enumerate(Y) change to enumerate(y)
    prevalence = get_prevalence(X_norm, celltype_indices)
    n_lambda = len(lmbdrange)
    
    # Evaluation metrics
    metrics = ['Percentage', 'Prevalence', 'Train_prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 
               'Precision', 'F1 score', 'loss_history', 'error_history']
                # 'Train_prevalence_global'
    metrics_dict = dict()
    for m in metrics :
        metrics_dict[m] = np.zeros(n_lambda)
    

    ### Perform train_test_split outside of test_lambda()
    # Train test split with index
    train_test_index = [True] * int(X_norm.shape[0] * 0.8) + [False] * (X_norm.shape[0] - int(X_norm.shape[0] * 0.8))
    random.shuffle(train_test_index)
    train_index = np.where(train_test_index)[0]
    test_index = np.where(np.logical_not(train_test_index))[0]
    train_X, test_X = X_norm[train_index], X_norm[test_index]
    train_y, test_y = y[train_index], y[test_index]


    # define a lambda test function that can be executed in parallel
    global test_lambda
    def test_lambda(lambd):
        start = time.time()
        # Calculate new prevalence
        # Training local prevalence: Get the prevalence of one cell type in training set
        train_celltype_indices = [idx for idx, label in enumerate(train_y) if label == 1]
        train_pvl = get_prevalence(train_X, train_celltype_indices)
        # Training global prevalence: Get the prevalence of all cells in training set
        # train_pvl_global = get_prevalence(train_X, np.arange(train_X.shape[0]))


        # Create new ADlasso object
        print('Lambda:', lambd)
        examined_lambda = ADlasso2(lmbd = lambd, device = device, echo = training_echo, alpha=alpha, max_iter=max_iter, tol=tol, gamma=gamma)  # set alpha, gamma, max_iter, tol?
        examined_lambda.fit(train_X, train_y, train_pvl)
        selected_set = examined_lambda.feature_set
        loss_history = examined_lambda.loss_
        error_history = examined_lambda.convergence_
        if np.sum(selected_set) > 1:
            feature_number = np.sum(selected_set)
            percentage = np.sum(selected_set)/n_features_i                
            prevalence_ = np.median(prevalence[selected_set != 0])  # WHY MEDIAN?
            train_prevalence = np.median(train_pvl[selected_set != 0])
            # train_prevalence_global = np.median(train_pvl_global[selected_set != 0])
            # new LR model and evaluation
            norm_LR = LogisticRegression(penalty=None)  # penalty='none'
            perf = evaluation(train_X, train_y, test_X, test_y, examined_lambda, norm_LR)
            auc = perf['AUC']
            aupr = perf['AUPR']
            mcc = perf['MCC']
            precision = perf['Precision']
            f1 = perf['F1 score']
        else:
            feature_number = 0
            percentage = 0
            prevalence_ = 0
            auc = 0
            aupr = 0
            mcc = -1
            train_prevalence = 0
            # train_prevalence_global = 0
            precision = f1 = 0

        end = time.time()
        print('lambda is : {lmb}, cost : {tm} min'.format(lmb = lambd, tm = round((end - start)/60, 3)))
        print('==========')
        
        return (percentage, prevalence_, train_prevalence, feature_number, auc, aupr, mcc, precision, f1, loss_history, error_history)
        # return metrics_dict

    # create a pool of worker processes
    with mp.Pool(processes=n_jobs) as pool:
        # map the lambda test function to the lambda range using the pool of workers
        print('*** Start parallel lambda tuning ***')
        results = pool.map(test_lambda, lmbdrange)

    # collect the results from the worker processes
    print('*** Collecting results ***')
    for i, result in enumerate(results):
        metrics_dict['Percentage'][i] = result[0]
        metrics_dict['Prevalence'][i] = result[1]
        metrics_dict['Train_prevalence'][i] = result[2]
        # metrics_dict['Train_prevalence_global'][i] = result[3]
        metrics_dict['Feature_number'][i] = result[3]
        metrics_dict['AUC'][i] = result[4]
        metrics_dict['AUPR'][i] = result[5]
        metrics_dict['MCC'][i] = result[6]
        metrics_dict['Precision'][i] = result[7]  # error here
        metrics_dict['F1 score'][i] = result[8]
        metrics_dict['loss_history'][i] = result[9]
        metrics_dict['error_history'][i] = result[10]

    metrics_dict['log_lambda_range'] = np.log(lmbdrange)

    return metrics_dict


def lambda_tuning_MP(X_norm, Y, lmbdrange, device='cpu', training_echo=False, 
                     max_iter=1000, tol=1e-4, alpha=0.01, gamma=0.1, n_jobs=1):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This procedure allows only 2 classes.")
    
    n_samples_i, n_features_i = X_norm.shape
    n_samples_r, n_features_r = X_norm.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [n_samples_i, len(Y)])
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, ADlasso is running with CPU.")
            device= 'cpu'

    if type(X_norm) is pd.DataFrame:
        X_norm = X_norm.to_numpy()
    
    # Calculate prevalence
    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    # Local prevalence: prevalence with one cell type
    celltype_indices = [idx for idx, label in enumerate(y) if label == 1]  # from enumerate(Y) change to enumerate(y)
    prevalence = get_prevalence(X_norm, celltype_indices)
    n_lambda = len(lmbdrange)

    
    ### Perform train_test_split outside of test_lambda()
    # Train test split with index
    train_test_index = [True] * int(X_norm.shape[0] * 0.8) + [False] * (X_norm.shape[0] - int(X_norm.shape[0] * 0.8))
    random.shuffle(train_test_index)
    train_index = np.where(train_test_index)[0]
    test_index = np.where(np.logical_not(train_test_index))[0]
    train_X, test_X = X_norm[train_index], X_norm[test_index]
    train_y, test_y = y[train_index], y[test_index]

    # test lambda function
    # global test_lambda
    def test_lambda(lambd):
        # Evaluation metrics
        metrics = ['Percentage', 'Prevalence', 'Train_prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 
                'Precision', 'F1 score', 'loss_history', 'error_history']
        metrics_dict = dict()
        for m in metrics :
            metrics_dict[m] = np.zeros(n_lambda)
    

        start = time.time()
        # Calculate new prevalence
        # Training local prevalence: Get the prevalence of one cell type in training set
        train_celltype_indices = [idx for idx, label in enumerate(train_y) if label == 1]
        train_pvl = get_prevalence(train_X, train_celltype_indices)

        # Create new ADlasso object
        print('Lambda:', lambd)
        examined_lambda = ADlasso2(lmbd = lambd, device = device, echo = training_echo, alpha=alpha, max_iter=max_iter, tol=tol, gamma=gamma)  # set alpha, gamma, max_iter, tol?
        examined_lambda.fit(train_X, train_y, train_pvl)
        selected_set = examined_lambda.feature_set
        metrics_dict['loss_history'] = examined_lambda.loss_
        metrics_dict['error_history'] = examined_lambda.convergence_
        if np.sum(selected_set) > 1:
            metrics_dict['Feature_number'] = np.sum(selected_set)  # feature_number
            metrics_dict['Percentage'] = np.sum(selected_set)/n_features_i  # percentage
            metrics_dict['Prevalence'] = np.median(prevalence[selected_set != 0])  # prevalence
            metrics_dict['Train_prevalence'] = np.median(train_pvl[selected_set != 0])  # train_prevalence
            # new LR model and evaluation
            norm_LR = LogisticRegression(penalty=None)  # penalty='none'
            perf = evaluation(train_X, train_y, test_X, test_y, examined_lambda, norm_LR)
            metrics_dict['AUC'] = perf['AUC']
            metrics_dict['AUPR'] = perf['AUPR']
            metrics_dict['MCC'] = perf['MCC']
            metrics_dict['Precision'] = perf['Precision']
            metrics_dict['F1 score'] = perf['F1 score']
        else:
            metrics_dict['Feature_number'] = 0
            metrics_dict['Percentage'] = 0
            metrics_dict['Percentage'] = 0
            metrics_dict['AUC'] = 0
            metrics_dict['AUPR'] = 0
            metrics_dict['MCC'] = -1
            metrics_dict['Train_prevalence'] = 0
            metrics_dict['Precision'] = metrics_dict['F1 score'] = 0

        end = time.time()
        print('lambda is : {lmb}, cost : {tm} min'.format(lmb = lambd, tm = round((end - start)/60, 3)))
        print('==========')
        
        return metrics_dict
    
    # create a pool of worker processes
    pool = mp.Pool(processes=n_jobs)
    results = []
    for lambd in lmbdrange:
        res = pool.apply_async(test_lambda, args=(lambd,))  # res is dict (ApplyAsync object)
        results.append(res)  # result is a list of dict
    
    pool.close()
    pool.join()
    DF_result = pd.DataFrame()
    for res in results:
        tmpDF = pd.DataFrame(res.get(), index=[0])  # res.get() is a dict, get value from res
        DF_result = pd.concat([DF_result, tmpDF], ignore_index=True)  # combine 1-D array tmpDF to DF_result

    return DF_result


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
    f1 = metrics.f1_score(y_test, y_pred)
    precision_met = metrics.precision_score(y_test, y_pred)
    aupr = metrics.auc(recall, precision)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    return {"AUC" : auroc, "AUPR" : aupr, "MCC" : mcc, 'Precision': precision_met, 'F1 score': f1}

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
    m_mean = np.mean(metrics_recode); m_err = np.std(metrics_recode)
    pvl_mean = np.mean(pvl_recode); pvl_err = np.std(pvl_recode)
    # pvl_mean = pvl_mean * 100 

    fig, ax1 = plt.subplots(figsize = (fig_width,fig_height))
    ax2 = ax1.twinx()
    ln1 = ax1.plot(lmbd_range, metrics_recode, marker='o', c = 'b', linestyle='--', label = metric); ax1.legend(loc='upper left')
    ln2 = ax2.plot(lmbd_range, [100*p for p in pvl_recode], marker='o', c = 'r', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set_xlabel("lambda"); ax1.set_ylabel(metric); ax2.set_ylabel("Prevalence (%)")
    ax1.set(xscale="log")
    ax2.set_ylim(0, 100)

    if metric in ['Feature_number', 'loss_history', 'error_history']:
        ax1.set(yscale="log")
    if savepath:
        plt.savefig(savepath, dpi=300)
    return fig

def lambda_decision(result_dict, k=3, savepath=None, fig_width=8, fig_height=4):
    # 20231018: change k from 2 to 3
    # Revised for compatibility with 1-D arrays of lmdb_range, loss_mean, and pvl_recode
    lmbd_range = np.exp(result_dict['log_lambda_range'])
    loss_recode = result_dict['loss_history']
    pvl_recode = result_dict['Prevalence']
    pvl_recode = pvl_recode * 100  # prevalence in percentage
        
    xs = np.log(lmbd_range); ys = loss_recode  # np.log(loss_recode)
    fig, ax1 = plt.subplots(figsize = (fig_width, fig_height))
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
        # ax1.plot([xs[msk][0], xs[msk][-1]], [ys_sl[msk][0], ys_sl[msk][-1]], color='r', zorder=5, linewidth = 2)
        
    
    segth = []; count = 0
    for i in range(len(dys_dt)):
        if dys_dt[i] not in segth:
            segth.append(dys_dt[i])
            count += 1
        if count == 1:
            selected_lambda = xs[i]
    
    ax1.errorbar(lmbd_range, loss_recode, marker='o', c='#33CCFF', linestyle='--', label ='BCE loss'); ax1.legend(loc='upper left')
    ax2.errorbar(lmbd_range, pvl_recode, marker='o', c='#FFAA33', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set(xscale="log"); # ax1.set(yscale="log")
    selected_lambda = np.exp(selected_lambda); plt.axvline(x=selected_lambda, color = 'black', linestyle=':')
    ax1.set_xlabel(f"lambda (optimal: {selected_lambda})")
    ax1.set_ylabel("loss"); ax2.set_ylabel("Prevalence (%)")
    ax2.set_ylim(0, 120)
    # plt.show()
    if savepath:
        plt.savefig(savepath, dpi=300)

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


