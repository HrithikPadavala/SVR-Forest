import sampler
import time
import Tree
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score

t0 = time.time()

data_name = 'balance'


def performance_stats(TP, FP, FN, TN):
    TPR = TP / (TP+FN)
    if TP+FP > 0:
        precision = TP / (TP+FP)
    else:
        precision = 0
    accuracy = (TP+TN) / (TP+FP+FN+TN)
    TNR = TN / (TN+FP)
    G_mean = np.sqrt(TPR*TNR)
    if precision > 0:
        F_measure = 2*TPR*precision / (TPR+precision)
    else:
        F_measure = 0
    return np.array([accuracy, precision, TPR, TNR, F_measure, G_mean])


def divide(n, m):
    '''Function to divide n samples into m roughly equal folds.'''
    n_low = n // m
    out = np.ones(m, dtype=int) * n_low
    remain = n % m
    out[0:remain] = out[0:remain] + 1
    return out


def id_divide(ids, m):
    '''Function to divide id_seq into m roughly equal folds.'''
    n = len(ids)
    n_in_folds = divide(n, m)
    id_lst = [None]*m
    loc = 0
    for i in range(m):
        loc_new = loc + n_in_folds[i]
        id_lst[i] = ids[loc:loc_new]
        loc = loc_new
    return id_lst


def experiment_runner(ids, Xall, Yall,d, n_cv, n_cv_out, pen_lst, weight, c0, alpha_lst):

    X = Xall[ids, :]
    Y = Yall[ids]
    n = len(Y)
    
    n1 = np.int_(np.sum(Y))
    n0 = n - n1
    n1_divide = divide(n1, n_cv_out)
    n0_divide = divide(n0, n_cv_out)
    # all variables starting with "id" are indices of X and Y (not Xall or Yall)
    id_1 = np.flatnonzero(Y)
    id_0 = np.flatnonzero(Y == 0)
    TPFP_svr = np.zeros(2)
    TPFP_svr_select = np.zeros(2)
    TPFP_duplicate = np.zeros(2)
    TPFP_SMOTE = np.zeros(2)
    TPFP_BSMOTE = np.zeros(2)
    TPFP_ADASYN = np.zeros(2)

    for test_fold_no in range(n_cv_out):
        id_1test_start = np.int_(np.sum(n1_divide[0:test_fold_no]))
        id_0test_start = np.int_(np.sum(n0_divide[0:test_fold_no]))
        id_1test = id_1[id_1test_start:(
            id_1test_start+n1_divide[test_fold_no])]
        id_1train = np.delete(id_1, np.arange(
            id_1test_start, id_1test_start+n1_divide[test_fold_no]))
        id_0test = id_0[id_0test_start:(
            id_0test_start+n0_divide[test_fold_no])]
        id_0train = np.delete(id_0, np.arange(
            id_0test_start, id_0test_start+n0_divide[test_fold_no]))
        n1train = len(id_1train)
        n1test = len(id_1test)
        n0train = len(id_0train)
        n0test = len(id_0test)
        id_1train_cv = id_divide(id_1train, n_cv)
        id_0train_cv = id_divide(id_0train, n_cv)
        id_cv = [None] * n_cv
        for k in range(n_cv):
            id_cv[k] = np.concatenate((id_1train_cv[k], id_0train_cv[k]))
        id_train = np.concatenate((id_1train, id_0train))
        Xtrain = X[id_train, :]
        Ytrain = Y[id_train]
        id_test = np.concatenate((id_1test, id_0test))
        Xtest = X[id_test, :]
        Ytest = Y[id_test]

        '''SVR tree'''
        F_lst = np.zeros(len(pen_lst))
        for j in range(len(pen_lst)):
            TP = 0
            FP = 0
            for k in range(n_cv):
                id_cv_copy = id_cv.copy()
                del id_cv_copy[k]
                id_temp = np.concatenate(id_cv_copy)
                Xtrain_temp = X[id_temp, :]
                Ytrain_temp = Y[id_temp]
                Xtest_temp = X[id_cv[k], :]
                Ytest_temp = Y[id_cv[k]]
                tr_svr = Tree.tree()
                tr_svr.fit_sv(Xtrain_temp, Ytrain_temp,d,
                              pen_lst[j], weight=weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
                Y_pred_temp = tr_svr.predict(Xtest_temp)
                TP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp == 0)])
            if TP > 0: 
                tpr = TP / n1train
                precision = TP / (TP+FP)
                F_lst[j] = 2*tpr*precision / (tpr+precision)
        para_id = np.argmax(F_lst)
        tr_svr = Tree.tree()
        tr_svr.fit_sv(Xtrain, Ytrain,d, pen_lst[para_id], weight=weight,
                      feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
        Y_pred = tr_svr.predict(Xtest)
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        
        FP = np.sum(Y_pred[np.flatnonzero(Ytest == 0)])
        TPFP_svr = TPFP_svr + np.array([TP, FP])
    
    #print(TPFP_svr[0])
    #D=D+TPFP_svr[0]
    #print(D)
    results_svr = performance_stats(
        TPFP_svr[0], TPFP_svr[1], n1-TPFP_svr[0], n0-TPFP_svr[1])
    
    return results_svr,TPFP_svr[0]
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common
class RandomForest:
    
    def __init__(self, n_trees=25, min_samples_split=2,
                 max_depth=100, n_feats=None, weight=1):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.weight = weight
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        global D
        TPFP_svr = np.zeros(2)
        for i in range(self.n_trees):
            tr_svr = Tree.tree()
            X_samp, y_samp = bootstrap_sample(X, y)
            F_lst = np.zeros(len(pen_lst))
            for j in range(len(pen_lst)):
                TP=0
                FP=0
                tr_svr = Tree.tree()
                tr_svr.fit_sv(X_samp, y_samp, D, pen_lst[j], weight=self.weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3),c0=4)
                Y_pred_temp = tr_svr.predict(X_samp)
                TP += np.sum(Y_pred_temp[np.flatnonzero(y_samp)])
                FP += np.sum(Y_pred_temp[np.flatnonzero(y_samp==0)])
            if TP > 0:
                tpr = TP / n1train
                precision = TP / (TP+FP)
                F_lst[j] = 2*tpr*precision / (tpr+precision)
            para_id = np.argmax(F_lst)
            tr_svr = Tree.tree()
            tr_svr.fit_sv(X_samp, y_samp, D, pen_lst[para_id], weight=self.weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
            Y_pred = tr_svr.predict(X_test)
            TP = np.sum(Y_pred[np.flatnonzero(y_test)])
        
            FP = np.sum(Y_pred[np.flatnonzero(y_test == 0)])
            TPFP_svr = TPFP_svr + np.array([TP, FP])
            D = D+TPFP_svr[0]
            self.trees.append(tr_svr)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

D=0     # D is the multiClassRegD variable
da0 = pd.read_table('balance-scale.data', sep=',', header=None)
   
#print(da0)
#da_pima=pd.read_csv('Exasens.csv', sep=';')
#da0=da_pima.values
#da1=pd.read_table('xab.dat', sep=' ', header=None)
da = np.delete(da0.values,-1 , axis=0)
print(da)
n, d = np.shape(da)
d = d-1
print(n)
print(d)
t1 = time.time()
print('head time: '+str(t1-t0))
foutput=[]
for iter in range(4):
    print("value of d main ="+str(D))
    Xall = da[:, 0:d]
    minClass=''
    if iter==0 or iter==3:
        minClass='L'
    elif iter==1 or iter==4:
        minClass='B'

    
    n1_ind = np.flatnonzero(da[:, d] == minClass)
    Yall = np.zeros(n, dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1
    seednum = 40
    np.random.seed(seednum)
    nexps = 1
    n1 = len(np.flatnonzero(Yall))
    n0 = n - n1
    id_permutes = np.zeros((nexps, n), dtype=int)
    for i in range(nexps):
        id_permutes[i, :] = np.random.permutation(n)
    id_filename = data_name+'_seed'+str(seednum)+'_ids'
    np.save(id_filename, id_permutes)
    n_cv = 5
    n_cv_out = 3
    train_ratio = 1 - 1/n_cv_out
    n1train = np.int_(n1*train_ratio)
    n0train = np.int_(n0*train_ratio)

# The below alpha_lst is for common datasets
    alpha_lst = np.array([0, 1/256, 1/128, 1/64, 1/32, 1/16, 0.125, 0.177, 0.25, 0.35, 0.5, 0.71, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11,
                     16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024, 1450, 2048, 2896, 4096]) * 10**(-3) * (n*train_ratio)**(-1/3)

    pen_lst = np.array([0, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89,
                   128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * (n0train+n1train)**(-1/3)
    weight = times+1
    c0 = 4
    inputs = [None]*nexps
    print('Run experiments for '+str(data_name))
    '''for i in range(nexps):
        inputs[i] = (id_permutes[i, :], Xall, Yall,D, n_cv,
                 n_cv_out, pen_lst, weight, c0, alpha_lst)'''
    X_train, X_test, y_train, y_test = train_test_split(Xall, Yall, test_size=0.2, random_state=0)
        
    model = RandomForest(weight=times+1)
    res = model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print('Accuracy:',accuracy_score(y_test, preds))
    print('ROC AUC:',roc_auc_score(y_test, preds))
    print('F1:',f1_score(y_test, preds))
    print('Precision:',precision_score(y_test, preds))
    print('Recall :',recall_score(y_test, preds) )